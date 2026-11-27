"""Fit a breast scorer with lesion-type supervision on frozen cross-fitted native proposals."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from medimage.auc_mammo_mil import DIMENSIONS, RECIPES, BreastMIL, infer
from medimage.auc_selection import save_scores
from medimage.auc_train import transfer
from medimage.common import checksum, load_manifest, read_json, write_json
from medimage.mammo_native import PIXEL_FORMAT
from medimage.radiography_train import save_checkpoint, validation_metrics


def load_proposals(source, frame, labels, development, validation, saved):
    config = saved['config']
    folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=config['seed'])
    oof = (np.zeros((len(development), 2, 19, DIMENSIONS), np.float16),
           np.zeros((len(development), 2, 19), bool))
    banks, provenance, versions = [], [], set()
    for index, (_, check) in enumerate(folds.split(development, labels[development, 0], frame.iloc[development].patient)):
        path = source / f'detector-{index + 1}.pt'
        bank_path = source / f'detector-{index + 1}-proposals.npz'
        detector = torch.load(path, map_location='cpu', weights_only=True)
        detector_hash = checksum(path)
        expected = {**config, 'fold': index, 'held_rows': development[check].tolist()}
        weights = saved['detectors'][index]
        if (detector['config'] != expected or detector['model'].keys() != weights.keys()
                or any(not torch.equal(value, weights[key]) for key, value in detector['model'].items())):
            raise ValueError('The proposal detector was trained on different cases or has different weights')
        with np.load(bank_path, allow_pickle=False) as recorded:
            if (recorded['detector_sha256'].item() != detector_hash
                    or not np.array_equal(recorded['development'], development[check])
                    or not np.array_equal(recorded['validation'], validation)):
                raise ValueError('Cached proposals do not match this detector or patient split')
            versions.add(recorded['feature_recipe'].item())
            oof[0][check], oof[1][check] = recorded['train_bag'], recorded['train_mask']
            bags, mask = recorded['val_bag'], recorded['val_mask']
            if bags.shape != (len(validation), *oof[0].shape[1:]) or mask.shape != bags.shape[:-1]:
                raise ValueError('The proposal tensor has unexpected views or patch dimensions')
            banks.append((bags, mask))
        provenance.append({'detector_sha256': detector_hash, 'bank_sha256': checksum(bank_path)})
    if len(versions) != 1:
        raise ValueError('Expected one extraction recipe for the three patient folds')
    for bag, mask in (oof, *banks):
        if not np.isfinite(bag).all() or not mask[:, :, 0].any(axis=1).all():
            raise ValueError('Every breast must have finite proposals and at least one present view')
    return oof, banks, provenance, versions.pop()


def rank_loss(logits, truth, included):
    positive = included & truth
    negative = included & ~truth
    difference = logits[positive, None] - logits[None, negative]
    return F.softplus(-difference).sum() / max(1, difference.numel())


class TrainingHead(nn.Module):
    def __init__(self, malignancy, lesion_type):
        super().__init__()
        self.malignancy = malignancy
        self.lesion_type = lesion_type

    def forward(self, embedding):
        return self.malignancy(embedding), self.lesion_type(embedding)


def train(args):
    torch.set_num_threads(4)
    frame, audit = load_manifest(args.cohort)
    source = read_json(args.source / 'result.json')
    best_file = args.source / 'best.pt'
    parent_sha = checksum(best_file)
    saved = torch.load(best_file, map_location='cpu', weights_only=True)
    config = saved['config']
    if (source['best_sha256'] != parent_sha or source['config'] != config or len(saved['detectors']) != 3
            or config['manifest'] != audit['manifest_sha256'] or config.get('pixel_format') != PIXEL_FORMAT
            or not config.get('fine_texture')):
        raise ValueError('Use the complete three-detector native fine-texture run for this cohort')
    development = np.flatnonzero(frame.split.eq('train').to_numpy())
    validation = np.flatnonzero(frame.split.eq('val').to_numpy())
    labels = frame[['positive']].to_numpy(np.float32, copy=True)
    types = frame.iloc[development].type.to_numpy()
    subtype = np.column_stack((np.isin(types, ['calc', 'calc+mass']), np.isin(types, ['mass', 'calc+mass'])))
    if not subtype.any(axis=1).all():
        raise ValueError('Unrecognized training lesion subtype')
    oof, banks, provenance, recipe = load_proposals(args.source, frame, labels, development, validation, saved)
    print('Verified cross-fitted proposals for 3 detectors', flush=True)
    config = {**config, 'epochs': args.epochs, 'head_seed': args.seed,
              'source_checkpoint_sha256': parent_sha, 'proposal_recipe_sha256': recipe, 'detector_provenance': provenance,
              'subtype_supervision': {'weight': .25, 'global_rank_weight': .17, 'within_type_weights': [0., .12]}}
    args.output.mkdir(parents=True, exist_ok=True)
    best, history = (-1., -1.), []
    for phase, within in enumerate(config['subtype_supervision']['within_type_weights']):
        for index, (width, dropout, rate) in enumerate(RECIPES):
            seed = args.seed + 500 + index
            torch.manual_seed(seed)
            options = {'width': width, 'dropout': dropout, 'contextual': True}
            model = BreastMIL(**options).cuda()
            subtype_head = nn.Linear(3 * width, 2).cuda()
            optimizer = torch.optim.AdamW([*model.parameters(), *subtype_head.parameters()], lr=rate, weight_decay=.02)
            training_head = TrainingHead(model.head, subtype_head)
            for epoch in range(args.epochs):
                model.head = training_head
                model.train()
                order = np.random.default_rng(seed + epoch).permutation(len(development))
                for start in range(0, len(order), 64):
                    rows = order[start:start + 64]
                    logits, lesion_logits = model(transfer(oof[0], rows), transfer(oof[1], rows))
                    truth = transfer(labels, development[rows])
                    types = transfer(subtype, rows)
                    positive = truth[:, 0] > .5
                    loss = F.binary_cross_entropy_with_logits(logits, truth)
                    loss = loss + .25 * F.binary_cross_entropy_with_logits(lesion_logits, types.float())
                    loss = loss + .17 * rank_loss(logits[:, 0], positive, torch.ones_like(positive))
                    if within:
                        within_loss = torch.stack([rank_loss(logits[:, 0], positive, group) for group in types.T]).mean()
                        loss = loss + within * within_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                model.head = training_head.malignancy
                scores = np.mean([infer(model, bank) for bank in banks], axis=0, dtype=np.float64)
                metrics = validation_metrics(labels[validation], scores)
                summary = {'recipe': 1 + phase * len(RECIPES) + index, 'epoch': epoch + 1, 'lr': rate,
                           'within_type_rank_weight': within, **options, **metrics}
                history.append(summary)
                rating = (metrics['macro_auroc'], metrics['macro_auprc'])
                if rating > best:
                    best = rating
                    save_checkpoint(args.output / 'best.pt', {'config': config, 'head': options,
                                                                'model': model.state_dict(), 'detectors': saved['detectors']})
                    save_scores(args.output / 'validation.npz', rows=validation, logits=scores)
                print(summary, flush=True)
            write_json(args.output / 'history.json', {'config': config, 'epochs': history})
    write_json(args.output / 'result.json', {'config': config, 'epochs': history,
                                            'best_sha256': checksum(args.output / 'best.pt'),
                                            'validation_sha256': checksum(args.output / 'validation.npz')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort', 'source', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=5323)
    parser.add_argument('--epochs', type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('Use a positive epoch count')
    train(args)


if __name__ == '__main__':
    main()
