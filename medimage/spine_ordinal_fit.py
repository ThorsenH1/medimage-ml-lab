"""Fit two ordinal severity boundaries using the leakage-safe MRI feature bank."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from medimage.auc_spine import auc_cells, interval
from medimage.common import checksum, read_json, write_json
from medimage.search_spine import SpineFeatureHead, SpineRecipe, loss_for
from medimage.search_spine_features import DEEP_DIM, TEXTURE_DIM
from medimage.spine_geometry_probe import classification_summary
from medimage.spine_localize import SpineCache, read_folds
from medimage.spine_metrics import read_submission, read_targets, score, submission

COMPACT = SpineRecipe('conv', 'hybrid', 96, .0008, .012, .28, .16, .65, .85)
CONTEXT = SpineRecipe('attention', 'deep', 96, .0007, .010, .33, .14, .60, .85)
RECIPES = ((COMPACT, 0.), (COMPACT, .18), (COMPACT, .36), (CONTEXT, .18))
BLEND_FRACTION = .25


def log_probs(boundaries: torch.Tensor) -> torch.Tensor:
    abnormal, severe_given_abnormal = boundaries.float().unbind(dim=-1)
    abnormal_log = F.logsigmoid(abnormal)
    return torch.stack((F.logsigmoid(-abnormal),
                        abnormal_log + F.logsigmoid(-severe_given_abnormal),
                        abnormal_log + F.logsigmoid(severe_given_abnormal)), dim=-1)


def boundary_rank(logits: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    comparable = positive[:, None] & negative[None, :]
    errors = F.softplus(logits[None, :] - logits[:, None]) * comparable
    support = comparable.sum(dim=(0, 1))
    by_target = errors.sum(dim=(0, 1)) / support.clamp_min(1)
    return by_target.sum() / (support > 0).sum().clamp_min(1)


def objective(output: torch.Tensor, truth: torch.Tensor, recipe: SpineRecipe, strength: float) -> torch.Tensor:
    base = loss_for(log_probs(output), truth, recipe)
    if not strength:
        return base
    abnormal = boundary_rank(output[..., 0], truth > 0, truth == 0)
    severe = boundary_rank(output[..., 1], truth == 2, truth == 1)
    return base + strength * (abnormal + severe) / 2


def fit_model(recipe, strength, pixels, metadata, labels, tr, va, args, seed):
    torch.manual_seed(seed)
    model = SpineFeatureHead(recipe)
    model.readout[-1] = nn.Linear(recipe.width, 2)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.lr, weight_decay=recipe.decay)
    training = torch.as_tensor(tr, device=args.device)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(training), device=args.device)
        for batch in training[order].split(args.batch):
            optimizer.zero_grad(set_to_none=True)
            boundaries = model(pixels[batch], metadata[batch])
            loss = objective(boundaries, labels[batch], recipe, strength)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite ordinal training objective at epoch {epoch + 1}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.)
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        predicted = [log_probs(model(pixels[batch], metadata[batch])).exp().cpu().numpy()
                     for batch in torch.as_tensor(va, device=args.device).split(args.batch)]
    return model, np.concatenate(predicted)


def load_fold(cache, assignments, directory, fold, digest):
    signature = read_json(directory / 'features.json')
    inner = (fold + 1) % 5
    if (signature['manifest'] != digest or signature['fold'] != fold
            or signature['inner_validation_fold'] != inner):
        raise ValueError('MRI features do not match the frozen nested split')
    with np.load(directory / 'geometry.npz', allow_pickle=False) as source:
        metadata = dict(source)
    inner_train = np.flatnonzero(~np.isin(assignments, [fold, inner]))
    known_folds = np.arange(5)[None]
    allowed_inner = (~np.isin(known_folds, [fold, inner])) & (known_folds != assignments[inner_train, None])
    allowed_outer = (known_folds != fold) & (known_folds != assignments[:, None])
    if (not np.array_equal(metadata['studies'], cache.studies)
            or not np.array_equal(metadata['inner_rows'], inner_train)
            or not np.array_equal(metadata['inner_provenance'], allowed_inner)
            or not np.array_equal(metadata['outer_provenance'], allowed_outer)):
        raise ValueError('The MRI feature localizer used an unavailable study or fold')
    outer = np.load(directory / 'outer.npy', mmap_mode='r', allow_pickle=False)
    nested = np.load(directory / 'inner.npy', mmap_mode='r', allow_pickle=False)
    feature_shape = (30, DEEP_DIM + TEXTURE_DIM)
    if outer.shape != (len(assignments), *feature_shape) or nested.shape != (len(inner_train), *feature_shape):
        raise ValueError('MRI feature geometry differs from the recorded format')
    return signature, metadata, outer, nested


def evaluate(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    cache = SpineCache.load(args.cache)
    assignments = read_folds(args.folds, cache.studies)
    digest = checksum(args.folds)
    truth = read_targets(args.root, cache.studies)
    prior = read_submission(args.reference, cache.studies)
    recipes = [{'features': asdict(recipe), 'ranking_weight': weight} for recipe, weight in RECIPES]
    plan = {'folds_sha256': digest, 'reference_sha256': checksum(args.reference), 'epochs': args.epochs,
            'batch': args.batch, 'seed': args.seed, 'device': args.device, 'threads': args.threads,
            'torch': str(torch.__version__), 'trainer_sha256': checksum(Path(__file__)), 'recipes': recipes,
            'choice': 'highest inner macro AUROC, then lowest inner weighted log loss',
            'ordinal_fraction': BLEND_FRACTION,
            'features': [checksum(args.features / f'fold-{fold}' / 'features.json') for fold in range(5)]}
    plan_path = args.output / 'comparison-plan.json'
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError('Use a separate output for a different ordinal experiment')
    write_json(plan_path, plan)
    plan_digest = checksum(plan_path)
    estimates = np.empty_like(prior)
    evidence = {}
    target_labels = torch.as_tensor(truth, device=args.device)
    for fold in range(5):
        outer_train, outer_valid = np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)
        checkpoint = args.output / f'fold-{fold}.pt'
        heldout = args.output / f'fold-{fold}-oof.csv'
        result = args.output / f'fold-{fold}-result.json'
        if result.exists():
            completed = read_json(result)
            if (completed['plan_sha256'] != plan_digest or completed['checkpoint_sha256'] != checksum(checkpoint)
                    or completed['predictions_sha256'] != checksum(heldout)):
                raise ValueError('Stored ordinal probabilities or weights differ from their locked fold')
            estimates[outer_valid] = read_submission(heldout, cache.studies[outer_valid])
            evidence[str(fold)] = completed
            print(f'Reusing completed ordinal fold {fold}', flush=True)
            continue
        folder = args.features / f'fold-{fold}'
        signature, meta, outer, inner = load_fold(cache, assignments, folder, fold, digest)
        inn_train = meta['inner_rows']
        inn_valid = np.flatnonzero(assignments == signature['inner_validation_fold'])
        dev = np.r_[inn_train, inn_valid]
        pixels = torch.as_tensor(np.concatenate((inner, outer[inn_valid])), dtype=torch.float32, device=args.device)
        quality = torch.as_tensor(np.concatenate((meta['inner_meta'], meta['outer_meta'][inn_valid])),
                                  device=args.device)
        inner_targets = target_labels[dev]
        train_rows, validation_rows = np.arange(len(inn_train)), np.arange(len(inn_train), len(dev))
        trials = []
        for index, (recipe, rank_weight) in enumerate(RECIPES):
            _, predicted = fit_model(recipe, rank_weight, pixels, quality, inner_targets, train_rows,
                                     validation_rows, args, args.seed + fold)
            labels = truth[inn_valid]
            auc = float(np.nanmean(auc_cells(labels, predicted)[0]))
            trials.append({'recipe': index, 'auroc': auc, 'loss': score(labels, predicted)['score']})
            print(f'Outer fold {fold}, inner recipe {index}, AUROC {auc:.5f}', flush=True)
        best = max(trials, key=lambda trial: (trial['auroc'], -trial['loss']))['recipe']
        del pixels, quality
        pixels = torch.as_tensor(np.array(outer), dtype=torch.float32, device=args.device)
        quality = torch.as_tensor(meta['outer_meta'], device=args.device)
        model, predictions = fit_model(*RECIPES[best], pixels, quality, target_labels, outer_train,
                                       outer_valid, args, args.seed + fold)
        estimates[outer_valid] = predictions
        torch.save({'weights': model.state_dict(), 'recipe': recipes[best], 'features': signature}, checkpoint)
        submission(cache.studies[outer_valid], predictions).to_csv(heldout, index=False)
        completed = {'selected': best, 'inner_trials': trials, 'studies': len(outer_valid),
                     'plan_sha256': plan_digest, 'checkpoint_sha256': checksum(checkpoint),
                     'predictions_sha256': checksum(heldout)}
        write_json(result, completed)
        evidence[str(fold)] = completed
        print(f'Outer fold {fold}, selected recipe {best}, scored {len(outer_valid)} held-out studies', flush=True)
        del model, pixels, quality

    combined = (1 - BLEND_FRACTION) * prior + BLEND_FRACTION * estimates
    summary = {'plan_sha256': plan_digest, 'studies': len(cache.studies), 'folds': evidence}
    for name, probabilities in (('reference', prior), ('ordinal', estimates), ('blend', combined)):
        summary[name] = {**classification_summary(truth, probabilities),
                         **interval(truth, probabilities, assignments, args.bootstrap)}
        print(name, summary[name], flush=True)
    submission(cache.studies, estimates).to_csv(args.output / 'ordinal-oof.csv', index=False)
    submission(cache.studies, combined).to_csv(args.output / 'blend-oof.csv', index=False)
    write_json(args.output / 'metrics.json', summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cache', 'folds', 'features', 'reference', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=48)
    parser.add_argument('--seed', type=int, default=5831)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if min(args.threads, args.epochs, args.batch, args.bootstrap) < 1:
        parser.error('Threads, epochs, batch size and bootstrap count must be positive')
    evaluate(args)


if __name__ == '__main__':
    main()
