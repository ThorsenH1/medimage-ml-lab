from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from medimage.auc_patches import FINE_TEXTURE_SCALES, PATCH, SCALES, PatchNet, train_patch_epoch
from medimage.auc_train import store, transfer
from medimage.common import checksum, read_json, write_json
from medimage.radiography_train import save_checkpoint, validation_metrics


SIZE = 13
DIMENSIONS = 520
RECIPES = ((96, .35, .0008), (128, .48, .0005))


def patch_tokens(model, resized, coordinates):
    chunks = []
    for start in range(0, len(resized), 80):
        with torch.autocast('cuda', dtype=torch.float16):
            features = model.encode(resized[start:start + 80])
            logits = model.head(features).float()
        chunks.append(torch.cat((features.float(), logits.softmax(1)), dim=-1))
    return torch.cat((torch.cat(chunks), coordinates), dim=1)


@torch.inference_mode()
def extract(model: PatchNet, arrays, rows):
    model.eval()
    size = 19 if model.fine_texture else SIZE
    scales = FINE_TEXTURE_SCALES if model.fine_texture else SCALES
    bags = np.zeros((len(rows), 2, size, DIMENSIONS), np.float16)
    valid = np.zeros((len(rows), 2, size), bool)
    for index, row in enumerate(rows):
        for view in np.flatnonzero(arrays['views'][row]):
            image = transfer(arrays['pixels'], [row])[:, view:view + 1].float()
            height, width = image.shape[-2:]
            raw = F.interpolate(image, (PATCH, PATCH), mode='bilinear', align_corners=False, antialias=True)
            center = torch.tensor([[.5, .5, 1.]], device=image.device)
            bags[index, view, 0] = patch_tokens(model, raw, center)[0].half().cpu().numpy()
            valid[index, view, 0] = True
            sequences = []
            for window, stride in scales:
                tissue = F.avg_pool2d((image > 10).float(), window, stride, window // 2)
                included = tissue.flatten() > .18
                if not included.any():
                    continue
                iy, ix = torch.where(included.reshape(tissue.shape[-2:]))
                if model.fine_texture:
                    padded = F.pad(image[0, 0], [window // 2] * 4)
                    windows = padded.unfold(0, window, stride).unfold(1, window, stride)
                    patches = windows[iy, ix, None]
                else:
                    patches = F.unfold(image, window, padding=window // 2, stride=stride)[0].T
                    patches = patches[included].reshape(-1, 1, window, window)
                coord = torch.stack((ix * stride / width, iy * stride / height,
                                     torch.full_like(ix, window, dtype=torch.float32) / width), dim=1)
                x = F.interpolate(patches, (PATCH, PATCH), mode='bilinear', align_corners=False, antialias=True)
                sequences.append(patch_tokens(model, x, coord))
            if not sequences:
                continue
            tokens = torch.cat(sequences)
            p = tokens[:, 512:517]
            evidence = torch.stack((1 - p[:, 0], p[:, 1:3].sum(1), p[:, 3:].sum(1)))
            proposals = evidence.topk(min(8, len(p)), dim=1).indices.T.flatten().cpu().tolist()
            take = list(dict.fromkeys(proposals))[:size - 1]
            bags[index, view, 1:1 + len(take)] = tokens[take].half().cpu().numpy()
            valid[index, view, 1:1 + len(take)] = True
    if not valid.any((1, 2)).all() or not np.isfinite(bags).all():
        raise ValueError('Each breast must produce finite proposals for an available view')
    return bags, valid


class BreastMIL(nn.Module):
    def __init__(self, width: int, dropout: float, contextual: bool = False):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, width), nn.GELU(), nn.Dropout(dropout))
        self.info = nn.Linear(8, width, bias=False)
        self.view = nn.Parameter(torch.randn(2, 1, width) * .04)
        self.context = nn.MultiheadAttention(width, 4, dropout=.12, batch_first=True) if contextual else None
        self.context_norm = nn.LayerNorm(width) if contextual else None
        self.attention = nn.Sequential(nn.Linear(width, width // 2), nn.Tanh(), nn.Linear(width // 2, 1))
        self.head = nn.Sequential(nn.LayerNorm(3 * width), nn.Dropout(dropout), nn.Linear(3 * width, 1))

    def forward(self, bag, valid):
        projected = self.project(bag[..., :512].float()) + self.info(bag[..., 512:].float()) + self.view
        token = projected.flatten(1, 2)
        keep = valid.flatten(1)
        if self.context is not None:
            normalized = self.context_norm(token)
            update, _ = self.context(normalized, normalized, normalized, key_padding_mask=~keep,
                                     need_weights=False)
            token = token + update
        attention = self.attention(token).squeeze(-1).float().masked_fill(~keep, -torch.inf).softmax(1)
        pooled = (token * attention[..., None]).sum(1)
        peak = token.masked_fill(~keep[..., None], -torch.inf).amax(1)
        global_views = valid[:, :, 0, None]
        context = (projected[:, :, 0] * global_views).sum(1) / global_views.sum(1)
        return self.head(torch.cat((pooled, peak, context), dim=1))


@torch.inference_mode()
def infer(head, bank):
    head.eval()
    return np.concatenate([
        head(torch.from_numpy(bank[0][start:start + 64]).cuda(),
             torch.from_numpy(bank[1][start:start + 64]).cuda()).float().cpu().numpy()
        for start in range(0, len(bank[0]), 64)])


def detector_for(weights: Path, config: dict, state=None):
    model = PatchNet(weights, config.get('fine_texture', False)).cuda().to(memory_format=torch.channels_last)
    if model.weight_sha256 != config['weight_sha256']:
        raise ValueError('Detector ImageNet weights differ from the locked provenance')
    if state is not None:
        model.load_state_dict(state)
    return model


def predict(head, saved, arrays, rows, weights):
    scores = []
    for state in saved['detectors']:
        detector = detector_for(weights, saved['config'], state)
        scores.append(infer(head, extract(detector, arrays, rows)))
        del detector
    return np.mean(scores, axis=0, dtype=np.float64)


def train_detectors(args, frame, labels, arrays, development, validation, config):
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=args.seed)
    folds = list(splitter.split(development, labels[development, 0], frame.iloc[development].patient))
    boxes = np.load(args.pixels / 'lesions.npy', mmap_mode='r', allow_pickle=False)
    if (boxes[frame.split.ne('train').to_numpy()] >= 0).any():
        raise ValueError('Only development rows can contribute lesion boxes')
    size = 19 if config.get('fine_texture') else SIZE
    oof = (np.zeros((len(development), 2, size, DIMENSIONS), np.float16),
           np.zeros((len(development), 2, size), bool))
    states, held_banks = [], []
    for fold, (fit, check) in enumerate(folds):
        if set(frame.iloc[development[fit]].patient) & set(frame.iloc[development[check]].patient):
            raise ValueError('A patient straddles the proposal-training folds')
        local = args.output / f'detector-{fold + 1}.pt'
        run = {**config, 'fold': fold, 'held_rows': development[check].tolist()}
        torch.manual_seed(args.seed + 100 * fold)
        detector = detector_for(args.weights, config)
        if local.is_file():
            saved = torch.load(local, map_location='cpu', weights_only=True)
            if saved['config'] != run:
                raise ValueError('The proposal fold checkpoint has different training membership')
            detector.load_state_dict(saved['model'])
        else:
            optimizer = torch.optim.AdamW([{'params': detector.encoder.parameters(), 'lr': .00013},
                                          {'params': detector.head.parameters(), 'lr': .0005}],
                                          weight_decay=.004, fused=True)
            scaler = torch.amp.GradScaler('cuda')
            for epoch in range(args.patch_epochs):
                progress = train_patch_epoch(detector, arrays, boxes, development[fit],
                                             args.seed + 100 * fold + epoch, 3, 64, optimizer, scaler)
                print(f'Detector {fold + 1} epoch {epoch + 1}: {progress}', flush=True)
            save_checkpoint(local, {'config': run, 'model': detector.state_dict()})
        states.append({name: parameter.cpu() for name, parameter in detector.state_dict().items()})
        train_bag, train_mask = extract(detector, arrays, development[check])
        oof[0][check], oof[1][check] = train_bag, train_mask
        held_banks.append(extract(detector, arrays, validation))
        print(f'Detector {fold + 1}: extracted unseen training and validation proposals', flush=True)
        del detector
    return oof, held_banks, states


def fit(args):
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    frame, audit, arrays, pixel_hash = store('mammo', args.cohort, args.pixels)
    labels = frame[['positive']].to_numpy(np.float32, copy=True)
    train = np.flatnonzero(frame.split.eq('train').to_numpy())
    validation = np.flatnonzero(frame.split.eq('val').to_numpy())
    config = {'task': 'mammo', 'backbone': 'patch-mil', 'seed': args.seed, 'epochs': args.epochs,
              'patch_epochs': args.patch_epochs, 'manifest': audit['manifest_sha256'],
              'pixel_sha256': pixel_hash, 'weight_sha256': PatchNet(args.weights).weight_sha256}
    if args.fine_texture:
        config['fine_texture'] = True
    oof, banks, states = train_detectors(args, frame, labels, arrays, train, validation, config)
    best, history = (-1., -1.), []
    for i, (width, dropout, lr) in enumerate(RECIPES):
        torch.manual_seed(args.seed + 500 + i)
        model = BreastMIL(width, dropout, contextual=args.fine_texture).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.02)
        for epoch in range(args.epochs):
            model.train()
            order = np.random.default_rng(args.seed + 500 * i + epoch).permutation(len(train))
            for start in range(0, len(order), 64):
                rows = order[start:start + 64]
                logits = model(transfer(oof[0], rows), transfer(oof[1], rows))
                truth = transfer(labels, train[rows])
                loss = F.binary_cross_entropy_with_logits(logits, truth)
                if args.fine_texture:
                    positive = truth[:, 0] > 0
                    difference = logits[positive, 0, None] - logits[~positive, 0][None]
                    loss = loss + .17 * F.softplus(-difference).sum() / max(1, difference.numel())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            val_scores = np.mean([infer(model, bank) for bank in banks], axis=0, dtype=np.float64)
            metrics = validation_metrics(labels[validation], val_scores)
            summary = {'recipe': i + 1, 'width': width, 'dropout': dropout, 'lr': lr,
                       'epoch': epoch + 1, **metrics}
            history.append(summary)
            rank = (metrics['macro_auroc'], metrics['macro_auprc'])
            if rank > best:
                best = rank
                head = {'width': width, 'dropout': dropout}
                if args.fine_texture:
                    head['contextual'] = True
                save_checkpoint(args.output / 'best.pt', {'config': config, 'head': head,
                                                            'model': model.state_dict(), 'detectors': states})
                np.savez_compressed(args.output / 'validation.partial.npz', rows=validation, logits=val_scores)
                (args.output / 'validation.partial.npz').replace(args.output / 'validation.npz')
            print(summary, flush=True)
        write_json(args.output / 'history.json', {'config': config, 'epochs': history})
    write_json(args.output / 'result.json', {'config': config, 'epochs': history,
                                            'best_sha256': checksum(args.output / 'best.pt'),
                                            'validation_sha256': checksum(args.output / 'validation.npz')})


def main():
    parser = argparse.ArgumentParser(description='Train mammography ROI and breast classifiers with group cross-fitting')
    for name in ('cohort', 'pixels', 'weights', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--patch-epochs', type=int, default=7)
    parser.add_argument('--seed', type=int, default=4731)
    parser.add_argument('--fine-texture', action='store_true',
                        help='Use fine-scale contrast channels, smaller proposals and cross-view attention')
    args = parser.parse_args()
    if min(args.epochs, args.patch_epochs) < 1:
        parser.error('Epoch counts must be positive')
    fit(args)


if __name__ == '__main__':
    main()
