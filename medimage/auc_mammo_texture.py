from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from medimage.auc_train import store, transfer
from medimage.common import checksum, write_json
from medimage.radiography_train import save_checkpoint, validation_metrics


def distribution(values):
    sample = values[::3]
    if len(sample):
        return [sample.mean(), sample.std(), *np.quantile(sample, [.1, .5, .9, .98, .997])]
    return [0.] * 7


def spot_features(image, tissue, scale):
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (scale, scale)))
    contrast = cv2.subtract(image, opened)
    maxima = cv2.dilate(contrast, np.ones((scale, scale), np.uint8)) == contrast
    y, x = np.nonzero(maxima & tissue & (contrast > 3))
    response = contrast[y, x].astype(np.float32)
    order = np.argsort(-response, kind='stable')[:64]
    coords = np.column_stack((x[order] / image.shape[1], y[order] / image.shape[0]))
    values = response[order]
    if len(coords) <= 1:
        return [len(coords), *distribution(values), *([0.] * 8)]
    radius = np.sqrt(np.square(coords[:, None] - coords[None]).sum(2))
    np.fill_diagonal(radius, np.inf)
    neighbors = radius.min(1)
    return [len(coords), *distribution(values), *distribution(neighbors),
            np.max((radius < .055).sum(0))]


def texture_features(image: np.ndarray) -> np.ndarray:
    tissue = cv2.erode((image > 10).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    gradients = []
    for stride in (1, 4, 16, 40):
        horizontal = tissue[:, stride:] & tissue[:, :-stride]
        vertical = tissue[stride:] & tissue[:-stride]
        texture_x = np.abs(image[:, stride:].astype(np.float32) - image[:, :-stride])[horizontal]
        texture_y = np.abs(image[stride:].astype(np.float32) - image[:-stride])[vertical]
        gradients.extend([*distribution(texture_x), *distribution(texture_y)])
    spots = [value for width in (3, 7, 15, 31) for value in spot_features(image, tissue, width)]
    histogram = np.bincount(image[tissue] // 16, minlength=16).astype(np.float32)
    histogram /= max(histogram.sum(), 1)
    return np.asarray([tissue.mean(), *distribution(image[tissue]), *histogram, *gradients, *spots], dtype=np.float32)


TEXTURE_DIMENSIONS = 144


def breast_features(arrays, rows, workers: int = 4) -> np.ndarray:
    cv2.setNumThreads(1)

    def one(row):
        views = arrays['views'][row]
        features = [texture_features(arrays['pixels'][row, view]) if available else
                    np.zeros(TEXTURE_DIMENSIONS, np.float32) for view, available in enumerate(views)]
        values = np.stack(features)
        mean = values.sum(0) / views.sum()
        return np.concatenate((values.flatten(), mean, views)).astype(np.float32)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        result = np.stack(list(executor.map(one, rows)))
    if not np.isfinite(result).all():
        raise ValueError('Mammography texture bank has a nonfinite feature')
    return result


class TextureClassifier(nn.Module):
    def __init__(self, dimensions, width, dropout):
        super().__init__()
        self.register_buffer('center', torch.zeros(dimensions))
        self.register_buffer('scale', torch.ones(dimensions))
        self.linear = nn.Linear(dimensions, 1)
        self.context = nn.Sequential(nn.Linear(dimensions, width), nn.LayerNorm(width), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(width, 1))
        self.dropout = nn.Dropout(dropout / 2)

    def forward(self, x):
        normalized = self.dropout(((x - self.center) / self.scale).clamp(-12, 12))
        return self.linear(normalized) + self.context(normalized)


@torch.inference_mode()
def predict(model, values):
    model.eval()
    return model(torch.from_numpy(values).cuda()).float().cpu().numpy()


def fit(args):
    torch.set_num_threads(4)
    frame, audit, arrays, pixel_digest = store('mammo', args.cohort, args.pixels)
    train = np.flatnonzero(frame.split.eq('train').to_numpy())
    validation = np.flatnonzero(frame.split.eq('val').to_numpy())
    labels = frame[['positive']].to_numpy(np.float32, copy=True)
    train_values = breast_features(arrays, train, args.workers)
    print(f'Extracted texture distributions for {len(train_values)} training breasts', flush=True)
    validation_values = breast_features(arrays, validation, args.workers)
    center = np.median(train_values, axis=0)
    scale = train_values.std(0).clip(.015)
    dimensions = train_values.shape[1]
    config = {'task': 'mammo', 'backbone': 'mammo-texture', 'manifest': audit['manifest_sha256'],
              'pixel_sha256': pixel_digest, 'dimensions': dimensions, 'epochs': args.epochs, 'seed': args.seed}
    args.output.mkdir(parents=True, exist_ok=True)
    best, histories = (-1., -1.), []
    recipes = ((32, .28, 3e-4, .05), (64, .45, 5e-4, .06), (96, .55, 6e-4, .10),
               (64, .35, 1e-3, .13))
    for recipe, (width, dropout, lr, decay) in enumerate(recipes):
        torch.manual_seed(args.seed + recipe)
        model = TextureClassifier(dimensions, width, dropout).cuda()
        model.center.copy_(torch.from_numpy(center))
        model.scale.copy_(torch.from_numpy(scale))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
        local_best = -1.
        for epoch in range(args.epochs):
            model.train()
            order = np.random.default_rng(args.seed + 100 * recipe + epoch).permutation(len(train))
            for start in range(0, len(order), 64):
                rows = order[start:start + 64]
                logits = model(transfer(train_values, rows))
                loss = F.binary_cross_entropy_with_logits(logits, transfer(labels, train[rows]))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            scores = predict(model, validation_values)
            metrics = validation_metrics(labels[validation], scores)
            local_best = max(local_best, metrics['macro_auroc'])
            summary = {'recipe': recipe + 1, 'width': width, 'dropout': dropout, 'lr': lr, 'weight_decay': decay,
                       'epoch': epoch + 1, **metrics}
            histories.append(summary)
            rank = (metrics['macro_auroc'], metrics['macro_auprc'])
            if rank > best:
                best = rank
                options = {'dimensions': dimensions, 'width': width, 'dropout': dropout}
                save_checkpoint(args.output / 'best.pt', {'config': config, 'head': options, 'model': model.state_dict()})
                np.savez_compressed(args.output / 'validation.partial.npz', rows=validation, logits=scores)
                (args.output / 'validation.partial.npz').replace(args.output / 'validation.npz')
        print(f'Texture recipe {recipe + 1}: validation AUROC {local_best:.6f}', flush=True)
    write_json(args.output / 'result.json', {'config': config, 'epochs': histories,
                                            'best_sha256': checksum(args.output / 'best.pt'),
                                            'validation_sha256': checksum(args.output / 'validation.npz')})


def main():
    parser = argparse.ArgumentParser(description='Train and compare breast-level texture feature classifiers')
    for name in ('cohort', 'pixels', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=4927)
    args = parser.parse_args()
    if min(args.epochs, args.workers) < 1:
        parser.error('Use positive epoch and worker counts')
    fit(args)


if __name__ == '__main__':
    main()
