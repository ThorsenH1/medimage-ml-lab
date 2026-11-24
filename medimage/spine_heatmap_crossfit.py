"""Build MRI regions with a heatmap localizer that holds out each study fold."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from medimage.common import checksum, read_json, write_json
from medimage.spine_heatmap import (DiscHeatmap, anchor_inputs, decode, fit_heatmaps, mm_metrics,
                                    observed_landmarks, pixel_grid)
from medimage.spine_localize import SpineCache, make_regions, read_folds, world_from_pixels


@torch.inference_mode()
def predict_heatmaps(model, inputs: np.ndarray, batch: int) -> np.ndarray:
    axes = pixel_grid(torch.device('cuda'))
    estimates = []
    with torch.autocast('cuda', dtype=torch.float16):
        for chunk in torch.from_numpy(inputs).split(batch):
            estimates.append(decode(model(chunk.cuda().float().div_(255.)), axes))
    return np.concatenate(estimates)


def anatomy_offsets(cache: SpineCache, xy: np.ndarray, geometry: np.ndarray, fit: np.ndarray) -> np.ndarray:
    anchor = world_from_pixels(xy[fit], geometry[fit])
    displacement = cache.points[fit] - np.tile(anchor, (1, 5, 1))
    displacement[np.abs(displacement) >= 90] = np.nan
    offsets = np.nanmedian(displacement, axis=0).astype(np.float32)
    if not np.isfinite(offsets).all():
        raise ValueError('A training fold does not contain enough anatomical coordinates for the atlas')
    return offsets


def crossfit(args: argparse.Namespace) -> None:
    torch.set_num_threads(4)
    cache = SpineCache.load(args.cache)
    folds = read_folds(args.folds, cache.studies)
    reference = read_json(args.reference / 'localization.json')
    fold_hash = checksum(args.folds)
    cache_hash = checksum(args.cache / 'audit.json')
    if reference['config']['folds_file_sha256'] != fold_hash or reference['config']['cache_sha256'] != cache_hash:
        raise ValueError('The frozen reference crops came from different studies')
    inputs, geometry = anchor_inputs(cache, np.arange(len(folds)))
    xy = observed_landmarks(cache, geometry)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {'folds_file_sha256': fold_hash, 'cache_sha256': cache_hash,
              'studies': len(folds), 'seed': args.seed, 'epochs': args.epochs,
              'batch': args.batch, 'lr': args.lr, 'method': 'cross-fitted disc heatmap',
              'heatmap_code_sha256': checksum(Path(__file__).with_name('spine_heatmap.py')),
              'crop_code_sha256': checksum(Path(__file__))}
    report_path = args.output / 'localization.json'
    results = {}
    if report_path.is_file():
        previous = read_json(report_path)
        if previous['config'] != config:
            raise ValueError('The output directory belongs to a different cross-fit configuration')
        results = previous['held_out_localization']
    for outer in args.outer:
        if str(outer) in results:
            if checksum(args.output / f'fold-{outer}.npz') != results[str(outer)]['region_sha256']:
                raise ValueError('A completed cross-fit region file changed')
            continue
        reference_path = args.reference / f'fold-{outer}.npz'
        with np.load(reference_path, allow_pickle=False) as saved:
            if not np.array_equal(saved['studies'], cache.studies) or not np.array_equal(saved['fold_per_study'], folds):
                raise ValueError('The reference regions are not aligned to these folds')
            old_meta = saved['metadata']
        trained_on = np.zeros((len(folds), 5), dtype=bool)
        coordinates = np.empty((len(folds), 30, 3, 5), np.float32)
        metadata = np.empty((len(folds), 30, 5), np.float32)
        checkpoints = {}
        held_metrics = None
        for source in range(5):
            excluded = tuple(sorted({outer, source}))
            fit = np.flatnonzero(~np.isin(folds, excluded))
            held = np.flatnonzero(folds == source)
            split_name = 'exclude-' + '-'.join(map(str, excluded))
            model_dir = args.output / 'models' / split_name
            checkpoint = model_dir / 'final.pt'
            model_audit = {'seed': args.seed + sum(7 ** index for index in excluded),
                           'epochs': args.epochs, 'batch': args.batch, 'lr': args.lr,
                           'folds_sha256': fold_hash, 'cache_sha256': cache_hash,
                           'heatmap_code_sha256': config['heatmap_code_sha256'],
                           'training_folds': sorted(set(folds[fit].tolist()))}
            if not checkpoint.is_file():
                print(f'MRI localization fit excluding {excluded}', flush=True)
                fit_heatmaps(inputs, xy, geometry, fit, np.empty(0, np.int64), model_audit, model_dir)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if saved['audit'] != model_audit:
                raise ValueError('Landmark checkpoint was fitted with different data or settings')
            network = DiscHeatmap().cuda().eval()
            network.load_state_dict(saved['state'])
            points = predict_heatmaps(network, inputs[held], args.batch)
            del network
            offsets = anatomy_offsets(cache, xy, geometry, fit)
            confidence = old_meta[held, :5, 1]
            coordinates[held], metadata[held], _ = make_regions(cache, held, geometry, points, confidence,
                                                               {'offsets': offsets})
            trained_on[held[:, None], np.unique(folds[fit])] = True
            checkpoints[str(source)] = checksum(checkpoint)
            if source == outer:
                held_metrics = mm_metrics(points, xy[held], geometry[held])
        expected = (np.arange(5)[None] != folds[:, None]) & (np.arange(5)[None] != outer)
        if not np.array_equal(trained_on, expected):
            raise ValueError('Each crop must hold out its source fold and its classifier evaluation fold')
        fold_file = args.output / f'fold-{outer}.npz'
        temporary = args.output / f'fold-{outer}.partial.npz'
        np.savez_compressed(temporary, studies=cache.studies, fold_per_study=folds,
                            trained_on=trained_on, region=coordinates, metadata=metadata)
        temporary.replace(fold_file)
        results[str(outer)] = {**held_metrics, 'region_sha256': checksum(fold_file), 'checkpoints': checkpoints}
        write_json(report_path, {'config': config, 'held_out_localization': results})
        print(f'MRI outer fold {outer} localization: {held_metrics}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'folds', 'reference', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--outer', type=int, choices=range(5), nargs='+', default=list(range(5)))
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=9409)
    args = parser.parse_args()
    if min(args.epochs, args.batch) < 1 or args.lr <= 0:
        parser.error('Epochs, batch size and learning rate must be positive')
    crossfit(args)


if __name__ == '__main__':
    main()
