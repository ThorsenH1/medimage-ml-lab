"""Evaluate fixed left-right test-time augmentation on the existing held-out MRI folds."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from medimage.auc_spine import interval
from medimage.common import checksum, read_json, write_json
from medimage.spine_data import STARTS
from medimage.spine_geometry_probe import classification_summary
from medimage.spine_localize import SpineCache, read_folds
from medimage.spine_metrics import read_submission, read_targets, submission
from medimage.spine_model import ROI_SIZE, SpineClassifier
from medimage.spine_train import RegionDataset, _load_fold, predict

SWAP = np.r_[0:5, 10:15, 5:10, 20:25, 15:20, 25:30]
AXIS_COSINE = .90


def reflection_axes(geometry: np.ndarray, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slots = coordinates[..., 0].astype(int)
    planes = geometry[np.arange(len(slots))[:, None, None], np.maximum(slots, 0), 1:, :]
    magnitudes = np.linalg.norm(planes, axis=-1, keepdims=True)
    vectors = np.divide(planes, magnitudes, out=np.zeros_like(planes), where=magnitudes > 0)
    sagittal_normal = np.abs(np.cross(vectors[:, :15, :, 0], vectors[:, :15, :, 1])[..., 0])
    axial_steps = np.abs(vectors[:, 15:, :, :, 0])
    axial_axis = np.argmax(axial_steps, axis=-1) + 1
    good = np.concatenate((sagittal_normal >= AXIS_COSINE, axial_steps.max(axis=-1) >= AXIS_COSINE), axis=1)
    valid = slots >= 0
    supported = valid[:, :15].any(axis=(1, 2)) & valid[:, 15:].any(axis=(1, 2))
    supported &= (good | ~valid).all(axis=(1, 2))
    axes = np.zeros_like(slots)
    axes[:, 15:] = np.where(valid[:, 15:], axial_axis, 0)
    return axes, supported


def reflect(images: np.ndarray, metadata: np.ndarray, axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = images.copy()
    result[:15] = images[:15, ::-1]
    horizontal, vertical = axes == 1, axes == 2
    result[horizontal] = images[horizontal][:, :, ::-1]
    result[vertical] = images[vertical][:, ::-1, :]
    quality = metadata.copy()
    # Sagittal slices are ordered from right to left in DICOM patient coordinates.
    for group, view in ((slice(0, 5), 0), (slice(5, 15), 1)):
        midpoint = (STARTS[view] + STARTS[view + 1] - 1) / (STARTS[-1] - 1)
        quality[group, 4] = (midpoint - quality[group, 4]) * quality[group, 0]
    return result[SWAP], quality[SWAP]


class ReflectedDataset(Dataset):
    def __init__(self, source: RegionDataset, axes: np.ndarray) -> None:
        self.source, self.axes = source, axes

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        images, metadata, labels = self.source[index]
        images, metadata = reflect(images, metadata, self.axes[index])
        return images, metadata, labels[SWAP[:25]]


def predict_reflections(model, cache, rows, coordinates, metadata, labels, resolution, batch):
    axes, supported = reflection_axes(cache.geometry[rows], coordinates[rows])
    if not supported.any():
        return supported, np.empty((0, 25, 3), dtype=np.float32)
    source = RegionDataset(cache, rows[supported], coordinates, metadata, labels, resolution=resolution)
    mirrored, _ = predict(model, DataLoader(ReflectedDataset(source, axes[supported]), batch_size=batch,
                                             pin_memory=True))
    return supported, mirrored[:, SWAP[:25]]


def evaluate(args: argparse.Namespace) -> None:
    torch.set_num_threads(4)
    cache = SpineCache.load(args.cache)
    folds = read_folds(args.folds, cache.studies)
    detail = read_json(args.comparison / 'metrics.json')
    fold_hash = checksum(args.folds)
    cache_hash = checksum(args.cache / 'audit.json')
    if detail['folds_sha256'] != fold_hash or detail['baseline_sha256'] != checksum(args.reference):
        raise ValueError('The comparison and reference must cover the same locked folds')
    previous = read_submission(args.reference, cache.studies)
    original = read_submission(args.comparison / 'detail-oof.csv', cache.studies)
    control = read_submission(args.comparison / 'equal-blend-oof.csv', cache.studies)
    np.testing.assert_allclose(control, .5 * previous + .5 * original, rtol=0, atol=1e-10)
    plan = {'axis_cosine': AXIS_COSINE, 'reflection_weight': .5, 'detail_blend_weight': .5,
            'folds_sha256': fold_hash, 'cache_sha256': cache_hash, 'folds': {},
            'comparison_sha256': checksum(args.comparison / 'metrics.json')}
    for fold in range(5):
        run = args.runs / f'fold-{fold}'
        regions = args.regions / f'fold-{fold}.npz'
        measured = detail['folds'][str(fold)]
        config = read_json(run / 'result.json')['config']
        hashes = {'region_sha256': checksum(regions), 'checkpoint_sha256': checksum(run / 'final.pt'),
                  'oof_sha256': checksum(run / 'oof.csv')}
        if (config['fold'] != fold or config['folds_sha256'] != fold_hash or config['cache_sha256'] != cache_hash
                or config['region_sha256'] != hashes['region_sha256'] or not config.get('spatial_detail')
                or hashes['checkpoint_sha256'] != measured['checkpoint_sha256']
                or hashes['oof_sha256'] != measured['prediction_sha256']):
            raise ValueError('Fold files and prior OOF comparison disagree')
        plan['folds'][str(fold)] = {**hashes, 'resolution': config.get('resolution', ROI_SIZE)}
    plan_path = args.output / 'comparison-plan.json'
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError('Use a separate directory for a different reflection study')
    write_json(plan_path, plan)

    # Every prediction comes from the sole grader trained without its outer-fold study.
    candidate = original.copy()
    counts = {}
    empty_labels = np.full(original.shape[:2], -1, dtype=np.int64)
    for fold in range(5):
        rows = np.flatnonzero(folds == fold)
        run = args.runs / f'fold-{fold}'
        fixed = read_submission(run / 'oof.csv', cache.studies[rows])
        np.testing.assert_allclose(fixed, original[rows], rtol=0, atol=1e-10)
        coordinates, quality = _load_fold(args.regions / f'fold-{fold}.npz', cache, folds, fold)
        state = torch.load(run / 'final.pt', map_location='cpu', weights_only=True)
        model = SpineClassifier(pretrained=False, spatial_detail=True).cuda()
        model.load_state_dict(state['model'])
        valid, mirrored = predict_reflections(model, cache, rows, coordinates, quality, empty_labels,
                                             plan['folds'][str(fold)]['resolution'], args.batch)
        subset = rows[valid]
        counts[str(fold)] = {'eligible': len(subset), 'studies': len(rows)}
        candidate[subset] = .5 * (original[subset] + mirrored)
        del model
        torch.cuda.empty_cache()
        print(f'Fold {fold}: {len(subset)} of {len(rows)} studies passed the orientation check', flush=True)

    truth = read_targets(args.root, cache.studies)
    blend = .5 * previous + .5 * candidate
    summary = {'plan_sha256': checksum(plan_path), 'studies': len(truth), 'coverage': counts}
    for name, probability in (('original_detail', original), ('reflected_detail', candidate),
                              ('original_blend', control), ('reflected_blend', blend)):
        summary[name] = {**classification_summary(truth, probability),
                         **interval(truth, probability, folds, args.bootstrap,
                                    reference=control if name == 'reflected_blend' else None)}
        print(name, summary[name], flush=True)
    submission(cache.studies, blend).to_csv(args.output / 'equal-blend-oof.csv', index=False)
    write_json(args.output / 'metrics.json', summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cache', 'folds', 'regions', 'runs', 'reference', 'comparison', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=6)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.batch < 1 or args.bootstrap < 1:
        parser.error('Use a positive batch size and bootstrap count')
    evaluate(args)


if __name__ == '__main__':
    main()
