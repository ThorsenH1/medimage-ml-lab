"""Learn a foraminal severity scorer from matching sagittal and axial MRI regions."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from medimage.auc_selection import save_scores
from medimage.auc_spine import classification_summary, interval
from medimage.common import checksum, read_json, write_json
from medimage.spine_data import STARTS, project
from medimage.spine_localize import SpineCache, read_folds
from medimage.spine_metrics import read_submission, read_targets, submission

FORAMINA = slice(5, 15)
PATCH_SIZE = 64
PLANE_LIMIT_MM = 8.
SIDE_PLANE_LIMIT_MM = 12.
PART = .15
LEARNER = {'max_iter': 120, 'learning_rate': .055, 'max_leaf_nodes': 15,
           'min_samples_leaf': 28, 'l2_regularization': 2., 'random_state': 6301, 'early_stopping': False}


def axial_regions(sagittal, geometry, valid, size):
    regions = np.zeros_like(sagittal)
    regions[..., 0] = -1
    center = sagittal[:, 1]
    chosen = np.flatnonzero(center[:, 0] >= 0)
    available = np.flatnonzero(valid[STARTS[2]:STARTS[3]]) + STARTS[2]
    if not len(available) or not len(chosen):
        return regions, np.zeros((len(center), 3), dtype=np.float32)
    source = geometry[center[chosen, 0].astype(int)]
    point = source[:, 0] + (center[chosen, 1:3, None] * source[:, 1:]).sum(axis=1)
    planes = geometry[available]
    xy, distance = project(point[:, None], planes[None])
    inside = ((xy >= 0) & (xy <= size - 1)).all(axis=-1)
    nearest = np.argmin(np.where(inside, distance, np.inf), axis=-1)
    row = np.arange(len(chosen))
    good = inside[row, nearest] & (distance[row, nearest] <= PLANE_LIMIT_MM)
    adjacent = nearest[:, None] + [-1, 0, 1]
    safe = adjacent.clip(0, len(available) - 1)
    accepted = (good[:, None] & (safe == adjacent) & inside[row[:, None], safe]
                & (distance[row[:, None], safe] <= SIDE_PLANE_LIMIT_MM))
    # Preserve the same physical field of view as the corresponding sagittal region.
    width_mm = (center[chosen, 3:5] * np.linalg.norm(source[:, 1:], axis=-1)).mean(axis=-1)
    step = np.linalg.norm(planes[safe, 1:], axis=-1)
    extent = np.clip(width_mm[:, None, None] / step, 18, size)
    projected = np.concatenate((available[safe, None], xy[row[:, None], safe], extent), axis=-1)
    regions[chosen] = np.where(accepted[..., None], projected, regions[chosen])
    quality = np.zeros((len(center), 3), dtype=np.float32)
    quality[chosen] = np.column_stack((good, accepted.mean(axis=1), np.where(good, distance[row, nearest] / 8., 0.)))
    return regions, quality


def grid(image):
    return cv2.resize(image, (6, 6), interpolation=cv2.INTER_AREA).ravel()


def crop_stack(volume, regions):
    patches = np.zeros((3, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    for i, (slot, x, y, width, height) in enumerate(regions):
        if slot < 0:
            continue
        sx, sy = width / PATCH_SIZE, height / PATCH_SIZE
        offset = (PATCH_SIZE - 1) / 2
        affine = np.array([[sx, 0, x - offset * sx], [0, sy, y - offset * sy]], dtype=np.float32)
        patches[i] = cv2.warpAffine(volume[int(slot)], affine, (PATCH_SIZE, PATCH_SIZE),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return patches


def appearance(patches, present):
    image = patches.astype(np.float32) / 255.
    central = image[1]
    mean = image[present].mean(axis=0) if present.any() else np.zeros_like(central)
    return np.concatenate((grid(central), grid(mean), grid(np.abs(np.diff(central, axis=0))),
                           grid(np.abs(np.diff(central, axis=1))), np.quantile(central, [.1, .25, .5, .75, .9])))


def study_features(cache, row, coords, metadata):
    sagittal = coords[FORAMINA]
    axial, quality = axial_regions(sagittal, cache.geometry[row], cache.valid[row], cache.volumes.shape[-1])
    volume = cache.volumes[row]
    sagittal_features, axial_features = [np.stack([appearance(crop_stack(volume, region), region[:, 0] >= 0)
                                                   for region in group]) for group in (sagittal, axial)]
    labels = np.tile(np.eye(5), (2, 1))
    side = np.repeat([0., 1.], 5)[:, None]
    primary = np.column_stack((labels, side, metadata[FORAMINA], sagittal_features)).astype(np.float32)
    extra = np.column_stack((quality, axial_features)).astype(np.float32)
    return primary, extra


def cross_fitted_regions(path, studies, folds, held_out):
    with np.load(path, allow_pickle=False) as saved:
        sources = (np.arange(5)[None] != folds[:, None]) & (np.arange(5)[None] != held_out)
        if (not np.array_equal(saved['studies'], studies) or not np.array_equal(saved['fold_per_study'], folds)
                or not np.array_equal(saved['trained_on'], sources)):
            raise ValueError('The localizer must exclude each scored study and its outer evaluation fold')
        return saved['region'], saved['metadata']


def fit_boundaries(features, labels, fit, check):
    train = features[fit].reshape(-1, features.shape[-1])
    targets = labels[fit].ravel()
    held = features[check].reshape(-1, features.shape[-1])
    observed = targets >= 0
    abnormal = targets > 0
    first = HistGradientBoostingClassifier(**LEARNER).fit(train[observed], abnormal[observed])
    second = HistGradientBoostingClassifier(**LEARNER).fit(train[abnormal], targets[abnormal] == 2)
    a = first.predict_proba(held)[:, 1]
    s = second.predict_proba(held)[:, 1]
    return np.column_stack((1 - a, a * (1 - s), a * s)).reshape(len(check), 10, 3)


def run(args):
    cv2.setNumThreads(1)
    cache = SpineCache.load(args.cache)
    folds = read_folds(args.folds, cache.studies)
    cache_hash, fold_hash = checksum(args.cache / 'audit.json'), checksum(args.folds)
    loc = read_json(args.regions / 'localization.json')['config']
    if loc['cache_sha256'] != cache_hash or loc['folds_file_sha256'] != fold_hash:
        raise ValueError('The region cache must match the MRI studies and fold membership')
    truth = read_targets(args.root, cache.studies)
    reference = read_submission(args.reference, cache.studies)
    reference_summary = classification_summary(truth, reference)
    plan = {'folds_sha256': fold_hash, 'cache_sha256': cache_hash, 'reference_sha256': checksum(args.reference),
            'trainer_sha256': checksum(Path(__file__)), 'studies': len(truth), 'learner': LEARNER,
            'patch_size': PATCH_SIZE, 'axial_plane_limit_mm': PLANE_LIMIT_MM, 'blend_fraction': PART,
            'regions_sha256': [checksum(args.regions / f'fold-{i}.npz') for i in range(5)]}
    plan_path = args.output / 'plan.json'
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError('Use a fresh output directory for a different cross-plane experiment')
    write_json(plan_path, plan)
    plan_sha = checksum(plan_path)
    predicted = {name: reference.copy() for name in ('sagittal', 'two_plane')}
    axial_counts = {}
    for fold in range(5):
        fit, check = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        path = args.output / f'fold-{fold}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if saved['plan_sha256'].item() != plan_sha or not np.array_equal(saved['rows'], check):
                    raise ValueError('Saved predictions came from a different fold or experiment')
                for name in predicted:
                    predicted[name][check, FORAMINA] = saved[name]
                axial_counts[str(fold)] = int(saved['axial_present'].item())
            continue
        coords, metadata = cross_fitted_regions(args.regions / f'fold-{fold}.npz', cache.studies, folds, fold)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(lambda row: study_features(cache, row, coords[row], metadata[row]),
                                 range(len(truth))))
        primary, secondary = [np.stack(values) for values in zip(*rows, strict=True)]
        combined = np.concatenate((primary, secondary), axis=-1)
        if not np.isfinite(combined).all():
            raise ValueError('Nonfinite MRI appearance features')
        first = fit_boundaries(primary, truth[:, FORAMINA], fit, check)
        second = fit_boundaries(combined, truth[:, FORAMINA], fit, check)
        predicted['sagittal'][check, FORAMINA] = first
        predicted['two_plane'][check, FORAMINA] = second
        count = int(secondary[check, :, 0].sum())
        axial_counts[str(fold)] = count
        save_scores(path, rows=check, plan_sha256=plan_sha, sagittal=first, two_plane=second, axial_present=count)
        print(f'Fold {fold}: axial region available for {count} of {10 * len(check)} targets', flush=True)
    report = {'plan_sha256': plan_sha, 'axial_coverage_by_fold': axial_counts,
              'axial_coverage_fraction': sum(axial_counts.values()) / (len(truth) * 10), 'reference': reference_summary}
    for name, probabilities in predicted.items():
        blended = reference.copy()
        blended[:, FORAMINA] = (1 - PART) * reference[:, FORAMINA] + PART * probabilities[:, FORAMINA]
        report[name] = {**classification_summary(truth, blended),
                        **interval(truth, blended, folds, args.bootstrap, reference=reference)}
        submission(cache.studies, blended).to_csv(args.output / f'{name}-blend-oof.csv', index=False)
        print(name, report[name], flush=True)
    write_json(args.output / 'metrics.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cache', 'folds', 'regions', 'reference', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if min(args.workers, args.bootstrap) < 1:
        parser.error('Use positive worker and bootstrap counts')
    with threadpool_limits(limits=4):
        run(args)


if __name__ == '__main__':
    main()
