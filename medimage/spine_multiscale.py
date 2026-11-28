"""Fit all lumbar MRI targets with local appearance at two physical scales."""
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
from medimage.spine_crossplane import appearance, crop_stack, cross_fitted_regions
from medimage.spine_localize import SpineCache, read_folds
from medimage.spine_metrics import GROUPS, read_submission, read_targets, submission

FIELD_RATIO = .65
BLEND = .15
FIELD_FEATURE_COUNT = 149
FEATURE_COUNT = 2 * FIELD_FEATURE_COUNT + 20
BOOSTER = {'max_iter': 140, 'learning_rate': .05, 'max_leaf_nodes': 15, 'min_samples_leaf': 36,
           'l2_regularization': 2.5, 'random_state': 6337, 'early_stopping': False}


def field_features(volume, coordinates):
    return np.stack([appearance(crop_stack(volume, region), region[:, 0] >= 0) for region in coordinates])


def descriptor(volume, coordinates, quality):
    coordinates = coordinates[:25]
    full = field_features(volume, coordinates)
    narrow = coordinates.copy()
    narrow[..., 3:] *= FIELD_RATIO
    close = field_features(volume, narrow)
    # Compare the intensity quantiles with the matching location at other levels within the study.
    observed = quality[:25, 0].reshape(5, 5) > 0
    quantiles = full[:, -5:].reshape(5, 5, 5)
    mask = observed[..., None]
    mean = (quantiles * mask).sum(axis=1, keepdims=True) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
    relative = ((quantiles - mean) * mask).reshape(25, 5)
    level = np.tile(np.eye(5), (5, 1))
    condition = np.repeat(np.eye(5), 5, axis=0)
    return np.column_stack((level, condition, quality[:25], relative, full, close)).astype(np.float32)


def feature_store(cache, layouts, output, plan_sha256, workers):
    path, receipt = output / 'descriptors.npy', output / 'descriptors.json'
    signature = {'plan_sha256': plan_sha256, 'shape': [5, len(cache.studies), 25, FEATURE_COUNT]}
    if receipt.is_file():
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if read_json(receipt) != signature or list(array.shape) != signature['shape'] or array.dtype != np.float32:
            raise ValueError('Stored multiscale features belong to another fold plan')
        return array
    output_array = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=tuple(signature['shape']))

    def describe(row):
        volume = np.array(cache.volumes[row], copy=True)
        return np.stack([descriptor(volume, regions[row], quality[row]) for regions, quality in layouts])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(cache.studies), 4 * workers):
            rows = range(start, min(start + 4 * workers, len(cache.studies)))
            for row, features in zip(rows, pool.map(describe, rows), strict=True):
                if not np.isfinite(features).all():
                    raise ValueError('Unexpected nonfinite multi-scale region descriptors')
                output_array[:, row] = features
            if start % (4 * workers * 8) == 0:
                print(f'Extracted both scales for {rows.stop}/{len(cache.studies)} studies', flush=True)
    output_array.flush()
    del output_array
    write_json(receipt, signature)
    return np.load(path, mmap_mode='r', allow_pickle=False)


def fit_family(features, labels, fit, check):
    channels = features.shape[-1]
    training = features[fit].reshape(-1, channels)
    actual = labels[fit].ravel()
    evaluation = features[check].reshape(-1, channels)
    known = actual >= 0
    abnormal = actual > 0
    at_least_moderate = HistGradientBoostingClassifier(**BOOSTER).fit(training[known], abnormal[known])
    severe_given_abnormal = HistGradientBoostingClassifier(**BOOSTER).fit(training[abnormal], actual[abnormal] == 2)
    a = at_least_moderate.predict_proba(evaluation)[:, 1]
    s = severe_given_abnormal.predict_proba(evaluation)[:, 1]
    return np.column_stack((1 - a, a * (1 - s), a * s)).reshape(len(check), labels.shape[1], 3)


def run(args):
    cv2.setNumThreads(1)
    cache = SpineCache.load(args.cache)
    folds = read_folds(args.folds, cache.studies)
    cache_sha, folds_sha = checksum(args.cache / 'audit.json'), checksum(args.folds)
    known_geometry = read_json(args.regions / 'localization.json')['config']
    if (known_geometry['cache_sha256'] != cache_sha or known_geometry['folds_file_sha256'] != folds_sha):
        raise ValueError('The MRI localization does not match the dataset and fold plan')
    paths = [args.regions / f'fold-{fold}.npz' for fold in range(5)]
    plan = {'cache_sha256': cache_sha, 'folds_sha256': folds_sha, 'regions_sha256': [checksum(path) for path in paths],
            'trainer_sha256': checksum(Path(__file__)), 'reference_sha256': checksum(args.reference),
            'secondary_reference_sha256': checksum(args.secondary_reference), 'narrow_field_ratio': FIELD_RATIO,
            'blend_fraction': BLEND, 'learner': BOOSTER, 'studies': len(cache.studies)}
    plan_path = args.output / 'plan.json'
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError('Use a fresh directory for a different MRI multiscale experiment')
    write_json(plan_path, plan)
    plan_sha = checksum(plan_path)
    layouts = [cross_fitted_regions(path, cache.studies, folds, fold) for fold, path in enumerate(paths)]
    features = feature_store(cache, layouts, args.output, plan_sha, args.workers)
    truth = read_targets(args.root, cache.studies)
    reference = read_submission(args.reference, cache.studies)
    previous = read_submission(args.secondary_reference, cache.studies)
    model = np.empty_like(reference)
    for fold in range(5):
        fit, check = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        path = args.output / f'fold-{fold}.npz'
        if path.is_file():
            with np.load(path, allow_pickle=False) as saved:
                if not np.array_equal(saved['rows'], check) or saved['plan_sha256'].item() != plan_sha:
                    raise ValueError('The stored estimates came from another MRI fold plan')
                model[check] = saved['predictions']
            continue
        for family in GROUPS:
            model[check, family] = fit_family(features[fold, :, family], truth[:, family], fit, check)
        save_scores(path, rows=check, predictions=model[check], plan_sha256=plan_sha)
        print(f'Trained and evaluated MRI fold {fold} on {len(check)} held-out studies', flush=True)

    blend = (1 - BLEND) * reference + BLEND * model
    report = {'plan_sha256': plan_sha, 'reference': classification_summary(truth, reference),
              'secondary_reference': classification_summary(truth, previous), 'standalone': classification_summary(truth, model),
              'blend': {**classification_summary(truth, blend),
                        **interval(truth, blend, folds, args.bootstrap, reference=previous)}}
    submission(cache.studies, model).to_csv(args.output / 'multiscale-oof.csv', index=False)
    submission(cache.studies, blend).to_csv(args.output / 'blend-oof.csv', index=False)
    write_json(args.output / 'metrics.json', report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cache', 'folds', 'regions', 'reference', 'secondary-reference', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if min(args.workers, args.bootstrap) < 1:
        parser.error('Use positive worker and bootstrap counts')
    with threadpool_limits(limits=4):
        run(args)


if __name__ == '__main__':
    main()
