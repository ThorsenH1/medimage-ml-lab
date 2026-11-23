from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from medimage.common import checksum, read_json, write_json
from medimage.spine_data import SIZE, STARTS, TARGET_VIEWS, anchor_slot, project

ROI_VIEWS = np.r_[TARGET_VIEWS, np.full(5, 2)]
ROI_WIDTH_MM = np.r_[np.repeat([54, 48, 48, 46, 46], 5), np.full(5, 49)]
HOG = cv2.HOGDescriptor((80, 80), (16, 16), (8, 8), (8, 8), 9)


@dataclass
class SpineCache:
    path: Path
    volumes: np.ndarray
    studies: np.ndarray
    geometry: np.ndarray
    valid: np.ndarray
    points: np.ndarray

    @classmethod
    def load(cls, path: Path) -> 'SpineCache':
        audit = read_json(path / 'audit.json')
        arrays = [np.load(path / f'{name}.npy', mmap_mode='r', allow_pickle=False)
                  for name in ('volume', 'studies', 'geometry', 'valid', 'points')]
        if any(len(item) != audit['config']['studies'] for item in arrays):
            raise ValueError('Inconsistent spine cache lengths')
        return cls(path, *arrays)


def _describe(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (80, 80), interpolation=cv2.INTER_AREA)
    hog = HOG.compute(image).reshape(-1)
    hog = hog / max(1, np.linalg.norm(hog))
    intensity = cv2.resize(image, (20, 20), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
    intensity -= intensity.mean()
    intensity /= max(1, np.linalg.norm(intensity))
    return np.r_[hog, .55 * intensity]


def image_features(cache: SpineCache) -> tuple[np.ndarray, np.ndarray]:
    cv2.setNumThreads(1)
    features, anchors = [], []
    for volume, present, geometry in zip(cache.volumes, cache.valid, cache.geometry, strict=True):
        slot = anchor_slot(present)
        anchors.append(geometry[slot])
        reference_view = int(np.searchsorted(STARTS[1:], slot, side='right'))
        first, last = STARTS[reference_view:reference_view + 2]
        neighbors = np.flatnonzero(present[first:last]) + first
        central = neighbors[np.argsort(np.abs(neighbors - slot))[:3]]
        composite = np.median(volume[central], axis=0).astype(np.uint8)
        features.append(np.r_[_describe(composite), np.eye(3, dtype=np.float32)[reference_view] * .45])
    return np.stack(features), np.stack(anchors)


def world_from_pixels(xy: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    return (geometry[:, None, 0] + xy[..., 0, None] * geometry[:, None, 1]
            + xy[..., 1, None] * geometry[:, None, 2])


def kernel(distances: np.ndarray) -> np.ndarray:
    return np.exp(-2.5 * distances) * .65 + np.exp(-.7 * distances) * .35


def fit_localizer(train: np.ndarray, distances: np.ndarray, anchors: np.ndarray, points: np.ndarray) -> dict:
    xy, _ = project(points[:, :5], anchors[:, None])
    allowed = ((xy >= 0) & (xy < SIZE)).all(axis=-1) & np.isfinite(xy).all(axis=-1)
    weights = np.zeros((len(train), 5, 2), dtype=np.float32)
    medians = np.empty((5, 2), dtype=np.float32)
    for level in range(5):
        enabled = allowed[train, level]
        identity = np.flatnonzero(enabled)
        subset = train[enabled]
        if len(subset) < 15:
            raise ValueError('Insufficient training keypoints to fit one spinal level')
        target = xy[subset, level]
        medians[level] = np.median(target, axis=0)
        training_kernel = kernel(distances[np.ix_(subset, subset)])
        training_kernel[np.diag_indices_from(training_kernel)] += .11
        weights[identity, level] = solve(training_kernel, target - medians[level], assume_a='pos')
    reference = world_from_pixels(xy, anchors)
    displacement = (points - np.tile(reference, (1, 5, 1)))[train]
    displacement = np.where(np.abs(displacement) < 90., displacement, np.nan)
    offsets = np.nanmedian(displacement, axis=0).astype(np.float32)
    if not np.isfinite(offsets).all():
        raise ValueError('An atlas offset could not be fitted')
    return {'weights': weights, 'medians': medians, 'offsets': offsets, 'train_indices': train}


def predict_keypoints(distances: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    affinities = kernel(distances)
    confidence = np.max(affinities, axis=1)
    xy = (affinities @ model['weights'].reshape(affinities.shape[1], -1)).reshape(-1, 5, 2) + model['medians']
    plausible = ((xy >= 4.) & (xy < SIZE - 4.)).all(axis=-1) & (confidence[:, None] >= .17)
    predicted = np.where(plausible[..., None], xy, model['medians'])
    per_level_confidence = np.where(plausible, confidence[:, None], 0.)
    return predicted.astype(np.float32), per_level_confidence.astype(np.float32), ~plausible


def roi_coordinates(world: np.ndarray, confidence: np.ndarray, geometry: np.ndarray,
                    present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.zeros((len(ROI_VIEWS), 3, 5), dtype=np.float32)
    coordinates[..., 0] = -1
    meta = np.zeros((len(ROI_VIEWS), 5), dtype=np.float32)
    for view in range(3):
        available = np.flatnonzero(present[STARTS[view]:STARTS[view + 1]]) + STARTS[view]
        if len(available) == 0:
            continue
        target_indices = np.flatnonzero(ROI_VIEWS == view)
        planes = geometry[available]
        xy, distance = project(world[target_indices, None], planes[None])
        clipped = np.clip(xy, 0, SIZE - 1)
        out_of_frame = np.linalg.norm(xy - clipped, axis=-1)
        best = np.argmin(distance + 2 * out_of_frame, axis=1)
        adjacent = np.clip(best[:, None] + np.arange(-1, 2), 0, len(available) - 1)
        batch = np.arange(len(target_indices))
        full_width = ROI_WIDTH_MM[target_indices, None, None]
        scale = np.linalg.norm(planes[adjacent, 1:], axis=-1)
        extent = np.clip(full_width / scale, 18, SIZE)
        coordinates[target_indices, :, 0] = available[adjacent]
        coordinates[target_indices, :, 1:3] = clipped[batch[:, None], adjacent]
        coordinates[target_indices, :, 3:] = extent
        meta[target_indices] = np.column_stack([
            np.ones(len(batch)), confidence[target_indices % 5],
            np.minimum(distance[batch, best], 40) / 40,
            np.minimum(out_of_frame[batch, best], 40) / 40,
            available[best] / (STARTS[-1] - 1),
        ])
    return coordinates, meta


def make_regions(cache: SpineCache, indices: np.ndarray, anchor_geometry: np.ndarray,
                 prediction: np.ndarray, confidence: np.ndarray, model: dict,
                 *, use_oracle: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = world_from_pixels(prediction, anchor_geometry[indices])
    world = np.tile(predicted, (1, 5, 1)) + model['offsets']
    if use_oracle:
        truth = cache.points[indices]
        world = np.where(np.isfinite(truth).all(axis=-1, keepdims=True), truth, world)
    world = np.concatenate([world, world[:, :5]], axis=1)
    regions, metadata = [], []
    for row, point, quality in zip(indices, world, confidence, strict=True):
        coords, meta = roi_coordinates(point, quality, cache.geometry[row], cache.valid[row])
        regions.append(coords)
        metadata.append(meta)
    return np.stack(regions), np.stack(metadata), world


def read_folds(path: Path, studies: np.ndarray) -> np.ndarray:
    table = pd.read_csv(path)
    if table.study_id.duplicated().any() or set(table.study_id) != set(studies) or set(table.fold) != set(range(5)):
        raise ValueError('Expected exactly five complete, mutually exclusive study folds')
    return table.set_index('study_id').loc[studies, 'fold'].to_numpy()


def _localization_report(cache: SpineCache, predicted: np.ndarray, median: np.ndarray,
                         anchors: np.ndarray, world: np.ndarray, regions: np.ndarray) -> dict:
    truth = cache.points[:, :5]
    true_xy, _ = project(truth, anchors[:, None])
    on_plane = world_from_pixels(true_xy, anchors)
    errors = np.linalg.norm(world_from_pixels(predicted, anchors) - on_plane, axis=-1)
    prior_errors = np.linalg.norm(world_from_pixels(median, anchors) - on_plane, axis=-1)
    world_errors = np.linalg.norm(world[:, :25] - cache.points, axis=-1)
    return {'study_count': len(truth), 'median_anchor_error_mm': float(np.nanmedian(errors)),
            'atlas_only_median_anchor_error_mm': float(np.nanmedian(prior_errors)),
            'mean_anchor_error_mm': float(np.nanmean(errors)),
            'within_10mm': float(np.nanmean(np.where(np.isfinite(errors), errors <= 10, np.nan))),
            'per_level_median_mm': np.nanmedian(errors, axis=0).tolist(),
            'median_3d_error_by_condition_mm': np.nanmedian(world_errors.reshape(-1, 5, 5), axis=(0, 2)).tolist(),
            'missing_primary_roi_fraction': float(np.mean(regions[:, :25, 1, 0] < 0))}


def crossfit(cache: SpineCache, folds_path: Path, output: Path) -> None:
    features, anchors = image_features(cache)
    folds = read_folds(folds_path, cache.studies)
    output.mkdir(parents=True, exist_ok=True)
    signature = {'folds_file_sha256': checksum(folds_path), 'cache_sha256': checksum(cache.path / 'audit.json'),
                 'code_sha256': checksum(Path(__file__)), 'study_count': len(folds)}
    if (output / 'localization.json').is_file():
        if read_json(output / 'localization.json')['config'] != signature:
            raise ValueError('Localization configuration changed')
        if not all((output / f'fold-{fold}.npz').is_file() and (output / f'localizer-{fold}.npz').is_file()
                   for fold in range(5)) or not (output / 'reference-features.npy').is_file():
            raise ValueError('Completed localization artifacts are missing')
        print('Found complete nested cross-fit localization', flush=True)
        return
    np.save(output / 'reference-features.npy', features, allow_pickle=False)
    distances = cdist(features, features, 'sqeuclidean').astype(np.float32)
    oof_xy = np.empty((len(folds), 5, 2), np.float32)
    oof_median = np.empty_like(oof_xy)
    oof_world = np.empty((len(folds), 30, 3), np.float32)
    oof_regions = np.empty((len(folds), 30, 3, 5), np.float32)
    oof_confidence = np.empty((len(folds), 5), dtype=np.float32)
    fit_bank = {}
    with threadpool_limits(limits=4):
        for outer in range(5):
            region = np.empty_like(oof_regions)
            metadata = np.empty((len(folds), 30, 5), np.float32)
            trained_on = np.zeros((len(folds), 5), np.bool_)
            for source in range(5):
                excluded = tuple(sorted({outer, source}))
                training = np.flatnonzero(~np.isin(folds, excluded))
                rows = np.flatnonzero(folds == source)
                if excluded not in fit_bank:
                    fit_bank[excluded] = fit_localizer(training, distances, anchors, cache.points)
                model = fit_bank[excluded]
                xy, confidence, _ = predict_keypoints(distances[np.ix_(rows, training)], model)
                region[rows], metadata[rows], world = make_regions(cache, rows, anchors, xy, confidence, model)
                trained_on[np.ix_(rows, np.unique(folds[training]))] = True
                if source == outer:
                    oof_xy[rows], oof_median[rows], oof_world[rows] = xy, model['medians'], world
                    oof_regions[rows], oof_confidence[rows] = region[rows], confidence
                    np.savez_compressed(output / f'localizer-{outer}.npz', **model)
                    if outer == 0:
                        oracle_region, oracle_meta, _ = make_regions(cache, rows, anchors, xy, confidence,
                                                                     model, use_oracle=True)
                        np.savez_compressed(output / 'oracle-fold-0.npz', rows=rows, region=oracle_region,
                                            metadata=oracle_meta)
            if trained_on[:, outer].any() or trained_on[np.arange(len(folds)), folds].any():
                raise AssertionError('A localization fit saw an outer validation study or its own row')
            np.savez_compressed(output / f'fold-{outer}.npz', studies=cache.studies, fold_per_study=folds,
                                region=region, metadata=metadata, trained_on=trained_on)
            print(f'Localization outer fold {outer + 1}/5 completed', flush=True)
    np.savez_compressed(output / 'oof-localization.npz', studies=cache.studies, xy=oof_xy,
                        confidence=oof_confidence, world_mm=oof_world, folds=folds)
    report = _localization_report(cache, oof_xy, oof_median, anchors, oof_world, oof_regions)
    report.update(config=signature, nested_fit_count=len(fit_bank),
                  keypoint_fallback_fraction=float((oof_confidence == 0).mean()))
    write_json(output / 'localization.json', report)
    print(report, flush=True)


def inference_regions(cache: SpineCache, directory: Path, fold: int) -> tuple[np.ndarray, np.ndarray]:
    features, anchors = image_features(cache)
    reference = np.load(directory / 'reference-features.npy', allow_pickle=False)
    with np.load(directory / f'localizer-{fold}.npz', allow_pickle=False) as saved:
        model = dict(saved)
    distances = cdist(features, reference[model['train_indices']], 'sqeuclidean')
    xy, confidence, _ = predict_keypoints(distances, model)
    regions, metadata, _ = make_regions(cache, np.arange(len(cache.studies)), anchors, xy, confidence, model)
    return regions, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description='Nested cross-fit MRI keypoints and physical ROI coordinates')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--folds', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    crossfit(SpineCache.load(args.cache), args.folds, args.output)


if __name__ == '__main__':
    main()
