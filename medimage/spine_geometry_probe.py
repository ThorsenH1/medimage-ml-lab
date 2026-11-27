"""Measure the effect of a learned localizer on a frozen outer-fold classifier."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader

from medimage.auc_spine import classification_summary
from medimage.common import checksum, write_json
from medimage.spine_data import project
from medimage.spine_heatmap import DiscHeatmap, anchor_inputs, decode, mm_metrics, pixel_grid
from medimage.spine_localize import (SpineCache, fit_localizer, image_features, make_regions, predict_keypoints,
                                     read_folds)
from medimage.spine_metrics import read_targets
from medimage.spine_model import SpineClassifier
from medimage.spine_train import RegionDataset, _load_fold, predict


@torch.inference_mode()
def infer_landmarks(cache: SpineCache, rows: np.ndarray, state: dict, batch: int) -> tuple[np.ndarray, np.ndarray]:
    images, geometry = anchor_inputs(cache, rows)
    network = DiscHeatmap().cuda().eval()
    network.load_state_dict(state)
    axes = pixel_grid(torch.device('cuda'))
    estimates = []
    with torch.autocast('cuda', dtype=torch.float16):
        for chunk in torch.from_numpy(images).split(batch):
            estimates.append(decode(network(chunk.cuda().float().div_(255.)), axes))
    return np.concatenate(estimates), geometry


def probe(args: argparse.Namespace) -> None:
    torch.set_num_threads(4)
    cache = SpineCache.load(args.cache)
    fold_hash = checksum(args.folds)
    cache_hash = checksum(args.cache / 'audit.json')
    localizer = torch.load(args.heatmap, map_location='cpu', weights_only=True)
    classifier = torch.load(args.classifier, map_location='cpu', weights_only=True)
    record, classifier_config = localizer['audit'], classifier['config']
    outer = record['outer_excluded']
    if (record['folds_sha256'] != fold_hash or record['cache_sha256'] != cache_hash
            or classifier_config['folds_sha256'] != fold_hash or classifier_config['cache_sha256'] != cache_hash
            or classifier_config['fold'] != outer or record['validation_fold'] == outer
            or outer in record['training_folds'] or classifier_config['limit'] is not None):
        raise ValueError('The localizer and classifier must both hold out the same complete outer fold')
    frozen_regions = args.localization / f'fold-{outer}.npz'
    if checksum(frozen_regions) != classifier_config['region_sha256']:
        raise ValueError('The classifier was fitted with a different reference localizer')
    folds = read_folds(args.folds, cache.studies)
    held = np.flatnonzero(folds == outer)
    fit = np.flatnonzero(folds != outer)
    geometry_report = {}
    points, held_geometry = infer_landmarks(cache, held, localizer['state'], args.batch)
    reference_features, anchor_geometry = image_features(cache)
    if not np.array_equal(held_geometry, anchor_geometry[held]):
        raise ValueError('Heatmap projection and registration use different anchor planes')
    distances = cdist(reference_features, reference_features, 'sqeuclidean').astype(np.float32)
    atlas = fit_localizer(fit, distances, anchor_geometry, cache.points)
    old_points, confidence, _ = predict_keypoints(distances[np.ix_(held, fit)], atlas)
    truth_points, _ = project(cache.points[held, :5], held_geometry[:, None])
    geometry_report['reference'] = mm_metrics(old_points, truth_points, held_geometry)
    geometry_report['heatmap'] = mm_metrics(points, truth_points, held_geometry)
    region, metadata = _load_fold(frozen_regions, cache, folds, outer)
    targets = read_targets(args.root, cache.studies)
    model = SpineClassifier(pretrained=False).cuda()
    model.load_state_dict(classifier['model'])
    baseline, truth = predict(model, DataLoader(
        RegionDataset(cache, held, region, metadata, targets), batch_size=args.batch, pin_memory=True))
    region[held], metadata[held], _ = make_regions(cache, held, anchor_geometry, points, confidence, atlas)
    changed, changed_truth = predict(model, DataLoader(
        RegionDataset(cache, held, region, metadata, targets), batch_size=args.batch, pin_memory=True))
    if not np.array_equal(truth, changed_truth):
        raise ValueError('The two classifiers did not score the same labels')
    summary = {'outer_fold': outer, 'held_out_studies': len(held), 'folds_sha256': fold_hash,
               'heatmap_sha256': checksum(args.heatmap), 'classifier_sha256': checksum(args.classifier),
               'geometry': geometry_report, 'original_coordinates': classification_summary(truth, baseline),
               'heatmap_coordinates': classification_summary(truth, changed)}
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / 'paired-predictions.npz', rows=held, truth=truth,
                        baseline=baseline, heatmap=changed)
    write_json(args.output / 'metrics.json', summary)
    print(summary, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cache', 'folds', 'localization', 'heatmap', 'classifier', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error('Batch size must be positive')
    probe(args)


if __name__ == '__main__':
    main()
