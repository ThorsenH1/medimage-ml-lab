"""Test a fixed MRI checkpoint interpolation on a held-out fold with matching geometry."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from medimage.common import checksum, read_json, write_json
from medimage.spine_geometry_probe import classification_summary
from medimage.spine_localize import SpineCache, read_folds
from medimage.spine_metrics import read_submission, read_targets, submission
from medimage.spine_model import ROI_SIZE, SpineClassifier
from medimage.spine_train import RegionDataset, _load_fold, predict


def interpolate(parent, tuned, fraction):
    if parent.keys() != tuned.keys():
        raise ValueError('Checkpoint parameter names differ')
    result = {}
    for key, before in parent.items():
        after = tuned[key]
        if before.dtype != after.dtype or before.shape != after.shape:
            raise ValueError(f'Checkpoint tensor differs for {key}')
        result[key] = torch.lerp(before, after, fraction) if before.is_floating_point() else before
    return result


def run(args):
    torch.set_num_threads(4)
    cache = SpineCache.load(args.cache)
    folds = read_folds(args.folds, cache.studies)
    chosen = np.flatnonzero(folds == args.fold)
    region_path = args.regions / f'fold-{args.fold}.npz'
    fold_sha, cache_sha, region_sha = checksum(args.folds), checksum(args.cache / 'audit.json'), checksum(region_path)
    region, metadata = _load_fold(region_path, cache, folds, args.fold)
    endpoints = {}
    for key, folder in (('parent', args.parent), ('tuned', args.tuned)):
        path = folder / 'final.pt'
        endpoint = torch.load(path, map_location='cpu', weights_only=True)
        config = endpoint['config']
        report = read_json(folder / 'result.json')
        if (config != report['config'] or config['fold'] != args.fold or config['limit'] is not None
                or not config.get('spatial_detail') or config['folds_sha256'] != fold_sha
                or config['cache_sha256'] != cache_sha or config['region_sha256'] != region_sha):
            raise ValueError('The endpoints must use the same locked fold, cache and coordinate samples')
        endpoints[key] = (endpoint, report, checksum(path))
    parent, parent_report, parent_sha = endpoints['parent']
    tuned, tuned_report, tuned_sha = endpoints['tuned']
    resolution = parent['config'].get('resolution', ROI_SIZE)
    if tuned['config'].get('resolution', ROI_SIZE) != resolution or tuned['config']['parent_sha256'] != parent_sha:
        raise ValueError('The tuned endpoint must be a direct continuation at the same resolution')

    truth = read_targets(args.root, cache.studies)
    subset = cache.studies[chosen]
    original = read_submission(args.parent / 'oof.csv', subset)
    adapted = read_submission(args.tuned / 'oof.csv', subset)
    measurements = {}
    for name, scores, report in (('parent', original, parent_report), ('tuned', adapted, tuned_report)):
        summary = classification_summary(truth[chosen], scores)
        if not np.isclose(summary['rsna_log_loss'], report['validation']['score'], atol=1e-7, rtol=0):
            raise ValueError('Saved held-out predictions do not reproduce the endpoint score')
        measurements[name] = summary

    model = SpineClassifier(pretrained=False, spatial_detail=True)
    model.load_state_dict(interpolate(parent['model'], tuned['model'], args.fraction))
    dataset = RegionDataset(cache, chosen, region, metadata, truth, resolution=resolution)
    probabilities, actual = predict(model.cuda(), DataLoader(dataset, batch_size=args.batch, pin_memory=True))
    np.testing.assert_array_equal(actual, truth[chosen])
    probability_control = (1 - args.fraction) * original + args.fraction * adapted
    measurements['probability_interpolation'] = classification_summary(actual, probability_control)
    measurements['parameter_interpolation'] = classification_summary(actual, probabilities)
    out = {'fold': args.fold, 'studies': len(subset), 'folds_sha256': fold_sha, 'cache_sha256': cache_sha,
           'region_sha256': region_sha, 'parent_sha256': parent_sha, 'tuned_sha256': tuned_sha,
           'tuned_fraction': args.fraction, 'reflection': False, 'metrics': measurements}
    args.output.mkdir(parents=True, exist_ok=True)
    submission(subset, probabilities).to_csv(args.output / 'interpolated-oof.csv', index=False)
    write_json(args.output / 'metrics.json', out)
    print(out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cache', 'folds', 'regions', 'parent', 'tuned', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--fraction', type=float, default=.25)
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.fraction < 1 or args.batch < 1:
        parser.error('Use a fraction between zero and one and a positive batch size')
    run(args)


if __name__ == '__main__':
    main()
