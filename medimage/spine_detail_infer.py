"""Lock and run the equal blend of the reference and new spatial MRI graders."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from medimage.common import checksum, read_json, write_json
from medimage.spine_data import SEVERITIES, TARGETS, prepare
from medimage.spine_heatmap import DiscHeatmap, anchor_inputs, observed_landmarks
from medimage.spine_heatmap_crossfit import anatomy_offsets, predict_heatmaps
from medimage.spine_localize import SpineCache, inference_regions, make_regions, read_folds
from medimage.spine_metrics import read_submission, submission
from medimage.spine_model import SpineClassifier
from medimage.spine_train import RegionDataset, predict


def lock(args):
    cache = SpineCache.load(args.train_cache)
    folds = read_folds(args.folds, cache.studies)
    audit = read_json(args.learned / 'localization.json')
    comparison = read_json(args.comparison)
    config = audit['config']
    fold_hash = checksum(args.folds)
    if (config['folds_file_sha256'] != fold_hash or comparison['folds_sha256'] != fold_hash
            or config['cache_sha256'] != checksum(args.train_cache / 'audit.json')):
        raise ValueError('The deployment bundle must match the frozen training study splits')
    _, geometry = anchor_inputs(cache, np.arange(len(folds)))
    xy = observed_landmarks(cache, geometry)
    entries = []
    for fold in range(5):
        key = str(fold)
        folder = args.runs / f'fold-{fold}'
        grade = comparison['folds'][key]
        learned = audit['held_out_localization'][key]
        recorded = read_json(folder / 'result.json')['config']
        if (checksum(folder / 'final.pt') != grade['checkpoint_sha256']
                or checksum(folder / 'oof.csv') != grade['prediction_sha256']
                or recorded['fold'] != fold or recorded['region_sha256'] != learned['region_sha256']):
            raise ValueError('Stored model artifacts do not match the measured grading result')
        fit = np.flatnonzero(folds != fold)
        center = args.learned / 'models' / f'exclude-{fold}' / 'final.pt'
        center_hash = learned['checkpoints'][key]
        if checksum(center) != center_hash:
            raise ValueError('The held-out localizer differs from the cross-fit report')
        entries.append({'fold': fold, 'heatmap_sha256': center_hash, 'grader_sha256': grade['checkpoint_sha256'],
                        'reference_localizer_sha256': checksum(args.reference / f'localizer-{fold}.npz'),
                        'offsets': anatomy_offsets(cache, xy, geometry, fit).tolist()})
    write_json(args.output, {'reference_fraction': .5, 'folds_sha256': fold_hash,
                             'comparison_sha256': checksum(args.comparison),
                             'reference_selection_sha256': checksum(args.reference_selection),
                             'reference_features_sha256': checksum(args.reference / 'reference-features.npy'),
                             'models': entries})
    print('Locked the five measured graders and their geometry for inference.', flush=True)


def infer(args):
    torch.set_num_threads(4)
    plan = read_json(args.plan)
    reference_plan = read_json(args.prior.with_suffix('.json'))
    if (reference_plan['selection_sha256'] != plan['reference_selection_sha256']
            or reference_plan['prediction_sha256'] != checksum(args.prior)
            or checksum(args.reference / 'reference-features.npy') != plan['reference_features_sha256']):
        raise ValueError('Reference model output or geometry differs from the deployment lock')
    if [entry['fold'] for entry in plan['models']] != list(range(5)):
        raise ValueError('Expected all five locked MRI models in fold order')
    prepare(args.root, args.cache, 'test', 4)
    cache = SpineCache.load(args.cache)
    rows = np.arange(len(cache.studies))
    reference = read_submission(args.prior, cache.studies)
    labels = np.full((len(rows), 25), -1, dtype=np.int64)
    inputs, geometry = anchor_inputs(cache, rows)
    folds = []
    first_regions = None
    for entry in plan['models']:
        fold = entry['fold']
        heatmap = args.learned / 'models' / f'exclude-{fold}' / 'final.pt'
        graded = args.runs / f'fold-{fold}' / 'final.pt'
        if (checksum(heatmap) != entry['heatmap_sha256'] or checksum(graded) != entry['grader_sha256']
                or checksum(args.reference / f'localizer-{fold}.npz') != entry['reference_localizer_sha256']):
            raise ValueError('A model artifact differs from the deployment lock')
        state = torch.load(heatmap, map_location='cpu', weights_only=True)
        localizer = DiscHeatmap().cuda().eval()
        localizer.load_state_dict(state['state'])
        points = predict_heatmaps(localizer, inputs, args.batch)
        del localizer
        _, old_metadata = inference_regions(cache, args.reference, fold)
        regions, metadata, _ = make_regions(cache, rows, geometry, points, old_metadata[:, :5, 1],
                                            {'offsets': np.asarray(entry['offsets'], dtype=np.float32)})
        state = torch.load(graded, map_location='cpu', weights_only=True)
        model = SpineClassifier(pretrained=False, spatial_detail=True).cuda()
        model.load_state_dict(state['model'])
        dataset = RegionDataset(cache, rows, regions, metadata, labels)
        probs, _ = predict(model, DataLoader(dataset, batch_size=args.batch, pin_memory=True))
        folds.append(probs)
        if args.trace and first_regions is None:
            first_regions = {'fold': fold, 'coordinates': regions[:, :25].tolist(), 'quality': metadata[:, :25].tolist()}
        del model
    model_probabilities = np.mean(folds, axis=0, dtype=np.float64)
    fraction = plan['reference_fraction']
    blended = fraction * reference + (1 - fraction) * model_probabilities
    desired = pd.read_csv(args.root / 'sample_submission.csv')
    result = submission(cache.studies, blended)
    if set(result.row_id) != set(desired.row_id) or list(desired.columns) != ['row_id', *SEVERITIES]:
        raise ValueError('Final MRI predictions differ from the requested target rows')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.set_index('row_id').loc[desired.row_id].reset_index().to_csv(args.output, index=False)
    np.testing.assert_allclose(read_submission(args.output, cache.studies), blended, atol=1e-8)
    digest = checksum(args.plan)
    report = {'selection_sha256': digest, 'prediction_sha256': checksum(args.output), 'rows': len(result),
              'studies': len(rows), 'selected': 'equal blend of reference and spatial-detail MRI graders'}
    write_json(args.output.with_suffix('.json'), report)
    if args.trace:
        write_json(args.trace, {'task': 'spine', 'selection_sha256': digest, 'studies': cache.studies.tolist(),
                                'targets': list(TARGETS), 'reference_fraction': fraction, 'new_model_fraction': 1 - fraction,
                                'reference_probabilities': reference.tolist(), 'new_model_probabilities': model_probabilities.tolist(),
                                'folds': list(range(5)), 'fold_probabilities': np.asarray(folds).tolist(),
                                'blended_probabilities': blended.tolist(), 'first_fold_regions': first_regions})
    print(f"Verified {len(result)} prediction rows from the locked MRI ensemble.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    shared = ('learned', 'reference', 'runs', 'output')
    build = sub.add_parser('lock')
    run = sub.add_parser('predict')
    for name in (*shared, 'train-cache', 'folds', 'comparison', 'reference-selection'):
        build.add_argument(f'--{name}', type=Path, required=True)
    for name in (*shared, 'plan', 'root', 'cache', 'prior'):
        run.add_argument(f'--{name}', type=Path, required=True)
    run.add_argument('--batch', type=int, default=8)
    run.add_argument('--trace', type=Path)
    args = parser.parse_args()
    if args.phase == 'lock':
        lock(args)
    else:
        if args.batch < 1:
            parser.error('Batch size must be positive')
        outcomes = [args.output.resolve(), args.output.with_suffix('.json').resolve()]
        if args.trace and args.trace.resolve() in outcomes:
            parser.error('Use different paths for the submission, summary and trace')
        infer(args)


if __name__ == '__main__':
    main()
