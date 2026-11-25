"""Test a fixed, narrower field of view for the lateral lumbar MRI targets."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from medimage.auc_spine import interval
from medimage.common import checksum, read_json, write_json
from medimage.spine_detail_infer import FIELD_FRACTION, LATERAL, ZOOM_BLEND, laterality_settings, smaller_field
from medimage.spine_geometry_probe import classification_summary
from medimage.spine_laterality import predict_reflections
from medimage.spine_localize import SpineCache, read_folds
from medimage.spine_metrics import read_submission, read_targets, submission
from medimage.spine_model import ROI_SIZE, SpineClassifier
from medimage.spine_train import RegionDataset, _load_fold, predict


def blend_lateral(baseline: np.ndarray, reference: np.ndarray, zoom: np.ndarray) -> np.ndarray:
    combined = baseline.copy()
    zoomed_ensemble = .5 * (reference[:, LATERAL] + zoom[:, LATERAL])
    combined[:, LATERAL] = (1 - ZOOM_BLEND) * baseline[:, LATERAL] + ZOOM_BLEND * zoomed_ensemble
    return combined


def score_crop_pass(args, cache, folds, files, base_shape, plan_sha, transform, name):
    blank_labels = np.full(base_shape[:2], -1, dtype=np.int64)
    predictions = np.empty(base_shape, dtype=np.float64)
    coverage = {}
    for fold in range(5):
        rows = np.flatnonzero(folds == fold)
        cached = args.output / f'fold-{fold}-{name}-oof.csv'
        note = cached.with_suffix('.json')
        if note.exists():
            saved = read_json(note)
            if saved['plan_sha256'] != plan_sha or saved['prediction_sha256'] != checksum(cached):
                raise ValueError('Cached crop predictions belong to another comparison')
            predictions[rows] = read_submission(cached, cache.studies[rows])
            coverage[str(fold)] = saved['reflected']
            print(f'Reusing measured {name} views for fold {fold}', flush=True)
            continue
        coordinates, quality = _load_fold(args.regions / f'fold-{fold}.npz', cache, folds, fold)
        coordinates = transform(coordinates)
        resolution = files[str(fold)]['resolution']
        saved = torch.load(args.runs / f'fold-{fold}' / 'final.pt', map_location='cpu', weights_only=True)
        model = SpineClassifier(pretrained=False, spatial_detail=True).cuda()
        model.load_state_dict(saved['model'])
        inputs = RegionDataset(cache, rows, coordinates, quality, blank_labels, resolution=resolution)
        p, _ = predict(model, DataLoader(inputs, batch_size=args.batch, pin_memory=True))
        eligible, reflected = predict_reflections(model, cache, rows, coordinates, quality, blank_labels,
                                                   resolution, args.batch)
        p[eligible] = .5 * (p[eligible] + reflected)
        predictions[rows] = p
        coverage[str(fold)] = int(eligible.sum())
        submission(cache.studies[rows], p).to_csv(cached, index=False)
        write_json(note, {'plan_sha256': plan_sha, 'prediction_sha256': checksum(cached),
                          'reflected': coverage[str(fold)]})
        print(f'Fold {fold}: scored {len(rows)} held-out studies with {name} crops', flush=True)
        del model
        torch.cuda.empty_cache()
    return predictions, coverage


def evaluate(args: argparse.Namespace) -> None:
    torch.set_num_threads(3)
    cache = SpineCache.load(args.cache)
    folds = read_folds(args.folds, cache.studies)
    fold_sha = checksum(args.folds)
    cache_sha = checksum(args.cache / 'audit.json')
    comparison_file = args.comparison / 'metrics.json'
    comparison = read_json(comparison_file)
    if comparison['baseline_sha256'] != checksum(args.reference):
        raise ValueError('Reference predictions differ from the original spatial comparison')
    files = {}
    for fold in range(5):
        run = args.runs / f'fold-{fold}'
        checkpoint = run / 'final.pt'
        region_file = args.regions / f'fold-{fold}.npz'
        model = read_json(run / 'result.json')['config']
        files[str(fold)] = {'region_sha256': checksum(region_file),
                            'checkpoint_sha256': checksum(checkpoint),
                            'oof_sha256': checksum(run / 'oof.csv'),
                            'resolution': model.get('resolution', ROI_SIZE)}
    approved = laterality_settings(args.laterality, comparison_file, comparison, fold_sha, cache_sha, files)

    baseline_path = args.laterality / 'equal-blend-oof.csv'
    baseline = read_submission(baseline_path, cache.studies)
    reference = read_submission(args.reference, cache.studies)
    plan = {'field_fraction': FIELD_FRACTION, 'zoom_fraction': ZOOM_BLEND,
            'source_predictions_sha256': checksum(baseline_path), 'reference_sha256': checksum(args.reference),
            'folds_sha256': fold_sha, 'cache_sha256': cache_sha, 'reflection': approved, 'models': files}
    stamp = args.output / 'comparison-plan.json'
    if stamp.exists() and read_json(stamp) != plan:
        raise ValueError('Use a new directory for a different crop and model comparison')
    write_json(stamp, plan)
    plan_sha = checksum(stamp)

    zoom, coverage = score_crop_pass(args, cache, folds, files, baseline.shape, plan_sha, smaller_field, 'zoom')

    truth = read_targets(args.root, cache.studies)
    result = blend_lateral(baseline, reference, zoom)
    control = classification_summary(truth, baseline)
    measured = read_json(args.laterality / 'metrics.json')['reflected_blend']
    if not np.isclose(control['macro_auroc'], measured['macro_auroc'], rtol=0, atol=1e-10):
        raise ValueError('Reflection baseline probabilities do not match their recorded score')
    report = {'plan_sha256': plan_sha, 'studies': len(truth), 'reflected_studies_by_fold': coverage,
              'baseline': control, 'zoom_blend': {**classification_summary(truth, result),
                                                **interval(truth, result, folds, args.bootstrap, reference=baseline)}}
    submission(cache.studies, result).to_csv(args.output / 'blend-oof.csv', index=False)
    write_json(args.output / 'metrics.json', report)
    print(report, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cache', 'folds', 'regions', 'runs', 'comparison', 'reference', 'laterality', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=3)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.batch < 1 or args.bootstrap < 1:
        parser.error('Use positive batch and bootstrap sizes')
    evaluate(args)


if __name__ == '__main__':
    main()
