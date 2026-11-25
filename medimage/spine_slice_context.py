"""Grade the side passages with an additional stack of more widely spaced axial slices."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from medimage.auc_spine import interval
from medimage.common import checksum, read_json, write_json
from medimage.spine_crop_support import SUBARTICULAR, wider_axial_stack
from medimage.spine_geometry_probe import classification_summary
from medimage.spine_lateral_zoom import score_crop_pass
from medimage.spine_localize import SpineCache, read_folds
from medimage.spine_metrics import read_submission, read_targets, submission
from medimage.spine_weight_audit import compensate

EXTRA_STACK_WEIGHT = .25


def evaluate(args: argparse.Namespace) -> None:
    torch.set_num_threads(3)
    cache = SpineCache.load(args.cache)
    folds = read_folds(args.folds, cache.studies)
    zoom_plan = args.zoom / 'comparison-plan.json'
    baseline_file = args.zoom / 'blend-oof.csv'
    prior = read_json(zoom_plan)
    files = prior['models']
    reported = read_json(args.grade_weight_report)
    support = read_json(args.support_report)
    if (prior['folds_sha256'] != checksum(args.folds) or prior['cache_sha256'] != checksum(args.cache / 'audit.json')
            or prior['reference_sha256'] != checksum(args.reference) or support['folds_sha256'] != prior['folds_sha256']
            or support['cache_sha256'] != prior['cache_sha256'] or reported['selected_sha256'] != checksum(baseline_file)):
        raise ValueError('The baseline and coverage audit must use the same frozen image data and folds')
    for fold, expected in files.items():
        region = checksum(args.regions / f'fold-{fold}.npz')
        checkpoint = checksum(args.runs / f'fold-{fold}' / 'final.pt')
        if expected['region_sha256'] != region or expected['checkpoint_sha256'] != checkpoint:
            raise ValueError('The graders and localization must be the same ones used for the narrow crops')
        if support['region_file_hashes']['learned'][fold] != region:
            raise ValueError('The region files differ from the measured coverage audit')

    baseline_unadjusted = read_submission(baseline_file, cache.studies)
    reference = read_submission(args.reference, cache.studies)
    baseline = compensate(baseline_unadjusted, reference)
    truth = read_targets(args.root, cache.studies)
    baseline_stats = classification_summary(truth, baseline)
    if not np.isclose(baseline_stats['macro_auroc'], reported['partially_corrected_lateral']['macro_auroc'],
                      rtol=0, atol=1e-10):
        raise ValueError('The grade-adjusted baseline differs from the recorded result')

    plan = {'zoom_plan_sha256': checksum(zoom_plan), 'zoom_predictions_sha256': checksum(baseline_file),
            'grade_weight_report_sha256': checksum(args.grade_weight_report), 'support_sha256': checksum(args.support_report),
            'extra_stack_weight': EXTRA_STACK_WEIGHT, 'subarticular_neighbors': [-2, 2],
            'code_sha256': checksum(Path(__file__).with_name('spine_crop_support.py'))}
    stamp = args.output / 'comparison-plan.json'
    if stamp.exists() and read_json(stamp) != plan:
        raise ValueError('Use a separate output directory for a different slice experiment')
    write_json(stamp, plan)
    plan_sha = checksum(stamp)
    wider, coverage = score_crop_pass(args, cache, folds, files, baseline.shape, plan_sha,
                                      lambda regions: wider_axial_stack(cache, regions), 'wide-axial')
    sub = SUBARTICULAR
    blend = baseline_unadjusted.copy()
    wide_ensemble = .5 * (reference[:, sub] + wider[:, sub])
    blend[:, sub] = (1 - EXTRA_STACK_WEIGHT) * blend[:, sub] + EXTRA_STACK_WEIGHT * wide_ensemble
    result = compensate(blend, reference)
    np.testing.assert_array_equal(result[:, :sub.start], baseline[:, :sub.start])
    report = {'plan_sha256': plan_sha, 'studies': len(truth), 'reflected_studies_by_fold': coverage,
              'baseline': baseline_stats, 'extra_axial_context': {**classification_summary(truth, result),
                                                                  **interval(truth, result, folds, args.bootstrap,
                                                                             reference=baseline)}}
    submission(cache.studies, result).to_csv(args.output / 'blend-oof.csv', index=False)
    write_json(args.output / 'metrics.json', report)
    print(report, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cache', 'folds', 'reference', 'regions', 'runs', 'zoom', 'grade-weight-report',
                 'support-report', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.batch < 1 or args.bootstrap < 1:
        parser.error('Use positive batch and bootstrap sizes')
    evaluate(args)


if __name__ == '__main__':
    main()
