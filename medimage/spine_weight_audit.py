"""Test fixed, partial compensation for severity-weighted training on side targets."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from medimage.auc_spine import interval
from medimage.common import checksum, read_json, write_json
from medimage.spine_detail_infer import GRADE_WEIGHT_EXPONENT, LATERAL, adjust_lateral_grades
from medimage.spine_geometry_probe import classification_summary
from medimage.spine_localize import SpineCache, read_folds
from medimage.spine_metrics import SAMPLE_WEIGHTS, check_probabilities, read_submission, read_targets


def compensate(selected: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Partially undo the grader's label weighting and retain the original reference blend."""
    spatial = 2 * selected - reference
    lateral = spatial[:, LATERAL]
    if (lateral < -1e-7).any() or not np.allclose(lateral.sum(axis=-1), 1., atol=1e-6):
        raise ValueError('Predictions do not reconstruct an equal reference–grader blend')
    corrected = adjust_lateral_grades(spatial)
    result = selected.copy()
    result[:, LATERAL] = .5 * (reference[:, LATERAL] + corrected[:, LATERAL])
    check_probabilities(result)
    return result


def evaluate(args: argparse.Namespace) -> None:
    cache = SpineCache.load(args.cache)
    folds = read_folds(args.folds, cache.studies)
    comparison_file = args.zoom / 'comparison-plan.json'
    metric_file = args.zoom / 'metrics.json'
    comparison = read_json(comparison_file)
    measured = read_json(metric_file)
    if (comparison['reference_sha256'] != checksum(args.reference)
            or comparison['folds_sha256'] != checksum(args.folds)
            or comparison['cache_sha256'] != checksum(args.cache / 'audit.json')
            or measured['plan_sha256'] != checksum(comparison_file)):
        raise ValueError('Use the frozen model inputs and study folds from the narrow-crop experiment')
    baseline_file = args.zoom / 'blend-oof.csv'
    baseline = read_submission(baseline_file, cache.studies)
    reference = read_submission(args.reference, cache.studies)
    truth = read_targets(args.root, cache.studies)
    before = classification_summary(truth, baseline)
    if not np.isclose(before['macro_auroc'], measured['zoom_blend']['macro_auroc'], rtol=0, atol=1e-10):
        raise ValueError('Baseline probabilities disagree with the recorded AUROC')
    adjusted = compensate(baseline, reference)
    report = {'studies': len(truth), 'selected_sha256': checksum(baseline_file), 'baseline': before,
              'training_grade_weights': SAMPLE_WEIGHTS.tolist(), 'inverse_weight_exponent': GRADE_WEIGHT_EXPONENT,
              'partially_corrected_lateral': {**classification_summary(truth, adjusted),
                                              **interval(truth, adjusted, folds, args.bootstrap, reference=baseline)}}
    write_json(args.output, report)
    print(report, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cache', 'folds', 'reference', 'zoom', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('Use a positive bootstrap count')
    evaluate(args)


if __name__ == '__main__':
    main()
