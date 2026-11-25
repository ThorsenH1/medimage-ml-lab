"""Test whether associations between adjacent disc levels improve held-out grading."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.special import softmax

from medimage.auc_spine import interval
from medimage.common import checksum, read_json, write_json
from medimage.spine_data import LEVELS, TARGETS
from medimage.spine_geometry_probe import classification_summary
from medimage.spine_localize import SpineCache, read_folds
from medimage.spine_metrics import check_probabilities, read_submission, read_targets
from medimage.spine_weight_audit import compensate

PAIRS = tuple((target, target + 1) for target in range(len(LEVELS), len(TARGETS) - 1)
              if (target + 1) % len(LEVELS))
CONTEXT_WEIGHT = .20
INDEPENDENT_PRIOR_COUNT = 24.


def fit_lift(labels: np.ndarray) -> np.ndarray:
    lifts = []
    for left, right in PAIRS:
        available = (labels[:, left] >= 0) & (labels[:, right] >= 0)
        pair = 3 * labels[available, left] + labels[available, right]
        joint = np.bincount(pair, minlength=9).reshape(3, 3).astype(np.float64)
        left_prior, right_prior = joint.sum(axis=1) + 1, joint.sum(axis=0) + 1
        independent = np.outer(left_prior / left_prior.sum(), right_prior / right_prior.sum())
        joint += INDEPENDENT_PRIOR_COUNT * independent
        joint /= joint.sum()
        lifts.append(joint / np.outer(joint.sum(axis=1), joint.sum(axis=0)))
    return np.stack(lifts)


def apply_context(scores: np.ndarray, lift: np.ndarray) -> np.ndarray:
    adjustment = np.zeros_like(scores)
    for (left, right), association in zip(PAIRS, lift, strict=True):
        adjustment[:, left] += np.log((scores[:, right] @ association.T).clip(.5, 2.))
        adjustment[:, right] += np.log((scores[:, left] @ association).clip(.5, 2.))
    lateral = slice(len(LEVELS), None)
    enriched = scores.copy()
    enriched[:, lateral] = softmax(np.log(scores[:, lateral].clip(1e-9)) +
                                  CONTEXT_WEIGHT * adjustment[:, lateral], axis=-1)
    check_probabilities(enriched)
    return enriched


def evaluate(args: argparse.Namespace) -> None:
    cache = SpineCache.load(args.cache)
    folds = read_folds(args.folds, cache.studies)
    compare = read_json(args.zoom / 'comparison-plan.json')
    weights = read_json(args.grade_weight_report)
    predictions = args.zoom / 'blend-oof.csv'
    if (compare['folds_sha256'] != checksum(args.folds)
            or compare['cache_sha256'] != checksum(args.cache / 'audit.json')
            or compare['reference_sha256'] != checksum(args.reference)
            or weights['selected_sha256'] != checksum(predictions)):
        raise ValueError('The context audit requires the frozen grading predictions and study folds')
    truth = read_targets(args.root, cache.studies)
    scores = compensate(read_submission(predictions, cache.studies), read_submission(args.reference, cache.studies))
    baseline = classification_summary(truth, scores)
    if not np.isclose(baseline['macro_auroc'], weights['partially_corrected_lateral']['macro_auroc'],
                      rtol=0, atol=1e-10):
        raise ValueError('The input probabilities do not match the measured grade-adjusted model')
    enriched = np.empty_like(scores)
    associations = {}
    for fold in range(5):
        held_out = folds == fold
        learned = fit_lift(truth[~held_out])
        enriched[held_out] = apply_context(scores[held_out], learned)
        associations[str(fold)] = learned.tolist()
    report = {'studies': len(truth), 'folds_sha256': checksum(args.folds),
              'grade_weight_report_sha256': checksum(args.grade_weight_report),
              'train_labels_sha256': checksum(args.root / 'train.csv'),
              'context_weight': CONTEXT_WEIGHT, 'independent_prior_count': INDEPENDENT_PRIOR_COUNT,
              'lateral_pairs': [[TARGETS[a], TARGETS[b]] for a, b in PAIRS], 'by_fold_lift': associations,
              'baseline': baseline, 'adjacent_level_context': {**classification_summary(truth, enriched),
                                                               **interval(truth, enriched, folds, args.bootstrap,
                                                                          reference=scores)}}
    write_json(args.output, report)
    print(report['adjacent_level_context'], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cache', 'folds', 'reference', 'zoom', 'grade-weight-report', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('Use a positive bootstrap count')
    evaluate(args)


if __name__ == '__main__':
    main()
