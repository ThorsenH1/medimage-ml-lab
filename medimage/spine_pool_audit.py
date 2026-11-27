"""Compare fixed probability pools for the two frozen MRI grading branches."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit, softmax

from medimage.auc_spine import interval
from medimage.common import checksum, read_json, write_json
from medimage.spine_data import TARGETS
from medimage.spine_geometry_probe import classification_summary
from medimage.spine_localize import read_folds
from medimage.spine_metrics import check_probabilities, read_submission, read_targets, submission

BOUNDARY_FRACTIONS = {'equal_factored': (.5, .5), 'original_presence': (.75, .25), 'native_presence': (.25, .75)}


def probability_pools(first: np.ndarray, second: np.ndarray) -> dict[str, np.ndarray]:
    check_probabilities(first)
    check_probabilities(second)
    if first.shape != second.shape:
        raise ValueError('The two graders must cover the same studies')
    both = np.stack((first, second))
    minimum = 1e-7
    geometric = softmax(np.log(both.clip(min=minimum)).mean(axis=0), axis=-1)

    # Each branch orders these two nested events, and a common monotone pool preserves that order.
    cumulative = np.stack((both[..., 1:].sum(axis=-1), both[..., 2]), axis=-1)
    at_least_moderate, severe = np.moveaxis(expit(logit(cumulative.clip(minimum, 1 - minimum)).mean(axis=0)), -1, 0)
    ordinal = np.stack((1 - at_least_moderate, at_least_moderate - severe, severe), axis=-1)
    check_probabilities(ordinal)
    return {'arithmetic': both.mean(axis=0), 'ordinal_log_odds': ordinal, 'geometric': geometric}


def factored_pool(first: np.ndarray, second: np.ndarray, method: str) -> np.ndarray:
    check_probabilities(first)
    check_probabilities(second)
    if first.shape != second.shape:
        raise ValueError('The two graders must cover the same studies')
    presence_weight, severity_weight = BOUNDARY_FRACTIONS[method]
    grades = np.stack((first, second))
    presence = grades[..., 1:].sum(axis=-1)
    severe_if_present = np.divide(grades[..., 2], presence, out=np.full_like(presence, .5), where=presence > 0)
    limits = (1e-7, 1 - 1e-7)
    odds_present = logit(presence.clip(*limits))
    odds_severe = logit(severe_if_present.clip(*limits))
    combined = expit(presence_weight * odds_present[0] + (1 - presence_weight) * odds_present[1])
    severe = expit(severity_weight * odds_severe[0] + (1 - severity_weight) * odds_severe[1])
    return np.stack((1 - combined, combined * (1 - severe), combined * severe), axis=-1)


def load_pool_inputs(args):
    studies = np.sort(pd.read_csv(args.root / 'train.csv', usecols=['study_id']).study_id.to_numpy(np.int64))
    folds = read_folds(args.folds, studies)
    fold_sha = checksum(args.folds)
    parent_path = args.native_comparison / 'metrics.json'
    plan_path = args.reflection / 'comparison-plan.json'
    metrics_path = args.reflection / 'metrics.json'
    parent, plan, recorded = (read_json(path) for path in (parent_path, plan_path, metrics_path))
    if (plan['folds_sha256'] != fold_sha or parent['folds_sha256'] != fold_sha
            or parent['baseline_sha256'] != checksum(args.original)
            or plan['comparison_sha256'] != checksum(parent_path) or recorded['plan_sha256'] != checksum(plan_path)
            or plan['detail_blend_weight'] != .5 or recorded['studies'] != len(studies)):
        raise ValueError('Expected the original-grid prediction and its locked equal-weight native reflection comparison')
    truth = read_targets(args.root, studies)
    baseline_path = args.reflection / 'equal-blend-oof.csv'
    baseline = read_submission(baseline_path, studies)
    previous = read_submission(args.original, studies)

    # The stored equal blend contains both reflected and unreflected native passes at their locked weights.
    native = np.clip(2 * baseline - previous, 0, 1)
    np.testing.assert_allclose(.5 * previous + .5 * native, baseline, atol=1e-10, rtol=0)
    check_probabilities(native)
    published = classification_summary(truth, baseline)
    for metric in ('macro_auroc', 'macro_average_precision', 'rsna_log_loss'):
        if not np.isclose(published[metric], recorded['reflected_blend'][metric], atol=1e-10, rtol=0):
            raise ValueError(f'Arithmetic control does not reproduce the published {metric}')
    provenance = {'studies': len(studies), 'fold_sha256': fold_sha, 'original_predictions_sha256': checksum(args.original),
                  'native_reflection_metrics_sha256': checksum(metrics_path),
                  'native_reflection_plan_sha256': checksum(plan_path), 'baseline_predictions_sha256': checksum(baseline_path)}
    return studies, folds, truth, previous, native, baseline, provenance


def run(args: argparse.Namespace) -> None:
    studies, folds, truth, previous, native, baseline, provenance = load_pool_inputs(args)
    results = {**provenance, 'selection': 'exploratory comparison of fixed formulas on established outer folds', 'pools': {}}
    args.output.mkdir(parents=True, exist_ok=True)
    pools = probability_pools(previous, native)
    pools['arithmetic'] = baseline
    for name, predictions in pools.items():
        summary = classification_summary(truth, predictions)
        if name != 'arithmetic':
            summary['paired_macro_auroc_delta'] = summary['macro_auroc'] - results['pools']['arithmetic']['macro_auroc']
            summary.update(interval(truth, predictions, folds, args.bootstrap, reference=baseline))
            submission(studies, predictions).to_csv(args.output / f'{name}-oof.csv', index=False)
        summary['fold_macro_auroc'] = {str(fold): classification_summary(truth[folds == fold],
                                                                         predictions[folds == fold])['macro_auroc']
                                       for fold in range(5)}
        results['pools'][name] = summary
        print(name, summary, flush=True)
    write_json(args.output / 'metrics.json', results)


def apply_pool(args: argparse.Namespace) -> None:
    report, plan, trace = (read_json(path) for path in (args.report, args.plan, args.trace))
    if (args.method not in report['pools'] or plan['folds_sha256'] != report['fold_sha256']
            or plan['laterality']['metrics_sha256'] != report['native_reflection_metrics_sha256']
            or trace['selection_sha256'] != checksum(args.plan) or trace['task'] != 'spine'
            or trace['targets'] != list(TARGETS) or trace['reference_fraction'] != .5 or trace['new_model_fraction'] != .5):
        raise ValueError('Use the prediction trace and report from the audited reflected ensemble')
    previous, native, baseline = (np.asarray(trace[key], dtype=np.float64)
                                  for key in ('reference_probabilities', 'new_model_probabilities', 'blended_probabilities'))
    pools = probability_pools(previous, native)
    np.testing.assert_allclose(pools['arithmetic'], baseline, atol=1e-8, rtol=0)
    pooled = factored_pool(previous, native, args.method) if args.method in BOUNDARY_FRACTIONS else pools[args.method]
    result = submission(np.asarray(trace['studies'], dtype=np.int64), pooled)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    write_json(args.output.with_suffix('.json'), {'pool': args.method, 'rows': len(result),
                                                 'selection_sha256': checksum(args.plan),
                                                 'report_sha256': checksum(args.report),
                                                 'prediction_sha256': checksum(args.output)})
    print(f'{args.method}: saved {len(result)} MRI prediction rows', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    compare = modes.add_parser('compare')
    for name in ('root', 'folds', 'original', 'native-comparison', 'reflection', 'output'):
        compare.add_argument(f'--{name}', required=True, type=Path)
    compare.add_argument('--bootstrap', type=int, default=1000)
    score = modes.add_parser('apply')
    for name in ('report', 'plan', 'trace', 'output'):
        score.add_argument(f'--{name}', required=True, type=Path)
    score.add_argument('--method', choices=('ordinal_log_odds', 'geometric', *BOUNDARY_FRACTIONS), default='geometric')
    args = parser.parse_args()
    if args.mode == 'apply':
        if args.output.suffix.lower() != '.csv':
            parser.error('Write pooled predictions to a .csv file')
        apply_pool(args)
        return
    if args.bootstrap < 1:
        parser.error('Use a positive bootstrap count')
    run(args)


if __name__ == '__main__':
    main()
