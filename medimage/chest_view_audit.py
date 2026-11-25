"""Check if acquisition-view offsets improve the locked chest model on grouped validation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from medimage.auc_selection import selected_scores
from medimage.chest_rank_blend import validation
from medimage.common import checksum, load_manifest, write_json
from medimage.radiography_evaluate import fit_calibration, patient_bootstrap, validation_metrics

VIEWS = ('AP', 'PA')
PRIOR_STRENGTH = 24.


def fit_offsets(logits: np.ndarray, truth: np.ndarray, projections: np.ndarray) -> dict[str, np.ndarray]:
    offsets = {}
    for name in VIEWS:
        rows = projections == name
        values = []
        for observed, labels in zip(logits[rows].T, truth[rows].T, strict=True):
            def gradient(offset):
                return float((expit(observed + offset) - labels).sum() + PRIOR_STRENGTH * offset)

            bracket = 1 + len(observed) / PRIOR_STRENGTH
            values.append(brentq(gradient, -bracket, bracket))
        offsets[name] = np.asarray(values)
    return offsets


def apply_offsets(logits: np.ndarray, views: np.ndarray, offsets: dict[str, np.ndarray]) -> np.ndarray:
    result = logits.copy()
    for name, delta in offsets.items():
        result[views == name] += delta
    return result


def evaluate(args: argparse.Namespace) -> None:
    base, truth, groups, _, _, components = validation(args)
    frame, _ = load_manifest(args.cohort)
    views = frame.loc[frame.split.eq('val'), 'view'].astype(str).str.upper().to_numpy()
    logits = selected_scores(base['selected'], components, base['component_calibration'])
    original = validation_metrics(truth, logits)
    if not np.isclose(original['macro_auroc'], base['selected']['metrics']['macro_auroc'], atol=1e-10, rtol=0):
        raise ValueError('The saved component scores do not reproduce the frozen validation winner')
    calibrated, adjusted = np.empty_like(logits), np.empty_like(logits)
    for fit, check in GroupKFold(n_splits=5).split(truth, groups=groups):
        parameters, _ = fit_calibration(truth[fit], logits[fit])
        transformed = logits * parameters[:, 0] + parameters[:, 1]
        offsets = fit_offsets(transformed[fit], truth[fit], views[fit])
        calibrated[check] = transformed[check]
        adjusted[check] = apply_offsets(transformed[check], views[check], offsets)
    metrics = {}
    for name, scores in (('frozen', logits), ('global_calibration', calibrated), ('view_offsets', adjusted)):
        row = validation_metrics(truth, scores)
        if name != 'frozen':
            row['mean_brier'] = float(np.square(expit(scores) - truth).mean())
        metrics[name] = row
    compared = patient_bootstrap(truth, adjusted, groups, repeats=args.bootstrap, reference=logits)
    report = {'base_lock_sha256': checksum(args.base_lock), 'validation_images': len(truth),
              'view_counts': {name: int((views == name).sum()) for name in VIEWS},
              'other_views': int((~np.isin(views, VIEWS)).sum()), 'intercept_prior_strength': PRIOR_STRENGTH,
              'validation': metrics, 'paired_bootstrap': compared}
    write_json(args.output, report)
    print({'validation': metrics, 'paired_auroc_delta_ci95': compared['paired_auroc_delta_ci95']}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-lock', 'cohort', 'work', 'previous', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.set_defaults(task='chest')
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('Use a positive bootstrap count')
    with threadpool_limits(limits=2):
        evaluate(args)


if __name__ == '__main__':
    main()
