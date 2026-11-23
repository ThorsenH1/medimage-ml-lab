from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

from medimage.spine_data import LABEL_MAP, SEVERITIES, TARGETS

SAMPLE_WEIGHTS = np.array([1., 2., 4.])
COMPONENTS = ('spinal', 'foraminal', 'subarticular', 'any_severe_spinal')
GROUPS = (slice(0, 5), slice(5, 15), slice(15, 25))


def read_targets(root: Path, studies: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(root / 'train.csv')
    if frame.study_id.duplicated().any() or list(frame.columns) != ['study_id', *TARGETS]:
        raise ValueError('Incorrect RSNA grade table schema')
    table = frame.set_index('study_id').loc[studies, list(TARGETS)]
    for label in table:
        if set(table[label].dropna()) - LABEL_MAP.keys():
            raise ValueError('Unexpected RSNA severity')
    return table.apply(lambda column: column.map(LABEL_MAP)).fillna(-1).to_numpy(np.int64, copy=True)


def check_probabilities(probabilities: np.ndarray) -> None:
    if (probabilities.ndim != 3 or probabilities.shape[1:] != (25, 3)
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=-1), 1., atol=1e-6)):
        raise ValueError('Expected finite [study, 25 targets, 3 severity grades] normalized probabilities')


def study_losses(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    check_probabilities(probabilities)
    if labels.shape != probabilities.shape[:2] or not np.isin(labels, [-1, 0, 1, 2]).all():
        raise ValueError('RSNA labels do not align with predictions')
    known = labels >= 0
    indexed = labels.clip(min=0)
    weight = SAMPLE_WEIGHTS[indexed] * known
    eps = np.finfo(np.float64).eps
    true_probability = np.take_along_axis(probabilities, indexed[..., None], axis=-1)[..., 0]
    weighted_neg_log_likelihood = -np.log(np.clip(true_probability, eps, 1)) * weight
    numerator = [weighted_neg_log_likelihood[:, family].sum(axis=1) for family in GROUPS]
    denominator = [weight[:, family].sum(axis=1) for family in GROUPS]
    p = np.where(known[:, :5], probabilities[:, :5, 2], 0).max(axis=1).astype(np.float64)
    p = np.clip(p, eps, 1 - eps)
    severe = (labels[:, :5] == 2).any(axis=1)
    any_weight = weight[:, :5].max(axis=1)
    numerator.append(-np.where(severe, np.log(p), np.log1p(-p)) * any_weight)
    denominator.append(any_weight)
    return np.column_stack(numerator), np.column_stack(denominator)


def score(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    numerator, denominator = study_losses(labels, probabilities.astype(np.float64))
    observed = denominator.sum(axis=0)
    if (observed <= 0).any():
        raise ValueError('RSNA score requires at least one observed target in each family')
    values = numerator.sum(axis=0) / observed
    return {'score': float(values.mean()), **dict(zip(COMPONENTS, values.tolist(), strict=True))}


def submission(studies: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    check_probabilities(probabilities)
    if len(studies) != len(probabilities) or len(studies) != len(set(studies)):
        raise ValueError('Prediction studies must be unique and aligned')
    result = pd.DataFrame(probabilities.reshape(-1, 3), columns=list(SEVERITIES))
    result.insert(0, 'row_id', [f'{study}_{target}' for study in studies for target in TARGETS])
    return result


def read_submission(path: Path, studies: np.ndarray) -> np.ndarray:
    table = pd.read_csv(path)
    expected_ids = [f'{study}_{target}' for study in studies for target in TARGETS]
    if list(table.columns) != ['row_id', *SEVERITIES] or table.row_id.duplicated().any() or set(table.row_id) != set(expected_ids):
        raise ValueError('RSNA rows do not match the specified cohort and target schema')
    values = table.set_index('row_id').loc[expected_ids, list(SEVERITIES)].to_numpy().reshape(-1, 25, 3)
    check_probabilities(values)
    return values


def apply_temperature(probabilities: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
    group_temperatures = temperatures[np.repeat(np.arange(3), [5, 10, 10])]
    return softmax(np.log(probabilities.clip(1e-7)) / group_temperatures[None, :, None], axis=-1)


def fit_temperature(truth: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    objective = lambda values: score(truth, apply_temperature(probabilities, np.exp(values)))['score'] + .008 * (values ** 2).mean()
    result = minimize(objective, np.zeros(3), method='L-BFGS-B', bounds=[(-.55, .55)] * 3)
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('RSNA temperature optimization failed')
    return np.exp(result.x)


def bootstrap_delta(labels: np.ndarray, candidate: np.ndarray, baseline: np.ndarray, repeats: int = 1000) -> list[float]:
    candidate_loss, weight = study_losses(labels, candidate.astype(np.float64))
    baseline_loss, _ = study_losses(labels, baseline.astype(np.float64))
    change = candidate_loss - baseline_loss
    rng = np.random.default_rng(199)
    deltas = []
    for _ in range(repeats):
        rows = rng.integers(len(labels), size=len(labels))
        deltas.append(float((change[rows].sum(axis=0) / weight[rows].sum(axis=0)).mean()))
    return np.percentile(deltas, [2.5, 97.5]).tolist()
