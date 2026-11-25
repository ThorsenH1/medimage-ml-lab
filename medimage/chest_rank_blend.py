"""Fit per-finding ranking blends with patient-disjoint validation predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from medimage.auc_selection import checked, fit_pairwise_weights, previous_validation, read_scores, selected_scores
from medimage.common import checksum, load_manifest, read_json, write_json
from medimage.radiography_data import CHEST_LABELS
from medimage.radiography_evaluate import fit_calibration, validation_metrics

STRENGTHS = (.03, .18, .70)
PAIR_BUDGET = 8192
RARE_CLASS_ANCHOR = 80


def fit_findings(truth, scores, anchors, strength):
    fractions = []
    for label, target in enumerate(truth.T):
        support = min(int(target.sum()), len(target) - int(target.sum()))
        if support == 0:
            fractions.append(anchors)
            continue
        regularization = strength * (1 + RARE_CLASS_ANCHOR / support)
        fractions.append(fit_pairwise_weights(target[:, None], scores[:, label], regularization,
                                               center=anchors, max_pairs=PAIR_BUDGET, seed=8119 + label))
    return np.stack(fractions).T


def validation(args):
    manifest, audit = load_manifest(args.cohort)
    base = read_json(args.base_lock)
    if (base['task'] != 'chest' or base['manifest'] != audit['manifest_sha256']
            or base['labels'] != list(CHEST_LABELS)):
        raise ValueError('Base lock and chest cohort describe different endpoints')
    rows = np.flatnonzero(manifest.split.eq('val').to_numpy())
    if len(rows) != base['validation_rows']:
        raise ValueError('Base lock was selected on a different validation partition')
    truth = manifest.iloc[rows][list(CHEST_LABELS)].to_numpy(np.int8)
    groups = manifest.iloc[rows]['pixel_cluster'].to_numpy()
    names = list(base['selected']['weights'])
    if len(names) < 2:
        raise ValueError('Ranking reweighting requires at least two verified finalists')
    anchors = np.asarray([base['selected']['weights'][name] for name in names], dtype=np.float64)
    anchors /= anchors.sum()
    source = args.previous / 'evaluation/chest/selection.json'
    if checksum(source) != base['candidates']['previous-locked']['prior_selection_sha256']:
        raise ValueError('Earlier chest model lock changed')
    scores = {}
    for name in names:
        if name == 'previous-locked':
            scores[name] = previous_validation(args, read_json(source), rows, len(CHEST_LABELS))
        else:
            member = base['candidates'][name]
            folder = checked(args.work, member['run'])
            checked(folder, 'best.pt', member['checkpoint_sha256'])
            values = checked(folder, 'validation.npz', member['validation_sha256'])
            scores[name] = read_scores(values, rows, num_labels=len(CHEST_LABELS))
    return base, truth, groups, names, anchors, scores


def calibrated_stack(scores, truth, fit):
    parameters = {name: fit_calibration(truth[fit], score[fit])[0] for name, score in scores.items()}
    values = np.stack([raw * parameters[name][:, 0] + parameters[name][:, 1]
                       for name, raw in scores.items()], axis=-1)
    return values, {name: value.tolist() for name, value in parameters.items()}


def lock(args):
    base, truth, groups, names, anchors, scores = validation(args)
    original = base['selected']
    arrangements = list(GroupKFold(n_splits=5).split(truth, groups=groups))
    predictions = {value: np.empty_like(truth, dtype=np.float64) for value in STRENGTHS}
    for fit, check in arrangements:
        calibrated, _ = calibrated_stack(scores, truth, fit)
        for strength in STRENGTHS:
            fractions = fit_findings(truth[fit], calibrated[fit], anchors, strength)
            predictions[strength][check] = np.sum(calibrated[check] * fractions.T, axis=-1)
    calibrated, parameters = calibrated_stack(scores, truth, np.arange(len(truth)))
    candidates = []
    for strength in STRENGTHS:
        final = fit_findings(truth, calibrated, anchors, strength)
        candidates.append({'method': 'calibrated-finding-rank-logit', 'weight_strength': strength,
                           'weights': original['weights'],
                           'label_weights': {name: item.tolist() for name, item in zip(names, final, strict=True)},
                           'metrics': validation_metrics(truth, predictions[strength])})
    ordered = sorted(candidates, key=lambda item: (item['metrics']['macro_auroc'], item['metrics']['macro_auprc']),
                     reverse=True)
    winner = ordered[0]
    improved = winner['metrics']['macro_auroc'] > original['metrics']['macro_auroc']
    comparison = {'base_lock_sha256': checksum(args.base_lock), 'base': original, 'ranked': ordered,
                  'validation_images': len(truth), 'patient_groups': len(np.unique(groups)),
                  'max_pairs_per_finding': PAIR_BUDGET, 'selected_ranking_blend': improved}
    summary = args.output / 'ranking-validation.json'
    if summary.exists() and read_json(summary) != comparison:
        raise ValueError('Use a fresh directory for a different validation comparison')
    write_json(summary, comparison)
    if improved:
        joint = selected_scores(winner, scores, parameters)
        calibration, thresholds = fit_calibration(truth, joint)
        frozen = {**base, 'source_lock_sha256': comparison['base_lock_sha256'],
                  'criterion': 'group cross-fitted validation macro AUROC against the existing finalist blend',
                  'component_calibration': parameters, 'ensemble_calibration': calibration.tolist(),
                  'thresholds': thresholds.tolist(), 'validation_ranking': [*ordered, original],
                  'selected': winner}
        destination = args.output / 'selection.json'
        if destination.exists() and read_json(destination) != frozen:
            raise ValueError('Use a fresh directory for a different test decision')
        write_json(destination, frozen)
    print({'baseline': original['metrics'], 'best_ranked': winner['metrics'], 'locked_for_test': improved}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-lock', 'work', 'previous', 'cohort', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.set_defaults(task='chest')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        lock(args)


if __name__ == '__main__':
    main()
