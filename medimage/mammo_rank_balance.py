"""Test lesion-balanced pairwise ranking using only breast-level validation data."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from medimage.auc_selection import (
    checked, fit_pairwise_differences, previous_validation, read_scores, selected_scores, stable_candidate,
)
from medimage.common import checksum, load_manifest, read_json, write_json
from medimage.radiography_evaluate import fit_calibration, patient_bootstrap, validation_metrics

LESION_TYPES = ('calc', 'mass')
WEIGHT_PENALTY = .12
SUBTYPE_SHARE = .50


def ranking_pairs(matrix, positive, types, balance):
    difference = matrix[positive, None] - matrix[None, ~positive]
    pair_count = np.prod(difference.shape[:2])
    if not pair_count:
        raise ValueError('Both validation labels are needed to form ranking pairs')
    matrix_weights = np.full(difference.shape[:2], (1 - balance) / pair_count)
    for name in LESION_TYPES:
        within = (types[positive, None] == name) & (types[None, ~positive] == name)
        if not within.any():
            raise ValueError(f'Both validation labels are needed for {name}')
        matrix_weights[within] += balance / (len(LESION_TYPES) * within.sum())
    return difference.reshape(-1, matrix.shape[1]), matrix_weights.ravel()


def score_recipe(truth, types, logits):
    cohorts = {name: validation_metrics(truth[types == name], logits[types == name]) for name in LESION_TYPES}
    return {**validation_metrics(truth, logits), 'subtype_metrics': cohorts,
            'worst_subtype_auroc': min(item['macro_auroc'] for item in cohorts.values())}


def validation_inputs(args):
    frame, audit = load_manifest(args.cohort)
    base = read_json(args.reference)
    if base['task'] != 'mammo' or base['manifest'] != audit['manifest_sha256'] or len(base['labels']) != 1:
        raise ValueError('The reference lock and mammography validation cohort differ')
    rows = np.flatnonzero(frame.split.eq('val').to_numpy())
    subset = frame.iloc[rows]
    if len(rows) != base['validation_rows']:
        raise ValueError('Expected the validation cohort used by the reference lock')
    previous = checked(args.previous, 'evaluation/mammo/selection.json',
                       base['candidates']['previous-locked']['prior_selection_sha256'])
    truth = subset[['positive']].to_numpy(np.int8)
    sources = {}
    for name in base['candidates']:
        if name == 'previous-locked':
            sources[name] = previous_validation(args, read_json(previous), rows, 1)
        else:
            entry = base['candidates'][name]
            directory = checked(args.work, entry['run'])
            checked(directory, 'best.pt', entry['checkpoint_sha256'])
            validation = checked(directory, 'validation.npz', entry['validation_sha256'])
            sources[name] = read_scores(validation, rows, num_labels=1)
    return base, subset, truth, sources


def spaces(sources, truth, fit):
    raw = np.column_stack(list(sources.values()))
    parameters = {name: fit_calibration(truth[fit], logits[fit])[0] for name, logits in sources.items()}
    slope, offset = np.vstack(list(parameters.values())).T
    calibrated = raw * slope + offset
    return {'raw': (raw, {}), 'calibrated': (calibrated, {key: value.tolist() for key, value in parameters.items()})}


def learn(matrix, truth, types, anchor, balance):
    difference, weights = ranking_pairs(matrix, truth[:, 0].astype(bool), types, balance)
    return fit_pairwise_differences(difference, WEIGHT_PENALTY, center=anchor, pair_weights=weights)


def anchor_weights(weights, sources):
    anchor = np.asarray([weights.get(name, 0.) for name in sources])
    return anchor / anchor.sum()


def run(args):
    base, frame, truth, sources = validation_inputs(args)
    groups, types = frame.patient.to_numpy(), frame.type.to_numpy()
    families = {key: entry.get('config', {}).get('original_sha256', entry.get('checkpoint_sha256', key))
                for key, entry in base['candidates'].items()}
    full_anchor = anchor_weights(base['selected']['weights'], sources)
    spaces_to_check = ('raw', 'calibrated')
    comparisons = ('nested_reference', 'pooled_pairs', 'lesion_balanced_pairs')
    recipes = {(space, name): np.empty_like(truth, dtype=np.float64)
               for space in spaces_to_check for name in comparisons}

    for fit, check in GroupKFold(n_splits=5).split(truth, groups=groups):
        # Refit the original blend inside the fitting partition too. The published weights saw every validation patient.
        inner_sources = {name: score[fit] for name, score in sources.items()}
        inner_reference = stable_candidate(truth[fit], inner_sources, groups[fit], families)
        anchor = anchor_weights(inner_reference['weights'], sources)
        for space, (matrix, _) in spaces(sources, truth, fit).items():
            fractions = (anchor, learn(matrix[fit], truth[fit], types[fit], anchor, 0.),
                         learn(matrix[fit], truth[fit], types[fit], anchor, SUBTYPE_SHARE))
            for name, weights in zip(comparisons, fractions, strict=True):
                recipes[space, name][check, 0] = matrix[check] @ weights

    metrics = {f'{space}-{name}': score_recipe(truth, types, logits)
               for (space, name), logits in recipes.items()}
    baseline = metrics['calibrated-nested_reference']
    published = base['selected']['metrics']['macro_auroc']
    candidates = [(space, name) for space in spaces_to_check for name in comparisons[1:]]
    ordered = sorted(candidates, key=lambda pair: metrics['-'.join(pair)]['macro_auroc'], reverse=True)
    eligible = [pair for pair in ordered
                if metrics['-'.join(pair)]['macro_auroc'] > max(published, baseline['macro_auroc'])
                and metrics['-'.join(pair)]['worst_subtype_auroc'] >= baseline['worst_subtype_auroc']]
    winner = '-'.join(eligible[0]) if eligible else None
    strongest = eligible[0] if eligible else ordered[0]
    paired = patient_bootstrap(truth, recipes[strongest], groups, repeats=1000,
                               reference=recipes['calibrated', 'nested_reference'])
    report = {'base_selection_sha256': checksum(args.reference), 'validation_breasts': len(truth),
              'validation_patients': len(np.unique(groups)),
              'subtype_counts': {name: int((types == name).sum()) for name in np.unique(types)},
              'weight_penalty': WEIGHT_PENALTY, 'additional_subtype_pair_share': SUBTYPE_SHARE,
              'anchor': 'original validation bagging refit inside each outer validation fitting partition',
              'published_validation_auroc': published, 'validation': metrics, 'selected_for_test': winner,
              'paired_validation': {'recipe': '-'.join(strongest), **paired}}
    path = args.output / 'ranking-validation.json'
    if path.exists() and read_json(path) != report:
        raise ValueError('Use a fresh directory for a different validation comparison')
    write_json(path, report)

    if winner:
        space, name = eligible[0]
        matrix, parameters = spaces(sources, truth, np.arange(len(truth)))[space]
        weights = learn(matrix, truth, types, full_anchor, SUBTYPE_SHARE if name == comparisons[-1] else 0.)
        fractions = {key: weight for key, weight in zip(sources, weights.tolist(), strict=True) if weight > 0}
        selected = {'method': f'{space}-{name.replace("_", "-")}-logit', 'weights': fractions,
                    'metrics': metrics[winner]}
        logits = selected_scores(selected, sources, parameters)
        calibration, thresholds = fit_calibration(truth, logits)
        locked = {**base, 'selected': selected, 'validation_ranking': [selected, base['selected']],
                  'criterion': 'patient-cross-fitted validation AUROC with a worst-subtype check',
                  'component_calibration': parameters, 'ensemble_calibration': calibration.tolist(),
                  'thresholds': thresholds.tolist(), 'source_lock_sha256': report['base_selection_sha256']}
        destination = args.output / 'selection.json'
        if destination.exists() and read_json(destination) != locked:
            raise ValueError('Use a fresh directory for a different test decision')
        write_json(destination, locked)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'cohort', 'work', 'previous', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.set_defaults(task='mammo')
    with threadpool_limits(limits=2):
        run(parser.parse_args())


if __name__ == '__main__':
    main()
