from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from medimage.common import checksum, read_json, write_json
from medimage.spine_localize import read_folds
from medimage.spine_metrics import (apply_temperature, bootstrap_delta, fit_temperature, read_submission,
                                    read_targets, score, submission)


def prior_predictions(labels: np.ndarray, folds: np.ndarray) -> np.ndarray:
    result = np.empty((*labels.shape, 3), dtype=np.float64)
    for fold in range(5):
        train = labels[folds != fold]
        counts = (train[..., None] == np.arange(3)).sum(axis=0) + 1
        result[folds == fold] = counts / counts.sum(axis=-1, keepdims=True)
    return result


def evaluate(args: argparse.Namespace) -> None:
    studies = np.sort(pd.read_csv(args.root / 'train.csv', usecols=['study_id']).study_id.to_numpy(np.int64))
    fold_hash = checksum(args.folds)
    folds = read_folds(args.folds, studies)
    labels = read_targets(args.root, studies)
    localized = np.empty((*labels.shape, 3), np.float64)
    args.output.mkdir(parents=True, exist_ok=True)
    experiments = []
    for fold in range(5):
        path = args.runs / f'fold-{fold}'
        result = read_json(path / 'result.json')
        config = result['config']
        if config['fold'] != fold or config['folds_sha256'] != fold_hash or config['limit'] is not None:
            raise ValueError('Spine assessment must include five complete compatible folds')
        mask = folds == fold
        localized[mask] = read_submission(path / 'oof.csv', studies[mask])
        reloaded = score(labels[mask], localized[mask])
        if abs(reloaded['score'] - result['validation']['score']) > 1e-7:
            raise ValueError('Stored fold probabilities cannot reproduce the recorded score')
        experiments.append(result)
    prior = prior_predictions(labels, folds)
    predictions = {'prior': prior, 'localized': localized}
    if args.baseline_oof is not None:
        baseline = read_submission(args.baseline_oof, studies)
        predictions['global_reference'] = baseline
        predictions['equal_weight_blend'] = (localized + baseline) / 2.
    calibrated = np.empty_like(localized)
    per_fold_temperatures = []
    for fold in range(5):
        train, held = folds != fold, folds == fold
        temperature = fit_temperature(labels[train], localized[train])
        calibrated[held] = apply_temperature(localized[held], temperature)
        per_fold_temperatures.append(temperature.tolist())
    assessment = {name: {'overall': score(labels, predictions_array),
                         'per_fold': [score(labels[folds == fold], predictions_array[folds == fold]) for fold in range(5)]}
                  for name, predictions_array in predictions.items()}
    selected = 'localized'
    if args.baseline_oof is not None:
        selected = 'global_reference'
        baseline_scores = [item['score'] for item in assessment[selected]['per_fold']]
        for name in ('localized', 'equal_weight_blend'):
            candidate = assessment[name]
            candidate['paired_delta_ci95_vs_global'] = bootstrap_delta(labels, predictions[name], baseline)
            candidate['folds_beating_global'] = sum(item['score'] < reference for item, reference in
                                                  zip(candidate['per_fold'], baseline_scores, strict=True))
            if (candidate['paired_delta_ci95_vs_global'][1] < 0 and candidate['folds_beating_global'] >= 3
                    and candidate['overall']['score'] < assessment[selected]['overall']['score']):
                selected = name
    report = {
        'selected_for_local_inference': selected,
        'promotion_rule': 'Lower full OOF loss, at least three folds better, and upper paired bootstrap CI below zero',
        'studies': len(studies), 'targets': int(labels.size), 'missing_labels': int((labels < 0).sum()),
        'folds_sha256': fold_hash, 'assessments': assessment, 'fold_experiments': experiments,
        'exploratory_cross_fold_calibration': {'temperature_by_held_fold': per_fold_temperatures,
                                               'score': score(labels, calibrated)},
        'localization': read_json(args.localization / 'localization.json'),
    }
    for name, array in predictions.items():
        submission(studies, array).to_csv(args.output / f'{name}-oof.csv', index=False)
    write_json(args.output / 'metrics.json', report)
    print({name: record['overall']['score'] for name, record in assessment.items()}, flush=True)
    print(f'Selected {selected}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Five-fold RSNA assessment, prespecified blending and bootstrap')
    for name in ('root', 'folds', 'runs', 'localization', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--baseline-oof', type=Path)
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
