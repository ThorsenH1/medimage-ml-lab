"""Describe the locked mammography benchmark by lesion type and available projections."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from medimage.auc_selection import read_scores
from medimage.common import checksum, load_manifest, read_json, write_json
from medimage.radiography_evaluate import patient_bootstrap, validation_metrics


def run(args):
    frame, audit = load_manifest(args.cohort)
    selection_file = args.benchmark / 'selection.json'
    selection = read_json(selection_file)
    benchmark = read_json(args.benchmark / 'metrics.json')
    selection_sha256 = checksum(selection_file)
    if (selection['task'] != 'mammo' or selection['manifest'] != audit['manifest_sha256']
            or selection_sha256 != benchmark['selection_sha256']):
        raise ValueError('The recorded mammography benchmark does not match the cohort or model lock')
    indices = np.flatnonzero(frame.split.eq('test').to_numpy())
    held = frame.iloc[indices]
    truth = held[['positive']].to_numpy(np.int8)
    scores = read_scores(args.benchmark / 'test-predictions.npz', indices, num_labels=1,
                         truth=truth, key='probabilities')
    reproduced = validation_metrics(truth, scores)['macro_auroc']
    if len(indices) != benchmark['test']['rows'] or not np.isclose(reproduced, benchmark['test']['macro_auroc'],
                                                                   atol=1e-12, rtol=0):
        raise ValueError('Saved predictions do not reproduce the benchmark')

    projection_count = held[['CC', 'MLO']].ne('').sum(axis=1).to_numpy()
    if not np.isin(projection_count, [1, 2]).all():
        raise ValueError('An evaluated breast has neither standard projection')
    types, groups = held.type.to_numpy(), held.patient.to_numpy()
    subsets = {}
    for label, count in (('one_projection', 1), ('two_projections', 2)):
        for kind, matches in (('all', np.ones(len(truth), dtype=bool)), ('calc', types == 'calc'), ('mass', types == 'mass')):
            included = (projection_count == count) & matches
            actual, predicted, patients = truth[included], scores[included], groups[included]
            metrics = validation_metrics(actual, predicted)
            uncertainty = patient_bootstrap(actual, predicted, patients, repeats=args.bootstrap)
            subsets[f'{label}_{kind}'] = {'breasts': len(actual), 'positive': int(actual.sum()), **metrics,
                                          'auroc_ci95': uncertainty['macro_auroc_ci95']}
    write_json(args.output, {'selection_sha256': selection_sha256, 'subsets': subsets})
    print(subsets, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort', 'benchmark', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('Use a positive bootstrap count')
    run(args)


if __name__ == '__main__':
    main()
