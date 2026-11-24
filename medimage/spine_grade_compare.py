"""Score the five held-out folds of a fixed MRI grading experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from medimage.auc_spine import interval
from medimage.common import checksum, read_json, write_json
from medimage.spine_geometry_probe import classification_summary
from medimage.spine_localize import read_folds
from medimage.spine_metrics import read_submission, read_targets, score, submission


def compare(args: argparse.Namespace) -> None:
    studies = np.sort(pd.read_csv(args.root / 'train.csv', usecols=['study_id']).study_id.to_numpy(np.int64))
    assignments = read_folds(args.folds, studies)
    fold_sha256 = checksum(args.folds)
    y = read_targets(args.root, studies)
    existing = read_submission(args.baseline, studies)
    model = np.empty_like(existing)
    folds = {}
    for fold in range(5):
        directory = args.runs / f'fold-{fold}'
        path = directory / 'oof.csv'
        report = read_json(directory / 'result.json')
        config = report['config']
        if (config['fold'] != fold or config['folds_sha256'] != fold_sha256 or config['limit'] is not None
                or not config.get('spatial_detail') or config['epochs'] != 4
                or not config.get('parent_sha256')):
            raise ValueError('Expected a complete fixed-budget spatial-detail model on each outer fold')
        rows = np.flatnonzero(assignments == fold)
        model[rows] = read_submission(path, studies[rows])
        if abs(score(y[rows], model[rows])['score'] - report['validation']['score']) > 1e-7:
            raise ValueError('The held-out probabilities do not reproduce their fold score')
        folds[str(fold)] = {'studies': len(rows), 'prediction_sha256': checksum(path),
                            'checkpoint_sha256': checksum(directory / 'final.pt'),
                            'model': classification_summary(y[rows], model[rows])}
    equal_blend = .5 * existing + .5 * model
    summary = {'studies': len(y), 'folds_sha256': fold_sha256, 'folds': folds,
               'baseline_sha256': checksum(args.baseline),
               'comparison': 'Fixed spatial-detail model and an equal-weight research blend on the existing folds'}
    for key, probabilities in (('baseline', existing), ('detail', model), ('equal_blend', equal_blend)):
        summary[key] = {**classification_summary(y, probabilities),
                        **interval(y, probabilities, assignments, args.bootstrap)}
        print(key, summary[key], flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    submission(studies, model).to_csv(args.output / 'detail-oof.csv', index=False)
    submission(studies, equal_blend).to_csv(args.output / 'equal-blend-oof.csv', index=False)
    write_json(args.output / 'metrics.json', summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'folds', 'runs', 'baseline', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('Bootstrap count must be positive')
    compare(args)


if __name__ == '__main__':
    main()
