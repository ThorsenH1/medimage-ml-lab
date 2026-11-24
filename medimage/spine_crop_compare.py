"""Compare old and learned MRI crops across the same held-out classifiers."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from medimage.auc_spine import interval
from medimage.common import checksum, read_json, write_json
from medimage.spine_geometry_probe import classification_summary
from medimage.spine_localize import SpineCache, read_folds
from medimage.spine_metrics import read_submission, read_targets, submission
from medimage.spine_model import SpineClassifier
from medimage.spine_train import RegionDataset, _load_fold, predict


def crop_scores(model, cache, paths, folds, fold, rows, truth, batch):
    region, metadata = _load_fold(paths / f'fold-{fold}.npz', cache, folds, fold)
    dataset = RegionDataset(cache, rows, region, metadata, truth)
    scores, observed = predict(model, DataLoader(dataset, batch_size=batch, pin_memory=True))
    if not np.array_equal(observed, truth[rows]):
        raise ValueError('The predicted rows do not match the frozen study labels')
    return scores


def compare(args: argparse.Namespace) -> None:
    torch.set_num_threads(4)
    cache = SpineCache.load(args.cache)
    study_fold = read_folds(args.folds, cache.studies)
    fold_hash = checksum(args.folds)
    cache_hash = checksum(args.cache / 'audit.json')
    for directory in (args.reference, args.learned):
        stamp = read_json(directory / 'localization.json')['config']
        if stamp['folds_file_sha256'] != fold_hash or stamp['cache_sha256'] != cache_hash:
            raise ValueError('Both localization sets must use the same frozen cache and study folds')
    y = read_targets(args.root, cache.studies)
    original = np.empty((*y.shape, 3), dtype=np.float64)
    proposed = np.empty_like(original)
    folds = {}
    for fold in range(5):
        rows = np.flatnonzero(study_fold == fold)
        weights = args.classifiers / f'fold-{fold}' / 'final.pt'
        state = torch.load(weights, map_location='cpu', weights_only=True)
        expected = {'fold': fold, 'folds_sha256': fold_hash, 'cache_sha256': cache_hash,
                    'region_sha256': checksum(args.reference / f'fold-{fold}.npz'), 'limit': None}
        if any(state['config'].get(name) != value for name, value in expected.items()):
            raise ValueError('Classifier checkpoint is incompatible with the original frozen fold')
        original[rows] = read_submission(args.classifiers / f'fold-{fold}' / 'oof.csv', cache.studies[rows])
        model = SpineClassifier(pretrained=False).cuda()
        model.load_state_dict(state['model'])
        proposed[rows] = crop_scores(model, cache, args.learned, study_fold, fold, rows, y, args.batch)
        del model
        folds[str(fold)] = {'studies': len(rows), 'classifier_sha256': checksum(weights),
                            'learned_region_sha256': checksum(args.learned / f'fold-{fold}.npz'),
                            'reference': classification_summary(y[rows], original[rows]),
                            'learned': classification_summary(y[rows], proposed[rows])}
        print(f"MRI fold {fold}: {folds[str(fold)]['learned']['macro_auroc']:.5f} AUROC", flush=True)
    report = {'studies': len(y), 'folds_sha256': fold_hash, 'folds': folds,
              'reference': {**classification_summary(y, original), **interval(y, original, study_fold, args.bootstrap)},
              'learned': {**classification_summary(y, proposed), **interval(y, proposed, study_fold, args.bootstrap)}}
    args.output.mkdir(parents=True, exist_ok=False)
    submission(cache.studies, proposed).to_csv(args.output / 'learned-oof.csv', index=False)
    write_json(args.output / 'metrics.json', report)
    print(f"Paired five-fold reference {report['reference']['macro_auroc']:.5f}; "
          f"learned {report['learned']['macro_auroc']:.5f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cache', 'folds', 'reference', 'learned', 'classifiers', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if min(args.batch, args.bootstrap) < 1:
        parser.error('Batch size and bootstrap count must be positive')
    compare(args)


if __name__ == '__main__':
    main()
