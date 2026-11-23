from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from medimage.spine_metrics import read_targets


def freeze(root: Path, output: Path) -> str:
    studies = pd.read_csv(root / 'train.csv', usecols=['study_id']).study_id.to_numpy(np.int64)
    labels = read_targets(root, studies)
    membership = (labels[..., None] == np.arange(3)).reshape(len(studies), -1).astype(np.int8)
    seed = 20240924
    splitter = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = np.full(len(studies), -1, dtype=np.int8)
    for fold, (_, validation) in enumerate(splitter.split(np.zeros((len(studies), 1), dtype=np.uint8), membership)):
        folds[validation] = fold
    if (folds < 0).any():
        raise ValueError('Study was not assigned to a frozen fold')
    cohort = pd.DataFrame({'study_id': studies, 'fold': folds,
                           'split_version': 'v1_iterative_multilabel', 'seed': seed})
    output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output, index=False)
    canonical = cohort.sort_values('study_id').to_csv(index=False, lineterminator='\n').encode()
    return hashlib.sha256(canonical).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description='Reproduce the original RSNA five-study-fold assignments')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(freeze(args.root, args.output))


if __name__ == '__main__':
    main()
