"""Measure how the frozen MRI scores distinguish neighboring severity grades."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from medimage.auc_spine import auc_cells
from medimage.common import checksum, write_json
from medimage.spine_data import LEVELS, SEVERITIES, TARGETS
from medimage.spine_localize import SpineCache
from medimage.spine_metrics import read_submission, read_targets


def summarize(truth: np.ndarray, probabilities: np.ndarray) -> list[dict]:
    auroc, _, _ = auc_cells(truth, probabilities)
    output = []
    width = len(LEVELS)
    for start in range(0, len(TARGETS), width):
        cells = list(range(start, start + width))
        name = TARGETS[start].removesuffix(f'_{LEVELS[0]}')
        comparisons = {}
        for other in (0, 2):
            raw_auc, conditional_auc = [], []
            for cell in cells:
                included = np.isin(truth[:, cell], [1, other])
                binary = truth[included, cell] == 1
                if not binary.any() or binary.all():
                    continue
                moderate = probabilities[included, cell, 1]
                denominator = moderate + probabilities[included, cell, other]
                conditional = np.divide(moderate, denominator, out=np.full_like(moderate, .5), where=denominator > 0)
                raw_auc.append(roc_auc_score(binary, moderate))
                conditional_auc.append(roc_auc_score(binary, conditional))
            if raw_auc:
                comparisons[SEVERITIES[other]] = {
                    'levels': len(raw_auc),
                    'moderate_score_auroc': float(np.mean(raw_auc)),
                    'conditional_score_auroc': float(np.mean(conditional_auc)),
                }
        output.append({'condition': name, 'macro_auroc': float(np.nanmean(auroc[cells])),
                       'auroc_by_grade': dict(zip(SEVERITIES, np.nanmean(auroc[cells], axis=0).tolist(), strict=True)),
                       'moderate_vs': comparisons})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cache', 'probabilities', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    cache = SpineCache.load(args.cache)
    truth = read_targets(args.root, cache.studies)
    probabilities = read_submission(args.probabilities, cache.studies)
    values = summarize(truth, probabilities)
    write_json(args.output, {'prediction_sha256': checksum(args.probabilities),
                             'label_sha256': checksum(args.root / 'train.csv'),
                             'studies': len(truth), 'results': values})
    for row in values:
        print(row, flush=True)


if __name__ == '__main__':
    main()
