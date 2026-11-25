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


def progress_figure(earlier: list[dict], current: list[dict], path: Path) -> None:
    from medimage.portfolio_figures import BLUE, TEAL, plt, write_figure

    conditions = [record['condition'] for record in earlier]
    if conditions != [record['condition'] for record in current] or len(conditions) != 5:
        raise ValueError('The MRI audits must cover the same five conditions in the same order')
    names = ['Central canal', 'Left foraminal', 'Right foraminal', 'Left subarticular', 'Right subarticular']
    rows = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), sharey=True, layout='constrained')
    fig.suptitle('MRI performance before and after left–right reflection', x=.025, ha='left', fontsize=17,
                 fontweight='bold')
    values = (
        ('All three grades', [record['macro_auroc'] for record in earlier],
         [record['macro_auroc'] for record in current]),
        ('Moderate vs. the other grades', [record['auroc_by_grade']['moderate'] for record in earlier],
         [record['auroc_by_grade']['moderate'] for record in current]),
    )
    for axis, (title, before, after) in zip(axes, values, strict=True):
        axis.set_title(title, loc='left', fontsize=11)
        axis.axvline(.90, linestyle='--', color='#68848a', linewidth=1)
        axis.hlines(rows, before, after, color='#9aaaae', linewidth=1.25)
        axis.scatter(before, rows, color=BLUE, marker='o', s=35, label='Original crops')
        axis.scatter(after, rows, color=TEAL, marker='D', s=35, label='Original and reflected')
        for score, row in zip(after, rows, strict=True):
            axis.text(score + .004, row, f'{score:.4f}', va='center', fontsize=8)
        axis.set_xlim(.77, .96)
        axis.set_xlabel('Macro AUROC across five levels')
        axis.grid(axis='x', color='#e3eaeb', linewidth=.6)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(rows, names)
    axes[0].invert_yaxis()
    axes[1].legend(loc='lower right', frameon=False, fontsize=8)
    write_figure(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cache', 'probabilities', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--reference-audit', type=Path)
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    if bool(args.reference_audit) != bool(args.figure):
        parser.error('Provide both the reference audit and an output figure')
    cache = SpineCache.load(args.cache)
    truth = read_targets(args.root, cache.studies)
    probabilities = read_submission(args.probabilities, cache.studies)
    values = summarize(truth, probabilities)
    write_json(args.output, {'prediction_sha256': checksum(args.probabilities),
                             'label_sha256': checksum(args.root / 'train.csv'),
                             'studies': len(truth), 'results': values})
    if args.figure:
        from medimage.common import read_json

        previous = read_json(args.reference_audit)
        if previous['label_sha256'] != checksum(args.root / 'train.csv') or previous['studies'] != len(truth):
            raise ValueError('The reference audit covers a different MRI cohort')
        progress_figure(previous['results'], values, args.figure)
    for row in values:
        print(row, flush=True)


if __name__ == '__main__':
    main()
