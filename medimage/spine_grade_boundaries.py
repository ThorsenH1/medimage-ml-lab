"""Compare frozen grade boundaries and test fixed two-step severity pools on the same folds."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from medimage.auc_spine import interval
from medimage.common import write_json
from medimage.spine_geometry_probe import classification_summary
from medimage.spine_metrics import submission
from medimage.spine_ordinal_audit import summarize
from medimage.spine_pool_audit import BOUNDARY_FRACTIONS, factored_pool, load_pool_inputs, probability_pools


def figure(report, destination):
    from medimage.portfolio_figures import BLUE, TEAL, plt, write_figure

    conditions = ['Central canal', 'Left foramen', 'Right foramen', 'Left subarticular', 'Right subarticular']
    positions = np.arange(len(conditions))
    fig, axes = plt.subplots(1, 2, figsize=(11.1, 4.4), sharey=True, layout='constrained')
    fig.suptitle('Which grade boundary is harder for each MRI branch?', x=.025, ha='left', fontsize=16, fontweight='bold')
    sources = (('original_grid', 'Original grid', BLUE, 'o'), ('reflected_native', 'Higher resolution', TEAL, 'D'),
               ('geometric', 'Geometric blend', '#8e5719', '^'))
    for axis, grade, title in zip(axes, ('normal_mild', 'severe'),
                                  ('Moderate vs. normal or mild', 'Moderate vs. severe'), strict=True):
        for shift, (key, label, color, marker) in zip((-.19, 0, .19), sources, strict=True):
            results = report['boundaries_by_condition'][key]
            values = [row['moderate_vs'][grade]['conditional_score_auroc'] for row in results]
            axis.scatter(values, positions + shift, s=38, label=label, marker=marker, color=color)
        axis.set_title(title, loc='left', fontsize=11)
        axis.set_xlabel('AUROC, average over five levels')
        axis.set_xlim(.72, .96)
        axis.grid(axis='x', alpha=.25)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(positions, conditions)
    axes[0].invert_yaxis()
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.20), ncol=3, frameon=False, fontsize=8)
    write_figure(fig, destination)


def run(args):
    studies, folds, truth, previous, native, arithmetic, provenance = load_pool_inputs(args)
    geometric = probability_pools(previous, native)['geometric']
    controls = {'original_grid': previous, 'reflected_native': native, 'arithmetic': arithmetic, 'geometric': geometric}
    report = {**provenance, 'controls': {key: classification_summary(truth, item) for key, item in controls.items()},
              'boundaries_by_condition': {key: summarize(truth, item) for key, item in controls.items()},
              'pools': {}, 'original_grid_fractions': dict(BOUNDARY_FRACTIONS),
              'comparison': 'Fixed probability factorizations compared with the existing geometric pool'}
    for name in BOUNDARY_FRACTIONS:
        predictions = factored_pool(previous, native, name)
        result = classification_summary(truth, predictions)
        result.update(interval(truth, predictions, folds, args.bootstrap, reference=geometric))
        result['paired_auroc_delta'] = result['macro_auroc'] - report['controls']['geometric']['macro_auroc']
        report['pools'][name] = result
        print(name, result, flush=True)
        submission(studies, predictions).to_csv(args.output / f'{name}-oof.csv', index=False)
    figure(report, args.output / 'grade-boundaries.png')
    write_json(args.output / 'metrics.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'folds', 'original', 'native-comparison', 'reflection', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('Use a positive bootstrap count')
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == '__main__':
    main()
