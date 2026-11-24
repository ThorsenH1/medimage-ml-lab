from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from medimage.common import checksum, read_json
from medimage.portfolio_figures import BLUE, OLD, TEAL, _macro_curve, plt, style, write_figure


def draw(task, truth, previous, current, report, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.7), layout='constrained')
    title = 'NIH ChestX-ray14' if task == 'chest' else 'CBIS-DDSM'
    style(fig, axes, f'{title} | locked official test evaluation')
    grid = np.linspace(0, 1, 161)
    test = report['test']
    reference = report['reference_test']
    for scores, label, color, performance in ((previous, 'Previous', OLD, reference),
                                               (current, 'Selected', TEAL, test)):
        roc = _macro_curve(truth, scores, grid, precision_recall=False)
        pr = _macro_curve(truth, scores, grid, precision_recall=True)
        axes[0, 0].plot(grid, roc, color=color, lw=2, label=f"{label} {performance['macro_auroc']:.3f}")
        axes[0, 1].plot(grid, pr, color=color, lw=2, label=f"{label} {performance['macro_auprc']:.3f}")
    axes[0, 0].plot(grid, grid, '--', color=OLD, alpha=.7)
    axes[0, 1].axhline(truth.mean(), color=OLD, linestyle='--', alpha=.7, label='Mean prevalence')
    axes[0, 0].set(title='Receiver operating characteristic', xlabel='False-positive rate', ylabel='True-positive rate')
    axes[0, 1].set(title='Precision and recall', xlabel='Recall', ylabel='Precision')
    for axis in axes[0]:
        axis.set(xlim=(0, 1), ylim=(0, 1.02))
        axis.legend(frameon=False, fontsize=9)
    target = axes[1, 0]
    if task == 'chest':
        names = sorted(test['labels'], key=lambda label: test['labels'][label]['auroc'])
        ordering = np.arange(len(names))
        target.scatter([test['labels'][name]['auroc'] for name in names], ordering, color=TEAL, s=21)
        target.scatter([reference['labels'][name]['auroc'] for name in names], ordering, color=OLD, s=19)
        target.set(yticks=ordering, yticklabels=[name.replace('_', ' ') for name in names], title='AUROC by finding')
        target.tick_params(axis='y', labelsize=8)
    else:
        subsets = [(name, value) for name, value in report['test_subsets'].items() if value['macro_auroc'] is not None]
        target.scatter([data['macro_auroc'] for _, data in subsets], np.arange(len(subsets)), color=TEAL)
        target.set(yticks=np.arange(len(subsets)),
                   yticklabels=[f"{name.replace('_', ' ').title()} · {val['rows']} breasts" for name, val in subsets],
                   title='AUROC by reference lesion type')
    target.axvline(.8, color=BLUE, linestyle=':', alpha=.6)
    target.set(xlim=(.45, 1), xlabel='AUROC')
    interval = report['test_uncertainty']['paired_auroc_delta_ci95']
    axes[1, 1].axvline(0, color=OLD, alpha=.7)
    axes[1, 1].hlines(0, *interval, color=TEAL, lw=2.5)
    axes[1, 1].scatter(report['auroc_delta'], 0, color=TEAL, s=45, zorder=3)
    axes[1, 1].set(title='Paired change with 95% patient-bootstrap interval',
                   yticks=[], ylim=(-.9, .9), xlabel='Selected − previous AUROC')
    write_figure(fig, path)


def render(output: Path, previous: Path):
    report = read_json(output / 'metrics.json')
    lock = read_json(output / 'selection.json')
    if report['selection_sha256'] != checksum(output / 'selection.json'):
        raise ValueError('Test report differs from the frozen decision')
    with (np.load(output / 'test-predictions.npz', allow_pickle=False) as selected,
          np.load(previous / 'evaluation' / report['task'] / 'test-predictions.npz', allow_pickle=False) as incumbent):
        if not np.array_equal(selected['rows'], incumbent['rows']) or not np.array_equal(
            selected['labels'], incumbent['labels']
        ):
            raise ValueError('Cannot compare two different test populations')
        draw(report['task'], selected['labels'], incumbent['probabilities'], selected['probabilities'],
             report, output / 'benchmark.png')
    current, reference, interim = report['test'], report['reference_test'], report['test_uncertainty']
    lo, hi = interim['macro_auroc_ci95']
    low, high = interim['paired_auroc_delta_ci95']
    lines = ['# Official test benchmark', '',
             f"Selected macro AUROC is **{current['macro_auroc']:.4f}** on {current['rows']:,} test samples "
             f"from {interim['patients']:,} patients. The 95% interval is {lo:.4f}–{hi:.4f}.",
             '', '![Benchmark plots](benchmark.png)', '',
             '| Model | AUROC | Average precision | Brier score |', '| --- | ---: | ---: | ---: |']
    for name, metrics in (('Previous locked model', reference), ('Selected model', current)):
        lines.append(f"| {name} | {metrics['macro_auroc']:.4f} | {metrics['macro_auprc']:.4f} | "
                     f"{metrics['macro_brier']:.4f} |")
    lines += ['', f"The paired change in AUROC is {report['auroc_delta']:+.4f}, with a 95% interval "
              f"of {low:+.4f} to {high:+.4f}. The procedure selected a model and thresholds on validation "
              "before scoring the previously used official benchmark.", '', '## Validation ranking', '',
              '| Rank | Components | Combination | AUROC | Average precision |',
              '| ---: | --- | --- | ---: | ---: |']
    for rank, recipe in enumerate(lock['validation_ranking'][:12], start=1):
        components = ', '.join(f'{key} {ratio:.2f}' for key, ratio in recipe['weights'].items())
        metrics = recipe['metrics']
        lines.append(f"| {rank} | {components} | {recipe['method']} | {metrics['macro_auroc']:.4f} | "
                     f"{metrics['macro_auprc']:.4f} |")
    lines += ['', '## Local held-out examples', '',
              'A fixed seed selects examples for the four reference and prediction outcomes. '
              'The table shows probabilities, validation thresholds, and held-out reference labels.', '',
              '| Example | Endpoint | View | Probability | Threshold | Decision | Reference |',
              '| ---: | --- | --- | ---: | ---: | ---: | ---: |']
    for row in report['examples']:
        lines.append(f"| {row['example']} | {row['target'].replace('_', ' ')} | {row['projection']} | "
                     f"{row['probability']:.3f} | {row['threshold']:.3f} | {row['decision']} | {row['reference']} |")
    lines += ['', 'The [metrics file](metrics.json) includes per-target scores, subgroup comparisons, '
              'bootstrap intervals and selected model components.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Render the local locked radiography benchmark')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    args = parser.parse_args()
    render(args.output, args.previous)


if __name__ == '__main__':
    main()
