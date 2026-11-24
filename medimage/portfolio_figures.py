from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import precision_recall_curve, roc_curve

BLUE = '#214f75'
TEAL = '#007c74'
OLD = '#778e9f'


def style(fig, axes, title: str) -> None:
    fig.suptitle(title, x=.03, ha='left', fontsize=17, weight='bold')
    for axis in np.ravel(axes):
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(alpha=.18)
        axis.set_axisbelow(True)


def write_figure(fig, output: Path) -> None:
    fig.savefig(output, dpi=160, facecolor='white', metadata={'Title': 'Aggregate benchmark metrics'})
    plt.close(fig)


def _macro_curve(labels: np.ndarray, scores: np.ndarray, grid: np.ndarray, *, precision_recall: bool) -> np.ndarray:
    curves = []
    for y, p in zip(labels.T, scores.T, strict=True):
        if precision_recall:
            precision, recall, _ = precision_recall_curve(y, p)
            curves.append(np.interp(grid, recall[::-1], precision[::-1]))
        else:
            false_positive, true_positive, _ = roc_curve(y, p)
            curves.append(np.interp(grid, false_positive, true_positive))
    return np.mean(curves, axis=0)


def _calibration(labels, proba, n=10):
    bins = np.minimum((proba.ravel() * n).astype(int), n - 1)
    counts = np.bincount(bins, minlength=n)
    usable = counts >= 8
    mean_x = np.bincount(bins, weights=proba.ravel(), minlength=n)[usable] / counts[usable]
    mean_y = np.bincount(bins, weights=labels.ravel(), minlength=n)[usable] / counts[usable]
    return mean_x, mean_y


def radiography(task: str, metrics: dict, truth: np.ndarray, previous: np.ndarray,
               selected: np.ndarray, output: Path) -> None:
    if task == 'chest':
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.9), layout='constrained', width_ratios=[1.5, 1])
        style(fig, axes, 'NIH ChestX-ray14 | repeat benchmark')
        names = sorted(metrics['test']['labels'], key=lambda key: metrics['test']['labels'][key]['auprc'])
        positions = np.arange(len(names))
        for name, label, color in (('previous', 'Initial model', OLD), ('test', 'Selected model', TEAL)):
            axes[0].scatter([metrics[name]['labels'][key]['auprc'] for key in names], positions, label=label, color=color)
        axes[0].scatter([metrics['test']['labels'][key]['prevalence'] for key in names], positions,
                        label='Prevalence', color=BLUE, marker='|', s=70)
        axes[0].set(yticks=positions, yticklabels=[key.replace('_', ' ') for key in names],
                    title='Average precision by finding', xlabel='Average precision', xlim=(0, 1))
        axes[0].legend(loc='lower right', frameon=False, fontsize=8)
        grid = np.linspace(0, 1, 101)
        for scores, name, color in ((previous, 'Initial model', OLD), (selected, 'Selected model', TEAL)):
            axes[1].plot(grid, _macro_curve(truth, scores, grid, precision_recall=False), label=name, color=color, linewidth=2)
        axes[1].plot(grid, grid, color='#a5a5a5', linestyle=':', label='Chance')
        axes[1].set(title='Macro ROC', xlabel='False-positive rate', ylabel='True-positive rate', xlim=(0, 1), ylim=(0, 1.02))
        axes[1].legend(loc='lower right', frameon=False, fontsize=8)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), layout='constrained')
        style(fig, axes, 'CBIS-DDSM | multi-view malignancy benchmark')
        grid = np.linspace(0, 1, 101)
        for scores, name, color in ((previous, 'Initial model', OLD), (selected, 'Selected model', TEAL)):
            axes[0, 0].plot(grid, _macro_curve(truth, scores, grid, precision_recall=False), label=name, color=color, linewidth=2)
            axes[0, 1].plot(grid, _macro_curve(truth, scores, grid, precision_recall=True), label=name, color=color, linewidth=2)
            x, y = _calibration(truth, scores)
            axes[1, 0].plot(x, y, 'o-', label=name, color=color)
        axes[0, 0].plot(grid, grid, color='#a5a5a5', linestyle=':')
        axes[0, 0].set(title='ROC', xlabel='False-positive rate', ylabel='True-positive rate', xlim=(0, 1), ylim=(0, 1.03))
        axes[0, 1].axhline(float(truth.mean()), color='#a5a5a5', linestyle=':', label='Prevalence')
        axes[0, 1].set(title='Precision–recall', xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.03))
        axes[1, 0].plot(grid, grid, color='#a5a5a5', linestyle=':')
        axes[1, 0].set(title='Reliability', xlabel='Mean predicted probability', ylabel='Observed frequency', xlim=(0, 1), ylim=(0, 1.03))
        outcomes = [('Overall', metrics['test'])]
        for key, label in (('both_views', 'Both views'), ('single_view', 'Single view'), ('mass_only', 'Mass'), ('calcification_only', 'Calcification')):
            if key in metrics['subgroups'] and metrics['subgroups'][key]['macro_auroc'] is not None:
                outcomes.append((label, metrics['subgroups'][key]))
        i = np.arange(len(outcomes))
        axes[1, 1].set(yticks=i, yticklabels=[f'{name} ({data["rows"]})' for name, data in outcomes],
                       title='Selected model AUROC by group', xlabel='AUROC', xlim=(.4, 1))
        axes[1, 1].scatter([data['macro_auroc'] for _, data in outcomes], i, c=[TEAL] + [BLUE] * (len(i) - 1))
        low, high = metrics['patient_bootstrap']['macro_auroc_ci95']
        axes[1, 1].hlines(0, low, high, color=TEAL, linewidth=2)
        axes[1, 1].invert_yaxis()
        axes[0, 0].legend(loc='lower right', frameon=False, fontsize=8)
        axes[0, 1].legend(loc='upper right', frameon=False, fontsize=8)
    write_figure(fig, output)


def spine(plan: dict, initial: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.1, 4.25), layout='constrained')
    style(fig, axes, 'RSNA LumbarDISC | paired five-fold evaluation')
    selected = plan['selected']
    reference = next(item for item in plan['candidates'] if item['name'] == 'incumbent')
    folds = np.arange(5)
    axes[0].plot(folds, reference['per_fold'], 'o-', color=OLD, label='Initial blend')
    axes[0].plot(folds, selected['per_fold'], 'o-', color=TEAL, label='Selected blend')
    axes[0].set(title='Weighted loss by outer fold', xticks=folds, xlabel='Outer fold', ylabel='Weighted log loss')
    axes[0].legend(frameon=False, fontsize=8)
    keys = ('spinal', 'foraminal', 'subarticular', 'any_severe_spinal')
    values = [selected['overall'][key] - reference['overall'][key] for key in keys]
    axes[1].barh(np.arange(4), values, color=TEAL)
    axes[1].set(yticks=range(4), yticklabels=['Spinal', 'Foraminal', 'Subarticular', 'Any severe spinal'],
                title='Change in component loss', xlabel='Selected minus initial')
    axes[1].axvline(0, color=OLD, linewidth=1)
    axes[1].xaxis.set_major_locator(MaxNLocator(4))
    axes[1].invert_yaxis()
    loc = initial['localization']
    levels = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
    axes[2].barh(range(5), loc['per_level_median_mm'], color=BLUE)
    axes[2].axvline(loc['atlas_only_median_anchor_error_mm'], color=OLD, linestyle='--', label='Static atlas median')
    axes[2].set(yticks=range(5), yticklabels=levels, title='Landmark error by level', xlabel='Median error (mm)')
    axes[2].invert_yaxis()
    axes[2].legend(loc='lower right', frameon=False, fontsize=8)
    write_figure(fig, output)


def search_rungs(groups: dict[str, list[dict]], budgets: list[int], title: str, output: Path, *, minimize: bool) -> None:
    count = len(groups)
    columns = 3 if count == 5 else 2
    fig, axes = plt.subplots(2, columns, figsize=(4.2 * columns, 5.7), layout='constrained', squeeze=False,
                             sharex=True, sharey=True)
    style(fig, axes, title)
    for axis, (name, rows) in zip(axes.ravel(), groups.items()):
        best_by_stage = []
        for rung in range(1, len(budgets) + 1):
            scores = sorted(row['score'] for row in rows if row['rung'] == rung)
            best = scores[0] if minimize else scores[-1]
            best_by_stage.append(best)
            jitter = np.linspace(-.07, .07, len(scores))
            axis.scatter(rung + jitter, scores, color=OLD, alpha=.68, s=18)
        axis.plot(range(1, len(budgets) + 1), best_by_stage, color=TEAL, marker='o')
        axis.set(title=name, xticks=range(1, len(budgets) + 1), xticklabels=[f'{value} ep' for value in budgets])
        axis.tick_params(labelbottom=True)
    for axis in axes.ravel()[count:]:
        axis.set_visible(False)
    fig.supylabel('Inner weighted log loss' if minimize else 'Inner validation objective', fontsize=10)
    write_figure(fig, output)
