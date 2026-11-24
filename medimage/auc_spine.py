from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from medimage.common import checksum, read_json, write_json
from medimage.spine_data import CONDITIONS, LEVELS, SEVERITIES, TARGETS
from medimage.spine_metrics import read_submission, read_targets, score


def auc_cells(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    auc, ap, counts = [], [], []
    for target, grades in zip(labels.T, probabilities.transpose(1, 0, 2), strict=True):
        known = target >= 0
        y = target[known, None] == np.arange(3)
        p = grades[known]
        counts.append(y.sum(axis=0))
        auc.append([roc_auc_score(truth, value) if 0 < truth.sum() < len(truth) else np.nan
                    for truth, value in zip(y.T, p.T, strict=True)])
        ap.append([average_precision_score(truth, value) if 0 < truth.sum() < len(truth) else np.nan
                   for truth, value in zip(y.T, p.T, strict=True)])
    return np.asarray(auc), np.asarray(ap), np.asarray(counts)


def interval(labels, proba, groups, repeats):
    statistics = []
    for index in range(labels.shape[1]):
        observed = np.flatnonzero(labels[:, index] >= 0)
        for grade in range(3):
            p = proba[observed, index, grade]
            order = np.argsort(-p, kind='stable')
            sorted_rows = observed[order]
            p = p[order]
            end = np.r_[np.flatnonzero(p[1:] != p[:-1]), len(p) - 1]
            statistics.append((sorted_rows, labels[sorted_rows, index] == grade, end))
    rng = np.random.default_rng(55318)
    folds = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    samples = []
    for _ in range(repeats):
        sample = np.concatenate([rng.choice(fold, len(fold), replace=True) for fold in folds])
        weight = np.bincount(sample, minlength=len(proba))
        estimates = []
        for rows, y, end in statistics:
            total = weight[rows]
            tp = np.cumsum(total * y)[end]
            fp = np.cumsum(total)[end] - tp
            if not tp[-1] or not fp[-1]:
                break
            estimates.append(np.trapezoid(np.r_[0, tp / tp[-1]], np.r_[0, fp / fp[-1]]))
        if len(estimates) == len(statistics):
            samples.append(np.mean(estimates))
    if len(samples) < repeats * .8:
        raise ValueError('Too many bootstrap replicates lost a severity class')
    return {'study_bootstrap_ci95': np.percentile(samples, [2.5, 97.5]).tolist(),
            'bootstrap_repeats': repeats, 'valid_bootstrap_replicates': len(samples)}


def figure(auc: np.ndarray, output: Path):
    from medimage.portfolio_figures import plt, write_figure

    fig, axes = plt.subplots(1, 3, figsize=(12.1, 4.05), layout='constrained', sharey=True)
    fig.suptitle('Lumbar MRI discrimination by severity and level', fontsize=17, weight='bold', x=.03, ha='left')
    families = ('Spinal canal', 'Left foraminal', 'Right foraminal', 'Left subarticular', 'Right subarticular')
    for index, axis in enumerate(axes):
        grid = auc[:, index].reshape(5, 5)
        glyph = axis.imshow(grid, vmin=.60, vmax=1, cmap='viridis')
        for y, x in np.ndindex(grid.shape):
            value = grid[y, x]
            axis.text(x, y, f'{value:.2f}', ha='center', va='center', color='white' if value < .83 else '#102520',
                      fontsize=9)
        axis.set_title(SEVERITIES[index].replace('_', ' / ').title(), fontsize=11, weight='bold')
        axis.set_xticks(np.arange(5), [val.replace('_', '–').upper() for val in LEVELS], rotation=35, ha='right')
        axis.set_yticks(np.arange(5), families)
        axis.tick_params(length=0)
    fig.colorbar(glyph, ax=axes, shrink=.72, label='One-vs-rest AUROC')
    write_figure(fig, output)


def evaluate(args):
    plan = read_json(args.selection)
    folds = pd.read_csv(args.folds)
    if checksum(args.folds) != plan['fold_sha256']:
        raise ValueError('MRI fold assignments differ from the selection provenance')
    studies = folds.study_id.to_numpy()
    validation = folds.fold.to_numpy()
    if folds.study_id.duplicated().any() or set(validation) != set(range(5)):
        raise ValueError('Expected five unique study-grouped partitions')
    labels = read_targets(args.root, studies)
    probabilities = read_submission(args.predictions, studies)
    if not np.isclose(score(labels, probabilities)['score'], plan['selected']['overall']['score'], atol=1e-9):
        raise ValueError('OOF probabilities do not reproduce the locked original score')
    auc, ap, counts = auc_cells(labels, probabilities)
    if not np.isfinite(auc).all():
        raise ValueError('Pooled MRI AUROC must cover all 75 target-grade cells')
    by_fold = []
    for fold in range(5):
        take = validation == fold
        value, _, _ = auc_cells(labels[take], probabilities[take])
        by_fold.append({'fold': fold, 'studies': int(take.sum()), 'macro_auroc': float(np.nanmean(value)),
                        'evaluable_cells': int(np.isfinite(value).sum())})
    cells = [{'target': target, 'severity': grade, 'positives': int(counts[i, j]),
              'negatives': int(counts[i].sum() - counts[i, j]), 'auroc': float(auc[i, j]), 'auprc': float(ap[i, j])}
             for i, target in enumerate(TARGETS) for j, grade in enumerate(SEVERITIES)]
    rng = np.random.default_rng(1028)
    order = rng.permutation(len(studies))
    examples = []
    unused = np.ones(len(studies), dtype=bool)
    for nominal in range(3):
        for correct in (True, False):
            valid = unused[:, None] & (labels[order] == nominal) & (
                (probabilities[order].argmax(2) == nominal) == correct)
            i, j = np.argwhere(valid)[0]
            unused[i] = False
            row = order[i]
            examples.append({'example': len(examples) + 1, 'fold': int(validation[row]), 'target': TARGETS[j],
                             'reference_grade': SEVERITIES[nominal],
                             'top_predicted_grade': SEVERITIES[int(probabilities[row, j].argmax())],
                             'probabilities': dict(zip(SEVERITIES, probabilities[row, j].tolist(), strict=True))})
    report = {'protocol': 'Frozen study-grouped five-fold, out-of-fold, pooled macro one-vs-rest',
              'model': plan['selected']['name'], 'selection_sha256': checksum(args.selection),
              'predictions_sha256': checksum(args.predictions), 'fold_sha256': plan['fold_sha256'],
              'studies': len(studies), 'observed_grades': int((labels >= 0).sum()),
              'missing_grades': int((labels < 0).sum()), 'evaluable_cells': len(cells),
              'macro_auroc': float(auc.mean()), 'macro_auprc': float(ap.mean()),
              'ordinal_auroc': {'moderate_or_severe': float(auc[:, 0].mean()), 'severe': float(auc[:, 2].mean())},
              'weighted_log_loss': score(labels, probabilities),
              'condition_auroc': dict(zip(CONDITIONS, auc.reshape(5, 5, 3).mean((1, 2)).tolist(), strict=True)),
              'severity_auroc': dict(zip(SEVERITIES, auc.mean(0).tolist(), strict=True)),
              'levels_auroc': dict(zip(LEVELS, auc.reshape(5, 5, 3).mean((0, 2)).tolist(), strict=True)),
              'folds': by_fold, 'cells': cells, 'examples': examples,
              **interval(labels, probabilities, validation, args.bootstrap)}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'spine-auroc.json', report)
    figure(auc, args.output / 'spine-auroc.png')
    print({key: report[key] for key in ('macro_auroc', 'macro_auprc', 'study_bootstrap_ci95', 'evaluable_cells')},
          flush=True)


def main():
    parser = argparse.ArgumentParser(description='Measure discrimination on frozen out-of-fold lumbar predictions')
    for name in ('root', 'folds', 'selection', 'predictions', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
