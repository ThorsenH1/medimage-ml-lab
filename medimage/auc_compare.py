"""Compare two frozen radiography models on their identical official test cohort."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from medimage.auc_selection import read_scores
from medimage.common import checksum, load_manifest, read_json, write_json
from medimage.portfolio_figures import BLUE, TEAL, plt, write_figure
from medimage.radiography_evaluate import patient_bootstrap


def compare(args: argparse.Namespace) -> None:
    frame, cohort = load_manifest(args.cohort)
    older, newer = [read_json(folder / 'selection.json') for folder in (args.reference, args.candidate)]
    keys = ('task', 'manifest', 'labels')
    if any(older[key] != newer[key] for key in keys) or newer['manifest'] != cohort['manifest_sha256']:
        raise ValueError('Both models must cover the same task, cohort and label order')
    labels = newer['labels']
    rows = np.flatnonzero(frame.split.eq('test').to_numpy())
    truth_keys = labels if newer['task'] == 'chest' else ['positive']
    truth = frame[truth_keys].to_numpy(dtype=np.int8)[rows]
    probabilities, metrics, digests = [], [], []
    for folder in (args.reference, args.candidate):
        report = read_json(folder / 'metrics.json')
        digest = checksum(folder / 'selection.json')
        if report['selection_sha256'] != digest or report['test']['rows'] != len(rows):
            raise ValueError('Each official test report must match its corresponding selection and cohort')
        probabilities.append(read_scores(folder / 'test-predictions.npz', rows, num_labels=len(labels),
                                         truth=truth, key='probabilities'))
        metrics.append(report['test'])
        digests.append(digest)
    paired = patient_bootstrap(truth, probabilities[1], frame.iloc[rows].patient.to_numpy(),
                               repeats=args.bootstrap, reference=probabilities[0])
    differences = {f'{key}_delta': metrics[1][key] - metrics[0][key]
                   for key in ('macro_auroc', 'macro_auprc', 'macro_brier')}
    by_finding = {label: {'reference': metrics[0]['labels'][label]['auroc'],
                          'candidate': metrics[1]['labels'][label]['auroc']} for label in labels}
    report = {'reference_sha256': digests[0], 'candidate_sha256': digests[1], 'test_rows': len(rows),
              'bootstrap_repeats': args.bootstrap, **paired, **differences, 'findings': by_finding}
    write_json(args.output, report)
    if args.figure:
        fig, axis = plt.subplots(figsize=(8.6, 5.4), layout='constrained')
        y = np.arange(len(labels))
        reference = [values['reference'] for values in by_finding.values()]
        candidate = [values['candidate'] for values in by_finding.values()]
        axis.hlines(y, np.minimum(reference, candidate), np.maximum(reference, candidate), color='#7a9da1')
        axis.scatter(reference, y, color=BLUE, s=36, label=args.reference_name)
        axis.scatter(candidate, y, color=TEAL, s=42, marker='D', label=args.candidate_name)
        axis.set(yticks=y, yticklabels=[label.replace('_', ' ') for label in labels], xlabel='AUROC')
        axis.set_title('Finding-level AUROC on the official test', loc='left', fontsize=15, fontweight='bold')
        axis.invert_yaxis()
        axis.legend(frameon=False, loc='lower left')
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='x', alpha=.22)
        write_figure(fig, args.figure)
    print({**differences, **paired}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort', 'reference', 'candidate', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--reference-name', default='Reference')
    parser.add_argument('--candidate-name', default='Candidate')
    parser.add_argument('--figure', type=Path)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('Use a positive bootstrap count')
    compare(args)


if __name__ == '__main__':
    main()
