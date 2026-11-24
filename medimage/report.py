from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from medimage.common import read_json


def _interval(values: list[float]) -> str:
    return f'{values[0]:.3f}–{values[1]:.3f}'


def make_figures(chest: dict, mammo: dict, spine: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 6.1), layout='constrained', width_ratios=[1.55, 1.1, 1])
    fig.suptitle('MedImage ML Lab', fontsize=19, weight='bold', x=.02, ha='left')
    neutral = '#395267'
    accent = '#007f77'
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='x', alpha=.18)
        axis.set_axisbelow(True)
    labels = list(chest['test_calibrated']['labels'].items())
    labels.sort(key=lambda entry: entry[1]['auprc'])
    y = np.arange(len(labels))
    axes[0].set_yticks(y, [name.replace('_', ' ') for name, _ in labels], fontsize=9)
    axes[0].scatter([value['auprc'] for _, value in labels], y, color=accent, label='AP', zorder=3)
    axes[0].scatter([value['prevalence'] for _, value in labels], y, color=neutral, marker='|', s=95,
                    label='Prevalence', zorder=2)
    axes[0].set(title='Chest X-ray · 14 findings', xlabel='Average precision', xlim=(0, 1))
    axes[0].legend(loc='lower right', frameon=False)
    groups = [('All breasts', mammo['test_calibrated'])]
    for key, title in (('both_views', 'Both views'), ('single_view', 'Single view'),
                       ('mass_only', 'Mass'), ('calcification_only', 'Calcification')):
        subset = mammo['test_subgroups'].get(key)
        if subset and subset['macro_auroc'] is not None:
            groups.append((title, subset))
    for index, (label, data) in enumerate(groups, 1):
        axes[1].scatter(data['macro_auroc'], index, color=accent if index == 1 else neutral, zorder=3)
    axes[1].set_yticks(range(1, len(groups) + 1), [f'{item[0]} ({item[1]["rows"]})' for item in groups], fontsize=9)
    low, high = mammo['patient_bootstrap']['macro_auroc_ci95']
    axes[1].hlines(1, low, high, color=accent, linewidth=2)
    axes[1].axvline(.5, color='gray', linestyle=':')
    axes[1].set(title='Mammography · test cohort', xlabel='AUROC', xlim=(.35, 1), ylim=(len(groups) + .65, .35))
    items = [(name, record['overall']['score']) for name, record in spine['assessments'].items()]
    subtitles = {'prior': 'Class prior', 'localized': 'Localized 2.5D', 'global_reference': 'Global 2.5D',
                 'equal_weight_blend': '50/50 blend'}
    indices = range(len(items))
    axes[2].barh(indices, [item[1] for item in items],
                 color=[accent if item[0] == spine['selected_for_local_inference'] else '#a1b5be' for item in items])
    axes[2].set_yticks(indices, [subtitles[name] for name, _ in items], fontsize=9)
    for index, (_name, value) in enumerate(items):
        axes[2].text(value + .015, index, f'{value:.3f}', va='center', fontsize=9)
    axes[2].invert_yaxis()
    axes[2].set(title='Lumbar MRI · five-fold OOF', xlabel='Weighted log loss (lower is better)', xlim=(0, max(v for _, v in items) * 1.16))
    for axis in axes:
        axis.title.set_ha('left')
        axis.title.set_position((0, 1.05))
    fig.savefig(output / 'results.png', dpi=160, facecolor='white', metadata={'Title': 'Aggregated MedImage ML Lab results'})
    plt.close(fig)


def report(artifacts: Path, output: Path) -> None:
    chest = read_json(artifacts / 'evaluation' / 'chest' / 'metrics.json')
    mammo = read_json(artifacts / 'evaluation' / 'mammo' / 'metrics.json')
    spine = read_json(artifacts / 'evaluation' / 'spine' / 'metrics.json')
    spine_inference = read_json(artifacts / 'inference' / 'spine.json')
    output.mkdir(parents=True, exist_ok=True)
    make_figures(chest, mammo, spine, output)
    lines = ['# Initial research benchmarks', '', '![Aggregate results for chest X-ray, mammography, and lumbar MRI](results.png)', '',
             'Each panel uses a task-specific dataset and metric.', '',
             '## NIH ChestX-ray14 and CBIS-DDSM', '',
             '| Benchmark | Selected model | Samples | AUROC (95% CI) | Average precision (95% CI) | Brier before → after |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for data, name in ((chest, 'NIH ChestX-ray14'), (mammo, 'CBIS-DDSM')):
        raw, result, ci = data['test_raw'], data['test_calibrated'], data['patient_bootstrap']
        lines.append(f'| {name} | `{data["selection"]["winner"]}` | {result["rows"]:,} | '
                     f'{result["macro_auroc"]:.3f} ({_interval(ci["macro_auroc_ci95"])}) | '
                     f'{result["macro_auprc"]:.3f} ({_interval(ci["macro_auprc_ci95"])}) | '
                     f'{raw["macro_brier"]:.4f} → {result["macro_brier"]:.4f} |')
    lines.extend(['', 'Chest X-ray uses macro averages over 14 findings. Mammography uses one malignancy endpoint. '
                  'Intervals use 400 resamples of entire patient groups. Chest X-ray groups merge IDs linked by identical images.', '',
                  'The Brier score measures mean squared probability error. '
                  'The before-and-after values show how validation-fitted calibration transfers to each benchmark.', '',
                  '### Validation runs', '',
                  '| Run | Selected epoch | AUROC | Average precision | Training epochs | Minutes | Peak reserved VRAM |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for data in (chest, mammo):
        for variant in data['selection']['variants']:
            path = artifacts / 'runs' / variant['run']
            history = read_json(path / 'history.json')['history']
            val = variant['best_validation']
            peak = max(item['peak_reserved_mib'] for item in history)
            minutes = sum(item['seconds'] for item in history) / 60.
            lines.append(f'| `{variant["run"]}` | {val["epoch"]} | {val["macro_auroc"]:.3f} | '
                         f'{val["macro_auprc"]:.3f} | {len(history)} | {minutes:.1f} | {peak / 1024:.2f} GiB |')
    lines.extend(['', '### Chest X-ray findings', '',
                  '| Finding | Positive test images | AUROC | Average precision |',
                  '| --- | ---: | ---: | ---: |'])
    for name, values in chest['test_calibrated']['labels'].items():
        lines.append(f'| {name.replace("_", " ")} | {values["positives"]:,} | {values["auroc"]:.3f} | {values["auprc"]:.3f} |')
    lines.extend(['', '## RSNA LumbarDISC', '',
                  '| Modell | Overall OOF | Spinal | Foraminal | Subarticular | Any severe spinal |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for name, record in spine['assessments'].items():
        keys = ('score', 'spinal', 'foraminal', 'subarticular', 'any_severe_spinal')
        lines.append(f'| `{name}` | ' + ' | '.join(f'{record["overall"][key]:.5f}' for key in keys) + ' |')
    lines.extend(['', '### Fold results', '',
                  '| Modell | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for name, record in spine['assessments'].items():
        lines.append(f'| `{name}` | ' + ' | '.join(f'{fold["score"]:.5f}' for fold in record['per_fold']) + ' |')
    comparison = spine['assessments'].get('equal_weight_blend')
    if comparison:
        lines.extend(['', '95% paired bootstrap intervals for loss differences against the global reference.', ''])
        for name in ('localized', 'equal_weight_blend'):
            record = spine['assessments'][name]
            lines.append(f'- `{name}` {_interval(record["paired_delta_ci95_vs_global"])} and {record["folds_beating_global"]} of 5 folds improved.')
    local = spine['localization']
    fold0 = spine['fold_experiments'][0]
    oracle = fold0['oracle_coordinates_sensitivity_only']['score']
    regular = fold0['validation']['score']
    lines.extend(['', f'The selected inference configuration is `{spine["selected_for_local_inference"]}`. '
                  f'Raw-DICOM-to-CSV inference was verified on {spine_inference["study_count"]} study with {spine_inference["rows"]} rows.', '',
                  f'Median landmark error was {local["median_anchor_error_mm"]:.2f} mm, versus {local["atlas_only_median_anchor_error_mm"]:.2f} mm for the static reference. '
                  f'{local["within_10mm"]:.1%} of landmarks were within 10 mm. '
                  f'{local["missing_primary_roi_fraction"]:.2%} of primary regions had a missing view.', '',
                  f'On fold 0 the loss was {regular:.5f} with estimated coordinates and {oracle:.5f} with available annotated coordinates. '
                  f'Exploratory calibration on saved OOF predictions yielded {spine["exploratory_cross_fold_calibration"]["score"]["score"]:.5f}.', ''])
    (output / 'results.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate aggregate-only figures and result tables')
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts'))
    parser.add_argument('--output', type=Path, default=Path('docs'))
    args = parser.parse_args()
    report(args.artifacts, args.output)


if __name__ == '__main__':
    main()
