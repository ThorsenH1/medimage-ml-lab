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
                    label='Prevalens', zorder=2)
    axes[0].set(title='Thorax · 14 etiketter', xlabel='Gjennomsnittlig presisjon', xlim=(0, 1))
    axes[0].legend(loc='lower right', frameon=False)
    groups = [('Alle bryster', mammo['test_calibrated'])]
    for key, title in (('both_views', 'Begge projeksjoner'), ('single_view', 'Én projeksjon'),
                       ('mass_only', 'Masse'), ('calcification_only', 'Forkalkning')):
        subset = mammo['test_subgroups'].get(key)
        if subset and subset['macro_auroc'] is not None:
            groups.append((title, subset))
    for index, (label, data) in enumerate(groups, 1):
        axes[1].scatter(data['macro_auroc'], index, color=accent if index == 1 else neutral, zorder=3)
    axes[1].set_yticks(range(1, len(groups) + 1), [f'{item[0]} ({item[1]["rows"]})' for item in groups], fontsize=9)
    low, high = mammo['patient_bootstrap']['macro_auroc_ci95']
    axes[1].hlines(1, low, high, color=accent, linewidth=2)
    axes[1].axvline(.5, color='gray', linestyle=':')
    axes[1].set(title='Mammografi · holdt ute test', xlabel='AUROC', xlim=(.35, 1), ylim=(len(groups) + .65, .35))
    items = [(name, record['overall']['score']) for name, record in spine['assessments'].items()]
    subtitles = {'prior': 'Klasseprior', 'localized': 'Lokalisert 2,5D', 'global_reference': 'Global 2,5D',
                 'equal_weight_blend': '50/50-blanding'}
    indices = range(len(items))
    axes[2].barh(indices, [item[1] for item in items],
                 color=[accent if item[0] == spine['selected_for_local_inference'] else '#a1b5be' for item in items])
    axes[2].set_yticks(indices, [subtitles[name] for name, _ in items], fontsize=9)
    for index, (_name, value) in enumerate(items):
        axes[2].text(value + .015, index, f'{value:.3f}', va='center', fontsize=9)
    axes[2].invert_yaxis()
    axes[2].set(title='Korsrygg · femfolds OOF', xlabel='Vektet log loss (lavere er bedre)', xlim=(0, max(v for _, v in items) * 1.16))
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
    lines = ['# Resultater', '', '![Aggregerte resultater for thorax, mammografi og korsrygg](results.png)', '',
             'Resultatene gjelder tre forskjellige forskningsoppgaver og datasett.', '',
             '## NIH ChestX-ray14 og CBIS-DDSM', '',
             '| Testsett | Valgt modell | Antall | AUROC (95 % KI) | Gjennomsnittlig presisjon (95 % KI) | Brier før → etter |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for data, name in ((chest, 'NIH ChestX-ray14'), (mammo, 'CBIS-DDSM')):
        raw, result, ci = data['test_raw'], data['test_calibrated'], data['patient_bootstrap']
        lines.append(f'| {name} | `{data["selection"]["winner"]}` | {result["rows"]:,} | '
                     f'{result["macro_auroc"]:.3f} ({_interval(ci["macro_auroc_ci95"])}) | '
                     f'{result["macro_auprc"]:.3f} ({_interval(ci["macro_auprc_ci95"])}) | '
                     f'{raw["macro_brier"]:.4f} → {result["macro_brier"]:.4f} |')
    lines.extend(['', 'Thorax bruker makrogjennomsnittet over 14 etiketter. Mammografi bruker én malignitetsvariabel. '
                  'Intervallene bruker 400 trekk med tilbakelegging av hele pasientgrupper. Thoraxgrupper slår sammen pasient-ID-er som deler identiske bilder.', '',
                  'Brier-tapet måler gjennomsnittlig kvadratfeil for de predikerte sannsynlighetene. '
                  'Sammenlign før og etter på hvert testsett for å se hvordan kalibreringen overføres fra valideringssettet.', '',
                  '### Valideringsforsøk', '',
                  '| Forsøk | Epoke | AUROC | Gjennomsnittlig presisjon | Trente epoker | Minutter | Maks reservert VRAM |',
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
    lines.extend(['', '### Etiketter og projeksjoner', '',
                  '| Thoraxetikett | Positive testbilder | AUROC | Gjennomsnittlig presisjon |',
                  '| --- | ---: | ---: | ---: |'])
    for name, values in chest['test_calibrated']['labels'].items():
        lines.append(f'| {name.replace("_", " ")} | {values["positives"]:,} | {values["auroc"]:.3f} | {values["auprc"]:.3f} |')
    lines.extend(['', '## RSNA LumbarDISC', '',
                  '| Modell | Samlet OOF | Spinal | Foraminal | Subartikulær | Alvorlig spinal |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for name, record in spine['assessments'].items():
        keys = ('score', 'spinal', 'foraminal', 'subarticular', 'any_severe_spinal')
        lines.append(f'| `{name}` | ' + ' | '.join(f'{record["overall"][key]:.5f}' for key in keys) + ' |')
    lines.extend(['', '### Foldresultater', '',
                  '| Modell | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for name, record in spine['assessments'].items():
        lines.append(f'| `{name}` | ' + ' | '.join(f'{fold["score"]:.5f}' for fold in record['per_fold']) + ' |')
    comparison = spine['assessments'].get('equal_weight_blend')
    if comparison:
        lines.extend(['', '95 % parvise bootstrapintervaller for endring i tap mot global referanse.', ''])
        for name in ('localized', 'equal_weight_blend'):
            record = spine['assessments'][name]
            lines.append(f'- `{name}` {_interval(record["paired_delta_ci95_vs_global"])} og {record["folds_beating_global"]} av 5 bedre folder.')
    local = spine['localization']
    fold0 = spine['fold_experiments'][0]
    oracle = fold0['oracle_coordinates_sensitivity_only']['score']
    regular = fold0['validation']['score']
    lines.extend(['', f'Valgt inferenskonfigurasjon er `{spine["selected_for_local_inference"]}`. '
                  f'Rå DICOM til CSV ble verifisert på {spine_inference["study_count"]} undersøkelse med {spine_inference["rows"]} rader.', '',
                  f'Medianen for landemerkeavvik var {local["median_anchor_error_mm"]:.2f} mm, mot {local["atlas_only_median_anchor_error_mm"]:.2f} mm for den statiske referansen. '
                  f'{local["within_10mm"]:.1%} av landemerkene lå innenfor 10 mm. '
                  f'{local["missing_primary_roi_fraction"]:.2%} av primærutsnittene hadde manglende sekvens.', '',
                  f'På fold 0 var tapet {regular:.5f} med estimerte koordinater og {oracle:.5f} med tilgjengelige annoterte koordinater. '
                  f'Utforskende kalibrering på de lagrede OOF-prediksjonene ga {spine["exploratory_cross_fold_calibration"]["score"]["score"]:.5f}.', ''])
    (output / 'results.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate aggregate-only figures and result tables')
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts'))
    parser.add_argument('--output', type=Path, default=Path('docs'))
    args = parser.parse_args()
    report(args.artifacts, args.output)


if __name__ == '__main__':
    main()
