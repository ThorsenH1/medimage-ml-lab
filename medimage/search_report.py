from __future__ import annotations

import argparse
from pathlib import Path

from medimage.common import read_json


def interval(values: list[float], *, decimals: int = 4) -> str:
    return f'[{values[0]:.{decimals}f}, {values[1]:.{decimals}f}]'


def report(root: Path, output: Path) -> None:
    radiography = {task: (read_json(root / 'runs' / task / 'search.json'),
                          read_json(root / 'runs' / task / 'adaptation' / 'result.json'),
                          read_json(root / 'evaluation' / task / 'metrics.json')) for task in ('chest', 'mammo')}
    spine = read_json(root / 'runs' / 'spine' / 'selection.json')
    inverse = read_json(root / 'inference' / 'spine.json')
    high_resolution = read_json(root / 'runs' / 'chest' / 'adaptation-reference' / 'result.json')
    prior = next(item for item in spine['candidates'] if item['name'] == 'incumbent')
    names = {'chest': 'Chest X-ray', 'mammo': 'Mammography'}
    starts = sum(values[2]['initial_search_configurations'] for values in radiography.values()) + spine['initial_trials']
    all_rungs = sum(len(item[0]['trials']) for item in radiography.values()) + sum(len(fold['trials']) for fold in spine['folds'])
    lines = ['# Extended model and hyperparameter search', '',
             f'The search started {starts} trial configurations across tasks and folds and completed {all_rungs} rung evaluations. '
             'Each X-ray profile was refitted with two seeds. The strongest new profile for each X-ray task also '
             'had its last layers fine-tuned from images. The initial chest X-ray model was also fine-tuned at 224 pixels. '
             'MRI finalists were refitted with two seeds per outer fold.', '',
             '## Results', '',
             '| Task | Selected model | Previous score | New score | 95% paired delta CI |',
             '| --- | --- | ---: | ---: | ---: |']
    for task, (_, _, metrics) in radiography.items():
        boot = metrics['patient_bootstrap']
        selected = metrics['selection']['name']
        lines.append(f'| {names[task]} | `{selected}` | {metrics["previous"]["macro_auroc"]:.4f} AUROC | '
                     f'{metrics["test"]["macro_auroc"]:.4f} AUROC | {interval(boot["paired_auroc_delta_ci95"])} |')
    selected = spine['selected']
    lines.append(f'| Lumbar MRI | `{selected["name"]}` | {prior["overall"]["score"]:.4f} log loss | '
                 f'{selected["overall"]["score"]:.4f} log loss | {interval(selected["delta_ci95"])} |')
    lines.extend(['', 'Higher AUROC and lower log loss are better. Chest X-ray uses macro averages over 14 findings. '
                  'The mammography cohort is lesion-enriched. Intervals estimate differences from the initial scores '
                  'with patient- or study-cluster bootstrap.', '',
                  '### Radiography', '',
                  '| Task | Test units | Average precision before → after | Brier before → after |',
                  '| --- | ---: | ---: | ---: |'])
    for task, (_, _, metrics) in radiography.items():
        old, new = metrics['previous'], metrics['test']
        lines.append(f'| {names[task]} | {new["rows"]:,} | {old["macro_auprc"]:.4f} → {new["macro_auprc"]:.4f} | '
                     f'{old["macro_brier"]:.4f} → {new["macro_brier"]:.4f} |')
    lines.extend(['', '### Validation before the benchmark lock', '',
                  '| Task | Candidate | Validation objective | AUROC | AP |', '| --- | --- | ---: | ---: | ---: |'])
    for task, (search, adapted, metrics) in radiography.items():
        plan = read_json(root / 'evaluation' / task / 'selection.json')
        entries = {item['name']: item for item in plan['candidates'] if len(item['weights']) == 1}
        additions = (high_resolution['name'],) if task == 'chest' else ()
        for name in ('incumbent-global', *(row['profile'] for row in search['finalists']), adapted['name'],
                     *additions, metrics['selection']['name']):
            if name not in entries:
                entries[name] = next(item for item in plan['candidates'] if item['name'] == name)
        for name, item in entries.items():
            measure = item['metrics']
            lines.append(f'| {names[task]} | `{name}` | {item["score"]:.5f} | {measure["macro_auroc"]:.4f} | {measure["macro_auprc"]:.4f} |')
    lines.extend(['', '### Chosen hyperparameters per radiography profile', '',
                  '| Task | Profile | Head | Width | LR | Weight decay | Dropout | Positive-weight exponent | Patch dropout |',
                  '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for task, (search, _, _) in radiography.items():
        for item in search['finalists']:
            recipe = item['recipe']
            lines.append(f'| {names[task]} | `{item["profile"]}` | {recipe["head"]} | {recipe["width"]} | '
                         f'{recipe["learning_rate"]} | {recipe["decay"]} | {recipe["dropout"]} | {recipe["class_power"]} | {recipe["tile_dropout"]} |')
    lines.extend(['', '### Lumbar MRI', '',
                  '| MRI candidate | Weighted log loss | Folds improved | 95% paired delta CI |',
                  '| --- | ---: | ---: | ---: |'])
    for item in spine['candidates']:
        lines.append(f'| `{item["name"]}` | {item["overall"]["score"]:.5f} | {item["better_folds"]} / 5 | {interval(item["delta_ci95"])} |')
    lines.extend(['', '| Outer fold | Inner validation fold | Selected head | Descriptor | Selected epochs | After inner temperature scaling |',
                  '| ---: | ---: | --- | --- | ---: | ---: |'])
    for row in spine['folds']:
        lines.append(f'| {row["fold"]} | {row["inner_val"]} | {row["recipe"]["kind"]} | {row["recipe"]["descriptor"]} | {row["fit_epochs"]} | {row["outer_calibrated"]["score"]:.5f} |')
    lines.extend(['', f'Raw DICOM inference produced {inverse["rows"]} valid submission rows from {inverse["studies"]} local test study.', '',
                  '## Protocol', '',
                  '- Chest X-ray and mammography reuse the official benchmark cohorts from the initial study. The search ranks configurations on '
                  'an inner patient-grouped part of training. Finalists train on the whole training set and compete against the '
                  'incumbent and prespecified blends on the original validation set. Chest X-ray maximizes '
                  'the mean of macro AUROC and AP. Mammography maximizes AUROC.',
                  '- The radiography search covers EfficientNet-B0 at 160 and 224 pixels, ResNet-50 at 224, and DINOv2 ViT-S at 224. '
                  'Each profile has 12 combinations of head, learning rate, dropout, width, weight decay, and positive weighting. '
                  'Successive halving advances 12 to 4 to 2 candidates with increasing epoch budgets. [Successive halving](https://proceedings.mlr.press/v51/jamieson16.html) · [DINOv2](https://arxiv.org/abs/2304.07193)',
                  '- Mammography uses one whole image and six overlapping tiles per available view. Radiography blends '
                  'combine logits after class-weight correction. MRI blends combine probabilities.',
                  '- MRI uses five outer folds and a separate inner holdout in each. Outer and inner validation data are excluded from '
                  'localizer fitting for inner training images. Local, convolutional, and attention heads use '
                  'deep and texture descriptors with varied ordinal penalties and severe-stenosis weights. MRI blends must win on '
                  'at least three folds and have a paired interval with upper bound below zero to replace the incumbent.',
                  '- MRI and chest X-ray select the refit epoch count on inner validation. The mammography search ranks configurations '
                  'at each rung endpoint.',
                  '- The X-ray test cohorts were used in the first investigation and are reused benchmarks. This round locks its chosen models, blend weights, and '
                  'calibration before scoring. The local MRI test has one study without available grades. '
                  'Performance is assessed on the frozen folds. The local study verifies the inference format.', '',
                  'Experiment checkpoints, detailed timings, and individual predictions are stored in the local search directory. '
                  'The repository contains training code and aggregate results.', ''])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Write an aggregate research report from completed search artifacts')
    parser.add_argument('--search', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report(args.search, args.output)


if __name__ == '__main__':
    main()
