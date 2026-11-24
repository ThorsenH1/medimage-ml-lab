from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from medimage.common import read_json, write_json
from medimage.demo import PHANTOM_STUDY
from medimage.portfolio_figures import radiography as radiography_figure
from medimage.portfolio_figures import search_rungs
from medimage.portfolio_figures import spine as spine_figure

OWNER = 'ThorsenH1'
ENGINE = f'https://github.com/{OWNER}/medimage-ml-lab'
PROJECTS = {
    'chest': ('chest-xray14-ml', 'Chest X-ray14 Multi-Label Classification',
              'Group-aware training and calibrated prediction for 14 findings on frontal chest radiographs.'),
    'mammo': ('cbis-ddsm-mammography-ml', 'CBIS-DDSM Multi-View Mammography',
              'Multi-view and multi-instance classification of malignancy in a lesion-enriched mammography cohort.'),
    'spine': ('rsna-lumbar-spine-ml', 'RSNA Lumbar Spine MRI Grading',
              'DICOM geometry, nested landmark localization, and severity grading at five lumbar levels.'),
}
SOURCE = {
    'chest': '[NIH release](https://nihcc.app.box.com/v/ChestXray-NIHCC) · [Dataset paper](https://arxiv.org/abs/1705.02315)',
    'mammo': '[TCIA collection](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) · [Data descriptor](https://doi.org/10.1038/sdata.2017.177) · [Local-format dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)',
    'spine': '[RSNA challenge](https://www.rsna.org/artificial-intelligence/ai-image-challenge/lumbar-spine-degenerative-classification-ai-challenge) · [Dataset paper](https://doi.org/10.1148/ryai.250480)',
}


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + '\n', encoding='utf-8', newline='\n')


def fmt(value: float | None, places: int = 4) -> str:
    return f'{value:.{places}f}' if value is not None else '—'


def interval(values: list[float]) -> str:
    return f'[{values[0]:+.4f}, {values[1]:+.4f}]'


def predict_command(task: str) -> str:
    if task == 'spine':
        return ('& $p -m medimage.search_predict spine --selection search-artifacts/runs/spine/selection.json '
                '--search search-artifacts/runs/spine --weights $hf --root $data '
                '--cache artifacts/cache/spine-test --localization artifacts/spine/localization '
                '--incumbent artifacts/inference/spine.csv --output search-artifacts/inference/spine.csv')
    images = "--image 'C:/images/frontal.png'" if task == 'chest' else "--cc 'C:/images/cc.jpg' --mlo 'C:/images/mlo.jpg'"
    return (f'& $p -m medimage.search_predict radiography --selection search-artifacts/evaluation/{task}/selection.json '
            f'--search search-artifacts/runs/{task} --weights $hf --original artifacts {images} '
            f'--output search-artifacts/inference/{task}.json')


def overview(task: str, metrics: dict) -> tuple[str, str, str]:
    if task == 'spine':
        selected = metrics['selected']
        initial = next(item for item in metrics['candidates'] if item['name'] == 'incumbent')
        figures = f"**{selected['overall']['score']:.4f} weighted log loss** over five held-out folds, down from {initial['overall']['score']:.4f}."
        sizes = '| Unit | Count |\n| --- | ---: |\n| Training studies across folds | 1,975 |\n| Held-out study–target predictions | 49,375 |\n| Local unlabeled test studies | 1 |'
        diagram = 'A[Raw DICOM] --> B[Physical coordinates]\n    B --> C[Held-out landmark fits]\n    C --> D[2.5D crops and descriptors]\n    D --> E[Nested head search]\n    E --> F[Probability blend and RSNA CSV]'
        return figures, sizes, diagram
    current, old = metrics['test'], metrics['previous']
    kind = 'macro AUROC' if task == 'chest' else 'AUROC'
    figures = f"**{current['macro_auroc']:.4f} {kind}** versus {old['macro_auroc']:.4f} in the initial run, evaluated on the same {current['rows']:,} test units."
    if task == 'chest':
        sizes = '| Partition | Images | Patient IDs |\n| --- | ---: | ---: |\n| Train | 76,084 | 24,643 |\n| Validation | 10,411 | 3,361 |\n| Official test | 25,596 | 2,797 |'
        diagram = 'A[Frontal radiograph] --> B[Group-aware cohort]\n    B --> C[Four transfer profiles]\n    C --> D[Inner search and refits]\n    D --> E[Validation selection and calibration]\n    E --> F[14 scores]'
    else:
        sizes = '| Partition | Breasts | Patients |\n| --- | ---: | ---: |\n| Train | 1,039 | 961 |\n| Validation | 260 | 240 |\n| Official test | 336 | 316 |'
        diagram = 'A[CC and MLO full images] --> B[Tissue crop and view mask]\n    B --> C[Whole image and tiles]\n    C --> D[Transfer heads and fine-tuning]\n    D --> E[Validation-locked blend]\n    E --> F[One breast-level score]'
    return figures, sizes, diagram


def readme(task: str, metrics: dict) -> str:
    _, title, description = PROJECTS[task]
    headline, cohort, diagram = overview(task, metrics)
    choice = metrics['selected']['name'] if task == 'spine' else metrics['selection']['name']
    run = f'pwsh -File scripts/Run-Project.ps1 -Task {task} -Python $p -DataRoot $data -InitialWeights $torchCache -WeightRoot $hf'
    backbone = ('Use the Hugging Face snapshot for `timm/resnet18.a1_in1k`.' if task == 'spine' else
                'Use the Hugging Face snapshots for `timm/efficientnet_b0.ra_in1k`, `timm/resnet50.a1_in1k`, and `timm/vit_small_patch14_dinov2.lvd142m`.')
    special = {
        'chest': 'The cohort groups patient IDs linked by identical images and retains the official NIH test list. Labels are report-derived. Macro metrics give rare findings the same top-level weight as common findings.',
        'mammo': 'The cohort groups patients across mass and calcification annotations and repairs 282 test image links. The model accepts either one projection or both. The endpoint applies to breasts selected for an annotated lesion.',
        'spine': 'The public score is a paired comparison across five outer folds. The local test has one unlabeled study and verifies the 25-row submission. Supply `-HistoricalOof` and `-HistoricalSubmission` to match the reported historical reference. The workflow can also build a fresh localized reference.',
    }[task]
    return f"""# {title}

{description}

{headline}

Selected configuration `{choice}`. See the [experiment report](REPORT.md), [search trajectories](figures/search.png), [aggregate metrics](metrics.json), and [synthetic prediction example](examples/README.md).

![Benchmark figure for {title}](figures/benchmark.png)

## Dataset

{cohort}

{special} {SOURCE[task]}

## Workflow

```mermaid
flowchart LR
    {diagram}
```

The project uses a pinned [shared training engine]({ENGINE}) and has a task-specific runner. The published artifact contains source, aggregate benchmark results, and synthetic examples. Dataset files, checkpoints, and individual evaluation predictions stay in local storage.

## Reproduce the experiment

Use Python 3.11, PowerShell 7, and a CUDA-capable GPU. The original workstation used an RTX 2080 SUPER with 8 GiB VRAM.

```powershell
python -m venv .venv
$p = './.venv/Scripts/python.exe'
& $p -m pip install torch==2.14.0+cu126 torchvision==0.29.0+cu126 --index-url https://download.pytorch.org/whl/cu126
& $p -m pip install -r requirements.txt
```

Set the data and pretrained weight locations for your machine. The initial ResNet-18 weight directory uses `hub/checkpoints/resnet18-f37072fd.pth`. The Hugging Face cache uses `refs/main` and `snapshots/<revision>/model.safetensors`. {backbone} [Initial ResNet-18 weights](https://download.pytorch.org/models/resnet18-f37072fd.pth)

```powershell
$data = 'D:/datasets/{task}'
$torchCache = 'D:/models/torch'
$hf = 'D:/models/huggingface/hub'
{run}
```

The runner writes the initial experiment under `artifacts/` and the extended search under `search-artifacts/`. For a larger volume, supply `-Artifacts` and `-Search` and use matching paths at inference. The prepared caches and training stages can resume from saved boundaries.

## Predict from raw input

Use the model selection and checkpoint roots produced by the runner.

```powershell
{predict_command(task)}
```

The [example](examples/README.md) runs the selected local models on deterministic synthetic inputs. This is a research and education project.
"""


def examples(task: str, source: Path, target: Path, metrics: dict) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if task == 'spine':
        shutil.copyfile(source / 'phantom.png', target / 'phantom.png')
        from medimage.spine_metrics import read_submission

        data = read_submission(source / 'prediction.csv', np.array([PHANTOM_STUDY]))
        frame = pd.read_csv(source / 'prediction.csv')
        electronic = {'input': 'synthetic MRI phantom', 'model': metrics['selected']['name'], 'rows': len(frame),
                      'predictions': frame.to_dict(orient='records')}
        display = ['| Target | Normal/Mild | Moderate | Severe |', '| --- | ---: | ---: | ---: |']
        for row, proba in zip(frame.row_id.iloc[:5], data[0, :5], strict=True):
            display.append(f'| {row.split("_", 1)[1]} | {proba[0]:.4f} | {proba[1]:.4f} | {proba[2]:.4f} |')
        image = '![Synthetic MRI planes](phantom.png)'
        footer = 'The JSON contains all 25 targets. Each row has three finite probabilities that sum to one.'
    else:
        record = read_json(source / 'prediction.json')
        if record['model'] != metrics['selection']['name']:
            raise ValueError('The synthetic example used a different selected model')
        electronic = {'input': 'synthetic radiography phantom', 'model': record['model'],
                      'projections': record['projections'], 'probabilities': record['probabilities']}
        display = ['| Output | Calibrated score |', '| --- | ---: |']
        for label, value in record['probabilities'].items():
            if not 0 <= value <= 1:
                raise ValueError('Synthetic probabilities must be finite and bounded')
            display.append(f'| {label.replace("_", " ")} | {value:.4f} |')
        images = ('phantom.png',) if task == 'chest' else ('phantom-cc.png', 'phantom-mlo.png')
        for name in images:
            shutil.copyfile(source / name, target / name)
        image = ' '.join(f'![Synthetic input {name}]({name})' for name in images)
        footer = 'The scores demonstrate the input and output contract for the prediction command.'
    write_json(target / 'prediction.json', electronic)
    write(target / 'README.md', f"""# Synthetic prediction example

The inputs are deterministic mathematical patterns generated by the shared engine. The scores below were produced by the selected trained model on these synthetic inputs. They illustrate the inference interface and have no clinical interpretation.

{image}

{chr(10).join(display)}

{footer} [Complete JSON response](prediction.json)
""")


def radiography_metrics(root: Path, original: Path, task: str, target: Path) -> dict:
    result = read_json(root / 'evaluation' / task / 'metrics.json')
    plan = read_json(root / 'evaluation' / task / 'selection.json')
    search = read_json(root / 'runs' / task / 'search.json')
    search_groups = {item['profile']: [row for row in search['trials'] if row['profile'] == item['profile']]
                     for item in search['finalists']}
    search_rungs(search_groups, search['budgets'], f'{PROJECTS[task][1]} | inner search',
                 target / 'figures' / 'search.png', minimize=False)
    old = result['previous']
    new = result['test']
    with np.load(root / 'evaluation' / task / 'test-predictions.npz', allow_pickle=False) as run, np.load(
            original / 'evaluation' / task / 'test-predictions.npz', allow_pickle=False) as baseline:
        if not np.array_equal(run['rows'], baseline['rows']) or not np.array_equal(run['labels'], baseline['labels']):
            raise ValueError('Radiography paired figure requires aligned predictions')
        radiography_figure(task, result, run['labels'], baseline['probabilities'], run['probabilities'],
                           target / 'figures' / 'benchmark.png')
    boot = result['patient_bootstrap']
    table = ['# Experiment report', '', f"The selected configuration is `{plan['selected']['name']}` with weights {json.dumps(plan['selected']['weights'])}.", '',
             '| Metric | Initial | Selected | 95% paired delta interval |', '| --- | ---: | ---: | ---: |']
    for metric, label, key in (('macro_auroc', 'AUROC', 'paired_auroc_delta_ci95'), ('macro_auprc', 'Average precision', 'paired_auprc_delta_ci95'),
                               ('macro_brier', 'Brier score', ''), ('macro_ece_10', 'Expected calibration error', '')):
        table.append(f'| {label} | {fmt(old[metric])} | {fmt(new[metric])} | {interval(boot[key]) if key else "—"} |')
    table.extend(['', f"The selected AUROC has a patient-cluster 95% CI of [{boot['macro_auroc_ci95'][0]:.4f}, {boot['macro_auroc_ci95'][1]:.4f}] on {new['rows']:,} test units. "
                  f"The paired comparison uses {boot['valid_replicates']} patient-cluster bootstrap draws. The test cohort was also used for the initial benchmark. The new model, blend, and calibration were locked before this round's test scoring.", '',
                  '## Validation candidates', '', '| Model or blend | Selection score | AUROC | Average precision |', '| --- | ---: | ---: | ---: |'])
    for item in plan['candidates']:
        score = item['metrics']
        table.append(f'| `{item["name"]}` | {item["score"]:.5f} | {score["macro_auroc"]:.5f} | {score["macro_auprc"]:.5f} |')
    table.extend(['', '## Search profile winners', '',
                  '| Profile | Epochs per refit | Head | Width | Learning rate | Weight decay | Dropout | Positive-weight exponent |',
                  '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |'])
    for item in search['finalists']:
        recipe = item['recipe']
        table.append(f'| `{item["profile"]}` | {item["fit_epochs"]} | {recipe["head"]} | {recipe["width"]} | {recipe["learning_rate"]} | {recipe["decay"]} | {recipe["dropout"]} | {recipe["class_power"]} |')
    table.extend(['', '## Benchmark breakdown', '', '| Finding or endpoint | Positive units | Initial AUROC | Selected AUROC | Initial AP | Selected AP |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for label, row in new['labels'].items():
        first = old['labels'][label]
        table.append(f'| {label.replace("_", " ")} | {row["positives"]:,} | {fmt(first["auroc"])} | {fmt(row["auroc"])} | {fmt(first["auprc"])} | {fmt(row["auprc"])} |')
    table.extend(['', '| Subgroup | Units | AUROC | Average precision | Brier |', '| --- | ---: | ---: | ---: | ---: |'])
    for name, summary in result['subgroups'].items():
        table.append(f'| {name.replace("_", " ")} | {summary["rows"]:,} | {fmt(summary["macro_auroc"])} | {fmt(summary["macro_auprc"])} | {fmt(summary["macro_brier"])} |')
    table.extend(['', '## Search protocol', '',
                  f"Each of four encoder and resolution profiles starts with 12 recipes. The resource rungs use {search['budgets']} epochs. "
                  f"Inner ranking uses {search['inner_train_count']:,} fit units and {search['inner_val_count']:,} held-out units grouped by patient. "
                  f"Each profile winner is refitted with two seeds on {search['refit_train_count']:,} units. "
                  f"Final choice and calibration use {search['final_val_count']:,} validation units.", '',
                  'The radiography blend combines logits after correcting for positive-class weighting. The displayed curves use a fixed 101-point grid. '
                  'Mammography reliability bins appear when they have at least eight samples. Aggregate metrics and trial summaries are in [metrics.json](metrics.json).'])
    write(target / 'REPORT.md', '\n'.join(table))
    write_json(target / 'metrics.json', {'benchmark': result, 'validation_candidates': plan['candidates'],
                                         'search_trials': search['trials'], 'profile_winners': search['finalists']})
    return result


def spine_metrics(root: Path, original: Path, target: Path) -> dict:
    plan = read_json(root / 'runs' / 'spine' / 'selection.json')
    initial = read_json(original / 'evaluation' / 'spine' / 'metrics.json')
    synthetic = read_json(root / 'inference' / 'spine.json')
    spine_figure(plan, initial, target / 'figures' / 'benchmark.png')
    search_groups = {f"Outer fold {item['fold']}": item['trials'] for item in plan['folds']}
    search_rungs(search_groups, plan['budgets'], 'RSNA LumbarDISC | nested search',
                 target / 'figures' / 'search.png', minimize=True)
    table = ['# Experiment report', '', f"The selected configuration is `{plan['selected']['name']}`.", '',
             '## Out-of-fold comparison', '', '| Model | Overall | Spinal | Foraminal | Subarticular | Any severe spinal | Folds improved | Paired delta CI |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for variant in plan['candidates']:
        keys = ('score', 'spinal', 'foraminal', 'subarticular', 'any_severe_spinal')
        table.append(f"| `{variant['name']}` | " + ' | '.join(fmt(variant['overall'][key]) for key in keys)
                     + f" | {variant['better_folds']}/5 | {interval(variant['delta_ci95'])} |")
    table.extend(['', '## Nested selection', '', '| Outer fold | Inner validation fold | Inner fit studies | Outer validation studies | Head | Descriptor | Epochs | Standalone loss | With inner calibration |',
                  '| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |'])
    for item in plan['folds']:
        table.append(f'| {item["fold"]} | {item["inner_val"]} | {item["inner_training_count"]} | {item["validation_studies"]} | '
                     f'{item["recipe"]["kind"]} | {item["recipe"]["descriptor"]} | {item["fit_epochs"]} | {fmt(item["outer"]["score"])} | {fmt(item["outer_calibrated"]["score"])} |')
    localizer = initial['localization']
    table.extend(['', '## Localization and endpoint', '',
                  f"Median landmark error was {localizer['median_anchor_error_mm']:.2f} mm. The training-fold atlas reference had a {localizer['atlas_only_median_anchor_error_mm']:.2f} mm median. "
                  f"{localizer['within_10mm']:.1%} of points were within 10 mm. Training-region localization excludes the study's own fold and the outer holdout, plus the inner holdout where applicable.", '',
                  'The classifier searches over 12 recipes per outer fold, varying head, descriptor, regularization, class weights, ordinal penalty, and the severe-stenosis term. '
                  'The inner holdout chooses the refit budget. Two seeds train on the outer development split. The paired interval and fold-count rule gate promotion against the previous ensemble.', '',
                  f"Raw inference on the local unlabeled test completed with {synthetic['rows']} finite competition rows. See [the synthetic example](examples/README.md) for a public input and output."
                  ])
    write(target / 'REPORT.md', '\n'.join(table))
    shared = {**plan, 'folds': [{key: value for key, value in fold.items() if key != 'checkpoints'} for fold in plan['folds']],
              'localization': localizer}
    write_json(target / 'metrics.json', shared)
    return plan


def build(task: str, search: Path, original: Path, demo: Path, output: Path, engine: str) -> None:
    if not re.fullmatch(r'[0-9a-f]{40}', engine):
        raise ValueError('Pin the shared source to a complete Git commit')
    source = Path(__file__).resolve().parents[1]
    slug = PROJECTS[task][0]
    target = output / slug
    (target / 'figures').mkdir(parents=True, exist_ok=True)
    (target / 'scripts').mkdir(exist_ok=True)
    metrics = (spine_metrics(search, original, target) if task == 'spine' else
               radiography_metrics(search, original, task, target))
    examples(task, demo / task, target / 'examples', metrics)
    for name in ('Run-Project.ps1', 'Search-Models.ps1',
                 *(['Run-Spine.ps1'] if task == 'spine' else ['Run-Radiography.ps1', 'Extract-SearchFeatures.ps1'])):
        shutil.copyfile(source / 'scripts' / name, target / 'scripts' / name)
    shutil.copyfile(source / '.gitignore', target / '.gitignore')
    write(target / 'README.md', readme(task, metrics))
    write(target / 'requirements.txt', f'medimage-ml-lab @ git+{ENGINE}.git@{engine}')
    print(f'Prepared {slug}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Export English project repositories with aggregate and synthetic evidence')
    for name in ('search', 'original', 'demo', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--task', choices=tuple(PROJECTS), nargs='+', default=tuple(PROJECTS))
    args = parser.parse_args()
    for task in args.task:
        build(task, args.search, args.original, args.demo, args.output, args.engine)


if __name__ == '__main__':
    main()
