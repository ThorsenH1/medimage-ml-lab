# MedImage ML Lab

Three medical-imaging machine-learning projects on a shared training engine. Each project covers dataset auditing, grouped splitting, GPU training, model selection, calibration, and inference from raw images.

| Project | Scope | Selected benchmark | Project repository |
| --- | --- | --- | --- |
| NIH ChestX-ray14 | 14 findings on frontal radiographs | 0.7900 macro AUROC, up from 0.7787 | [Chest X-ray classification](https://github.com/ThorsenH1/chest-xray14-ml) |
| CBIS-DDSM | Malignancy in a lesion-enriched mammography cohort | 0.7468 AUROC, up from 0.6858 | [Multi-view mammography](https://github.com/ThorsenH1/cbis-ddsm-mammography-ml) |
| RSNA LumbarDISC | Three grades for 25 lumbar condition-level targets | 0.8534 macro AUROC and 0.5260 weighted log loss on five held-out folds | [Lumbar MRI grading](https://github.com/ThorsenH1/rsna-lumbar-spine-ml) |

[Discrimination experiments](docs/auroc.md) · [Extended model search](docs/model-search.md) · [Initial benchmarks](docs/results.md) · [Research sources](docs/research.md) · [Study guide](docs/study-guide.md)

## Shared engine

- [Chest X-ray methods](docs/chest.md) covers multi-label classification, patient grouping, duplicate controls, and the official NIH benchmark.
- [Mammography methods](docs/mammography.md) covers CC/MLO projections, metadata repair, and multi-instance learning.
- [Lumbar MRI methods](docs/lumbar.md) covers DICOM geometry, nested landmark fitting, and the exact four-part RSNA metric.

The project repositories pin this shared engine by commit. Each includes its own training entry point, aggregate performance data, figures, and prediction examples.

## Install

The experiments ran on Windows with Python 3.11, PyTorch 2.14.0, CUDA 12.6, and an NVIDIA RTX 2080 SUPER with 8 GiB VRAM. Training uses FP16 autocast, FP32 losses, and disk-backed pixel and embedding stores. [PyTorch AMP documentation](https://docs.pytorch.org/docs/2.14/amp.html)

~~~powershell
python -m venv .venv
$p = '.\.venv\Scripts\python.exe'
& $p -m pip install torch==2.14.0+cu126 torchvision==0.29.0+cu126 --index-url https://download.pytorch.org/whl/cu126
& $p -m pip install -e .
& $p -m pytest -q
~~~

The initial runs use the ImageNet ResNet-18 checkpoint under `hub/checkpoints/resnet18-f37072fd.pth`. The extended runs use local Hugging Face snapshots for `timm/efficientnet_b0.ra_in1k`, `timm/resnet50.a1_in1k`, `timm/vit_small_patch14_dinov2.lvd142m`, and `timm/resnet18.a1_in1k`. Preserve the standard `refs/main` and `snapshots/<revision>/model.safetensors` layout. [Torchvision weights](https://download.pytorch.org/models/resnet18-f37072fd.pth) · [DINOv2](https://arxiv.org/abs/2304.07193)

Source datasets, trained parameters, and individual benchmark predictions are held locally. The public projects contain code, documentation, aggregate evidence, and synthetic examples. The models are research and education prototypes.

## Reproduce the workstation runs

Use the commands below with your own data and weight paths as needed. Run GPU jobs sequentially.

~~~powershell
$p = 'D:\rsna2024_work\envs\rsna\python.exe'
$weights = 'D:\rsna2024_work\torch-cache'
$mammo = '../CBIS DDSM Breast Cancer Image Dataset'
$spine = '../lumbar_spine_degenerative_classification'
pwsh -File scripts/Run-Radiography.ps1 -Python $p -Weights $weights -Task chest -DataRoot ../chest_xray
pwsh -File scripts/Run-Radiography.ps1 -Python $p -Weights $weights -Task mammo -DataRoot $mammo
pwsh -File scripts/Run-Spine.ps1 -Python $p -Weights $weights -DataRoot $spine
~~~

For the historical MRI comparator, use the inputs listed in the [MRI methods](docs/lumbar.md). The extended search uses the first run's cohorts and local checkpoints and needs at least 20 GiB of additional disk space.

~~~powershell
$search = 'D:\medimage_ml_search'
$hf = Join-Path ([Environment]::GetFolderPath('UserProfile')) '.cache\huggingface\hub'
New-Item -ItemType Directory -Path $search -Force | Out-Null
& $p -m medimage.search_features pixels chest --cohort artifacts/cache/chest --root ../chest_xray --output "$search/pixels/chest"
& $p -m medimage.search_features pixels mammo --cohort artifacts/cache/mammo-full --root $mammo --output "$search/pixels/mammo"
pwsh -File scripts/Extract-SearchFeatures.ps1 -Python $p -WeightRoot $hf -SearchRoot $search
pwsh -File scripts/Search-Models.ps1 -Python $p -WeightRoot $hf -SearchRoot $search -SpineData $spine
pwsh -File scripts/Verify-Search.ps1 -Python $p -WeightRoot $hf -SearchRoot $search -ChestData ../chest_xray -MammographyData $mammo -SpineData $spine
~~~

The search ranks heads on inner development splits, refits finalists with two seeds, and locks the winner before benchmark scoring. MRI uses a separate inner holdout inside each outer fold and checks ROI fitting provenance. The strongest new X-ray transfer profile for each task is fine-tuned directly on images. The thorax comparison also includes a higher-resolution continuation of the initial model.

Caches and long runs resume from their last saved boundary. Each Python module exposes its inputs through `--help`. For prediction, retain `selection.json` and its declared checkpoint roots.

~~~powershell
& $p -m medimage.search_predict radiography --selection "$search/evaluation/chest/selection.json" --search "$search/runs/chest" --weights $hf --original artifacts --image 'C:\images\frontal.png' --output "$search/inference/new-chest.json"
~~~

For mammography, use the mammography selection and run directories and pass `--cc` and/or `--mlo` for the same breast. For MRI, use `medimage.search_predict spine` with an RSNA-format cohort and a matching first-round submission.
