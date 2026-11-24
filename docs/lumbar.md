# RSNA LumbarDISC lumbar MRI

The system predicts three grades for five condition/side categories at five lumbar levels. The inputs are sagittal T2/STIR, sagittal T1, and axial T2. DICOM spacing, position, and orientation map resampled pixels back to physical coordinates.

## Localization and grading

1. A kernel model estimates five mid-sagittal landmarks from image descriptors. It mixes two RBF bandwidths and has a training-fold median fallback.
2. A fold-specific atlas estimates offsets from those landmarks to the foraminal and subarticular targets.
3. The targets are projected into their views in millimeters. Each 2.5D crop contains adjacent slices. Missing views have an explicit mask.
4. A shared ResNet-18 encodes cropped regions. Attention combines levels, conditions, and localization quality. The loss combines severity-weighted cross-entropy, an ordinal component, and a study-level severe spinal canal signal.

## Validation and leakage controls

The cohort contains 1,975 studies on five frozen folds. The outer held-out fold is excluded from locator fitting. The train row's own fold is excluded when generating that row's crop. During hyperparameter search, the inner validation fold is also excluded when preparing inner training crops. The pipeline checks these provenance masks before fitting. The initial fold-0 sensitivity analysis evaluates the same classifier with known coordinates to measure the effect of localization quality.

The exact RSNA weighted log loss is the primary metric. The historical reference uses five uniformly spaced 224-pixel slices per series, a shared global ResNet-18, and mean pooling. Promotion requires improvement on at least three folds and an upper paired bootstrap bound below zero.

The local test contains one unlabeled study. The integration test checks 25 rows in the supplied template's order. [Official challenge and data](https://www.rsna.org/artificial-intelligence/ai-image-challenge/lumbar-spine-degenerative-classification-ai-challenge)

## Historical comparison

The original workstation uses five historical checkpoints and their original OOF predictions. These are local inputs for the published comparison. The MRI project can also run independently with a newly trained localized reference.

~~~powershell
$ref = 0..4 | ForEach-Object { "D:\rsna2024_work\runs\global_resnet18_v1_fold$_\best.pt" }
& $p -m medimage.spine_reference --root ../lumbar_spine_degenerative_classification --checkpoints @ref --trusted-legacy-format --output artifacts/baseline/global-reference.csv
pwsh -File scripts/Run-Spine.ps1 -Python $p -Weights $weights -DataRoot ../lumbar_spine_degenerative_classification -BaselineOof 'D:\rsna2024_work\oof\global_resnet18_v1\oof_predictions.csv' -BaselineSubmission artifacts/baseline/global-reference.csv
~~~

## Limitations and learning objectives

Axial selection favors coverage, while sagittal selection favors slice count. Truncated coverage, unusual anatomy, and acquisition differences affect the atlas. The project illustrates DICOM geometry, severity-aware scoring, nested validation, and how a paired comparison supports an ensemble decision. [Initial results](results.md) · [Extended search](model-search.md)
