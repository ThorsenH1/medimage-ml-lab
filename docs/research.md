# Research sources and methods

## Chest X-ray

The original NIH paper describes labels extracted from reports in the first release. The project uses the later 14-label release and its official lists. Report labels can be incomplete and can differ from expert image review. [Original NIH paper](https://arxiv.org/abs/1705.02315) · [RSNA data card](https://atlas.rsna.org/cards/66962171-1dcb-431d-89b0-08a561f1c8ed)

## Mammography

CBIS-DDSM provides pathology, full mammograms, and separate lesion annotations. The annotated cohort has a different sampling scheme from a screening population. The local JPEG conversion was joined using the exact full-image references. In rows without a series description, study and series UIDs and full-resolution dimensions support the repair. [TCIA collection](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) · [Data paper](https://doi.org/10.1038/sdata.2017.177) · [JPEG distribution](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)

## Lumbar MRI

LumbarDISC was collected to grade degenerative findings at five lumbar levels. The competition metric uses weighted log loss for three condition families plus an any-severe spinal canal component. [Dataset paper](https://doi.org/10.1148/ryai.250480) · [Challenge and rules](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/rules) · [Metric discussion](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/521786)

Public solutions motivated localize-then-classify. The third-place team mapped sagittal positions to oblique axial slices in millimeters. The fifth-place team modeled level relations following heatmap localization. This engine uses kernel regression, DICOM projection, fold-specific atlas offsets, and 2.5D classifiers. [First place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution) · [Second place](https://github.com/brendanartley/RSNA-2024-Competition) · [Third place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453) · [Fourth place](https://github.com/yu4u/kaggle-rsna2024-4th) · [Fifth place](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/two-people-5th-place-solution) · [Seventh place](https://github.com/lhwcv/solution-rsna-2024-lumbar-spine) · [Ninth place](https://github.com/adamnarai/kaggle-rsna-2024)

Projection uses ImagePositionPatient, ImageOrientationPatient, and PixelSpacing. The letterbox conversion accounts for OpenCV's pixel-center convention. [DICOM image-plane standard](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html)

## Training

The initial runs use ImageNet ResNet-18. The extended study compares transfer profiles and progressively allocates training through grouped inner validation. DINOv2 supplies self-supervised visual representations. CUDA training uses FP16 autocast with gradient scaling. [Torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) · [DINOv2](https://arxiv.org/abs/2304.07193) · [Successive halving](https://proceedings.mlr.press/v51/jamieson16.html) · [PyTorch AMP](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
