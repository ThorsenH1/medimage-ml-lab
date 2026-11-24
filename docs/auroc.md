# Medical imaging discrimination experiments

The experiments target macro AUROC on the established patient-grouped radiography benchmarks and observed grades on the frozen MRI folds. The previous radiography test scores were 0.7900 on NIH ChestX-ray14 and 0.7468 on CBIS-DDSM. The MRI audit measures 0.8534 on the existing out-of-fold predictions, with a 95% study-bootstrap interval of 0.8472–0.8599. [MRI protocol](lumbar.md)

## Experimental design

| Task | Training unit | Validation and test protocol |
| --- | --- | --- |
| Chest radiography | Frontal radiograph with 14 binary findings | Development grouped by patient and exact-image component. The original test manifest has 25,596 images. |
| Mammography | One breast with one or two whole views | Development grouped by patient. The official test has 336 breasts after the prior overlap and ambiguity audit. |
| Lumbar MRI | DICOM study with 25 anatomical targets | Five frozen outer folds. Each held-out study contributes one out-of-fold prediction. |

Radiography model selection reads the training checkpoints and validation logits. It checks files and row alignment, fits score scaling on five disjoint validation groups, and ranks single models and blends by validation macro AUROC. Average precision breaks ties. Final calibration and thresholds are fitted on validation. Test evaluation loads the frozen decision and scores its components on the same official benchmark used for earlier experiments. The study does not establish performance on a new external population.

## Chest radiography

The initial DenseNet-121 experiments use external CheXpert supervision and ImageNet supervision. The CheXpert artifact comes from TorchXRayVision's CheXpert-only release. Its SHA-256 is verified before reading its encoder tensors. NIH-containing releases are excluded. The 384-pixel experiment also warm starts from the existing NIH development checkpoint and adapts its late residual stages on the original training split. [CheXpert checkpoint source](https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/models.py) · [Torchvision DenseNet-121](https://download.pytorch.org/models/densenet121-a639ec97.pth)

The ensemble compares shared weights with per-finding weights regularized towards the shared mix. Each set of per-finding weights is estimated within the grouped validation procedure. All 14 findings contribute equally to the primary metric.

## Mammography

The preprocessing normalizes intensity, keeps a connected breast region, aligns the chest wall, preserves the aspect ratio, and records the same geometric transform for training lesion marks. The training tables link lesion cases to full views and a full-size mask series. Val and test images are passed as pixels with a view-availability mask.

The 1024-pixel training cache contains 2,105 aligned training lesions. Masks for four combinations of lesion type and pathology supervise the spatial head on 1,839 complete training views. The classifier fuses CC and MLO tokens and learns to handle a missing projection. The feature-pyramid variant makes intermediate encoder features available to the spatial head and synchronizes geometric augmentation between image and mask. The patch branch uses five patch classes, scans the entire breast at two scales, and chooses a pooling rule on the held-out validation breasts. [Full-image localization and classification](https://pmc.ncbi.nlm.nih.gov/articles/PMC6715802/) · [Dual-view lesion supervision](https://www.mdpi.com/2306-5354/13/12/1410)

The Mammo-CLIP variant imports only image-encoder tensors from the publicly released contrastive EfficientNet-B5 checkpoint. The authors document UPMC mammography pretraining. The state loader verifies the source checksum and checks every converted encoder tensor during loading. [Mammo-CLIP paper](https://papers.miccai.org/miccai-2024/paper/0926_paper.pdf) · [Checkpoint](https://huggingface.co/shawn24/Mammo-CLIP/blob/main/Pre-trained-checkpoints/b5-model-best-epoch-7.tar)

## Reports and prediction

The local evaluation report includes ROC, precision–recall, subgroup summaries, paired patient-bootstrap intervals and examples with held-out reference labels. Public portfolio reports use aggregate benchmarks and synthetic input examples.

```powershell
& $p -m medimage.auc_report --output "$experiment/evaluation/mammo-locked-1" --previous $search
& $p -m medimage.auc_predict --selection "$experiment/evaluation/mammo-locked-1/selection.json" --work $experiment --previous $search --original artifacts --weights $hf --clip "$experiment/weights/mammo-clip-b5.tar" --cc path/to/full-cc.png --mlo path/to/full-mlo.png --output "$experiment/inference/mammography.json"
```
