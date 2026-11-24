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

The Mammo-CLIP variant imports only image-encoder tensors from the publicly released contrastive EfficientNet-B5 checkpoint. Ghosh and colleagues document UPMC mammography pretraining. The state loader verifies the source checksum and checks every converted encoder tensor during loading. Local research adaptations use the authors' CC BY-NC-SA 4.0 weights. [Mammo-CLIP paper](https://papers.miccai.org/miccai-2024/paper/0926_paper.pdf) · [Checkpoint and license](https://huggingface.co/shawn24/Mammo-CLIP)

The follow-up experiments include a 1,536-pixel spatial model, a higher-resolution Mammo-CLIP refinement and a multi-task classifier with mass and calcification specialists. The latter learns its routing from known training lesion types. At inference the classifier estimates routing from the images and needs only the available whole views. The EfficientNet-B3 experiment uses a pinned ImageNet initialization and an independently seeded dual-view fine-tune, motivated by published breast-level comparisons. [Direct and two-view classifier study](https://doi.org/10.1002/ima.70436) · [Pinned ImageNet checkpoint](https://huggingface.co/timm/efficientnet_b3.ra2_in1k/blob/707c9b711f7f114b4c1601d566a4aaf9fdba4a49/model.safetensors)

The breast proposal ensemble splits development patients into three disjoint folds and fits each patch detector on two folds. Each development breast contributes proposals from a detector that never saw that patient. The attention head learns breast-level malignancy from these proposals. Validation and test predictions average across detectors using only whole images. A grouped validation comparison also supports regularized nonnegative blends that optimize smooth positive–negative ranking loss.

The texture experiment uses intensity distributions, spatial differences and clusters of bright local peaks within breast tissue. It measures several scales from whole views, retains a projection-availability mask and fits feature normalization on training breasts. The classifier tests four regularization and capacity settings to measure whether these image descriptors complement the neural models.

For two trained spatial models, validation also compares a fixed test-time augmentation policy. The policy averages the original logit with a mean across four deterministic, mild rotation and intensity transformations. Reapplying the selected policy requires the same whole-view pixels and projection availability.

A downstream transfer experiment initializes the B5 image tower from the authors' separately fine-tuned RSNA breast-cancer model. It discards the source head and fits the CBIS-DDSM breast head and spatial objective on development patients. Another branch derives per-projection malignancy supervision from the complete training masks and blends a learned projection score with the breast-level score. [Author classification script](https://github.com/batmanlab/Mammo-CLIP/blob/main/src/scripts/classifier_fine_tune_b5.sh) · [Released downstream checkpoints](https://huggingface.co/shawn24/Mammo-CLIP/tree/main/Downstream-checkpoints)

Nested blend comparison aggregates the highest-scoring small combinations across three seeded repeats of five patient-grouped folds. Each held-out validation prediction uses candidates chosen and calibrated on the other validation folds. The alternative convex method fits the positive–negative ranking objective on the same separated fit rows. A test run can lock the winner among these nested policies using their held-out AUROC.

## Reports and prediction

The local evaluation report includes ROC, precision–recall, subgroup summaries, paired patient-bootstrap intervals and examples with held-out reference labels. Public portfolio reports use aggregate benchmarks and synthetic input examples.

```powershell
& $p -m medimage.auc_report --output "$experiment/evaluation/mammo-locked-1" --previous $search
& $p -m medimage.auc_predict --selection "$experiment/evaluation/mammo-locked-1/selection.json" --work $experiment --previous $search --original artifacts --weights $hf --clip "$experiment/weights/mammo-clip-b5.tar" --cc path/to/full-cc.png --mlo path/to/full-mlo.png --output "$experiment/inference/mammography.json"
```
