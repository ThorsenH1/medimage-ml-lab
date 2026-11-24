# NIH ChestX-ray14

The task produces 14 binary scores per frontal chest radiograph. The original attention model uses label-specific queries over a 5 × 5 feature grid and models interactions between labels. The global comparison shares its ResNet-18 backbone. Their epoch budgets and OneCycle schedules differ, so the results compare complete training recipes.

## Cohort and validation

All 112,120 images in the source lists were identified. An audit found five groups of pixel-identical originals across the first partition under different patient IDs. The final split groups these aliases together. Four development patient IDs covering 29 images were dropped because their groups occurred in the official test set. The complete official test list was retained. The absence of cross-partition pixel duplicates was also verified at 224 pixels.

The final cohort has 76,084 training images from 24,643 patient IDs and 10,411 validation images from 3,361 other patient IDs. Test has 25,596 images from 2,797 patient IDs in 2,795 exact-image-linked clusters. The validation set contains 22 positive hernia images.

The selection criterion averages macro AUROC and macro average precision. Each label receives logistic calibration fitted on validation. Evaluation reports ranking, calibration, acquisition-view and demographic subgroups, and patient-cluster bootstrap intervals. Validation-derived F1 thresholds describe benchmark behavior.

## Limitations and learning objectives

Labels were extracted from reports. Acquisition view, equipment, and correlated findings can affect ranking. AP/PA, age, and sex analyses describe subgroup behavior on this dataset. The project illustrates how precision, prevalence, and repeated observations per patient affect interpretation. [NIH dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) · [RSNA data card](https://atlas.rsna.org/cards/66962171-1dcb-431d-89b0-08a561f1c8ed)

## Inference

Use the locked model plan and its checkpoint roots. The [shared README](../README.md) shows the prediction command. The [project repository](https://github.com/ThorsenH1/chest-xray14-ml) includes a synthetic example. [Initial results](results.md) · [Extended search](model-search.md)
