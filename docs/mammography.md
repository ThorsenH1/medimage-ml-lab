# CBIS-DDSM mammography

The task estimates whether an annotated breast has at least one malignant lesion. It uses a CC image, an MLO image, or both. Foreground segmentation removes background and aligns the chest wall to the left. Projection masks flow through training and inference.

## Experiment

The global comparison uses one whole-image representation per available view. The multi-instance experiment uses the whole image and six overlapping crops per view. Gated attention combines patches, while cross-view interaction can combine CC and MLO. The endpoint is malignant if any recorded lesion in the breast is malignant.

Patient grouping spans the mass and calcification tables. The cohort drops 31 patients with conflicting official split assignments and 27 breasts whose projections could not be linked unambiguously. For 282 calcification test rows with missing series descriptions, exact study and series UIDs and full-image dimensions support the join. Repairing the mapping increased the test cohort from 221 to 336 breasts. Training and validation rows and pixels were verified to be identical across the repair.

The final cohort contains 1,039 training, 260 validation, and 336 test breasts. Model selection maximizes validation AUROC. The original recipes had different epoch budgets and showed overfitting after the first few epochs. The extended search evaluates four encoder/resolution profiles and fine-tunes the strongest new one from images.

## Limitations and learning objectives

This is a retrospective, lesion-enriched cohort of digitized film mammograms. The endpoint and prevalence follow that sampling scheme. The projection, mass, and calcification subgroups describe model behavior within this cohort. [TCIA collection and descriptor](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)

The project shows how endpoint definition, DICOM-to-JPEG matching, and patient grouping change a classification experiment.

## Inference

Pass full views of the same breast through `--cc` and/or `--mlo`. Use the matching selection file and checkpoints. The [project repository](https://github.com/ThorsenH1/cbis-ddsm-mammography-ml) includes the exact command and synthetic examples. [Initial results](results.md) · [Extended search](model-search.md)
