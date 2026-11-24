# Study guide

Use the [initial results](results.md), [extended search report](model-search.md), and [research sources](research.md) to prepare a project presentation.

| Topic | Example | Question |
| --- | --- | --- |
| Image geometry | MRI crops use a physical width and a local DICOM basis | How does anisotropic spacing change a square crop defined in pixels? |
| Sampling | CBIS-DDSM includes breasts with annotated lesions | How would precision and calibration change at screening prevalence? |
| Leakage | Patients contribute multiple images and a localizer can learn a study's coordinates | Which datasets must be held out when one trained stage supplies inputs to the next? |
| Class imbalance | Chest X-ray evaluation reports AUROC, average precision, and prevalence | Which conditions rank reasonably well while absolute precision stays low? |
| Model selection | The search varies architecture, learning rate, regularization, and training duration | What evidence justifies promoting a more complex model? |
| Uncertainty | Bootstrap samples preserve whole patient clusters and pair MRI predictions | How could sampling individual images change the interval? |

## Practical analyses

1. Describe how each cohort was assembled and how grouping affects the split. Use aggregate counts in the presentation.
2. Trace each pipeline from raw image to probabilities. Identify fitted parameters and fixed preprocessing rules.
3. Choose a result that challenged your hypothesis. Explain what was measured on validation before the benchmark.
4. Discuss the population represented by each benchmark and the data needed to estimate transfer to a different institution.
