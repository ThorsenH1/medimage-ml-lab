# Extended model and hyperparameter search

The search started 156 trial configurations across tasks and folds and completed 234 rung evaluations. Each X-ray profile was refitted with two seeds. The strongest new profile for each X-ray task also had its last layers fine-tuned from images. The initial chest X-ray model was also fine-tuned at 224 pixels. MRI finalists were refitted with two seeds per outer fold.

## Results

| Task | Selected model | Previous score | New score | 95% paired delta CI |
| --- | --- | ---: | ---: | ---: |
| Chest X-ray | `incumbent+adapted-incumbent-224-0.50` | 0.7787 AUROC | 0.7900 AUROC | [0.0098, 0.0130] |
| Mammography | `incumbent+adapted-dino-224-0.25` | 0.6858 AUROC | 0.7468 AUROC | [0.0298, 0.0988] |
| Lumbar MRI | `incumbent+raw-0.15` | 0.5307 log loss | 0.5260 log loss | [-0.0067, -0.0024] |

Higher AUROC and lower log loss are better. Chest X-ray uses macro averages over 14 findings. The mammography cohort is lesion-enriched. Intervals estimate differences from the initial scores with patient- or study-cluster bootstrap.

### Radiography

| Task | Test units | Average precision before → after | Brier before → after |
| --- | ---: | ---: | ---: |
| Chest X-ray | 25,596 | 0.2386 → 0.2487 | 0.0592 → 0.0587 |
| Mammography | 336 | 0.6372 → 0.7073 | 0.2180 → 0.1998 |

### Validation before the benchmark lock

| Task | Candidate | Validation objective | AUROC | AP |
| --- | --- | ---: | ---: | ---: |
| Chest X-ray | `adapted-incumbent-224` | 0.51940 | 0.8234 | 0.2154 |
| Chest X-ray | `incumbent-global` | 0.51545 | 0.8151 | 0.2158 |
| Chest X-ray | `effb0-224` | 0.46737 | 0.7793 | 0.1554 |
| Chest X-ray | `dino-224` | 0.46651 | 0.7761 | 0.1569 |
| Chest X-ray | `adapted-effb0-224` | 0.46599 | 0.7736 | 0.1583 |
| Chest X-ray | `effb0-160` | 0.45449 | 0.7676 | 0.1414 |
| Chest X-ray | `r50-224` | 0.45156 | 0.7619 | 0.1412 |
| Chest X-ray | `incumbent+adapted-incumbent-224-0.50` | 0.52710 | 0.8271 | 0.2271 |
| Mammography | `incumbent-global` | 0.73830 | 0.7383 | 0.7422 |
| Mammography | `adapted-dino-224` | 0.72216 | 0.7222 | 0.6781 |
| Mammography | `dino-224` | 0.71586 | 0.7159 | 0.6585 |
| Mammography | `effb0-224` | 0.70671 | 0.7067 | 0.6517 |
| Mammography | `effb0-160` | 0.70552 | 0.7055 | 0.6573 |
| Mammography | `r50-224` | 0.70498 | 0.7050 | 0.6702 |
| Mammography | `incumbent+adapted-dino-224-0.25` | 0.77832 | 0.7783 | 0.7443 |

### Chosen hyperparameters per radiography profile

| Task | Profile | Head | Width | LR | Weight decay | Dropout | Positive-weight exponent | Patch dropout |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chest X-ray | `effb0-160` | attention | 224 | 0.0014 | 0.012 | 0.45 | 0.5 | 0.16 |
| Chest X-ray | `r50-224` | attention | 128 | 0.001 | 0.007 | 0.3 | 0.5 | 0.12 |
| Chest X-ray | `effb0-224` | attention | 128 | 0.001 | 0.007 | 0.3 | 0.5 | 0.12 |
| Chest X-ray | `dino-224` | mlp | 256 | 0.0015 | 0.008 | 0.35 | 0.5 | 0.08 |
| Mammography | `effb0-160` | linear | 0 | 0.003 | 0.015 | 0.12 | 0.5 | 0.0 |
| Mammography | `r50-224` | attention | 160 | 0.00048 | 0.003 | 0.35 | 0.25 | 0.0 |
| Mammography | `effb0-224` | linear | 0 | 0.003 | 0.015 | 0.12 | 0.5 | 0.0 |
| Mammography | `dino-224` | mlp | 256 | 0.00035 | 0.0008 | 0.33 | 0.25 | 0.0 |

### Lumbar MRI

| MRI candidate | Weighted log loss | Folds improved | 95% paired delta CI |
| --- | ---: | ---: | ---: |
| `incumbent+raw-0.15` | 0.52605 | 5 / 5 | [-0.0067, -0.0024] |
| `incumbent+cal-0.15` | 0.52640 | 5 / 5 | [-0.0064, -0.0021] |
| `incumbent+raw-0.30` | 0.52773 | 3 / 5 | [-0.0066, 0.0009] |
| `incumbent+cal-0.30` | 0.52835 | 3 / 5 | [-0.0059, 0.0015] |
| `incumbent` | 0.53069 | 0 / 5 | [0.0000, 0.0000] |
| `incumbent+raw-0.50` | 0.53766 | 1 / 5 | [0.0012, 0.0131] |
| `incumbent+cal-0.50` | 0.53838 | 1 / 5 | [0.0021, 0.0137] |
| `incumbent+raw-0.70` | 0.55672 | 1 / 5 | [0.0179, 0.0348] |
| `incumbent+cal-0.70` | 0.55684 | 1 / 5 | [0.0182, 0.0348] |
| `feature_calibrated` | 0.60747 | 0 / 5 | [0.0637, 0.0909] |
| `feature_uncalibrated` | 0.61397 | 0 / 5 | [0.0690, 0.0989] |

| Outer fold | Inner validation fold | Selected head | Descriptor | Selected epochs | After inner temperature scaling |
| ---: | ---: | --- | --- | ---: | ---: |
| 0 | 1 | attention | hybrid | 6 | 0.62708 |
| 1 | 2 | attention | hybrid | 4 | 0.58704 |
| 2 | 3 | attention | hybrid | 6 | 0.57245 |
| 3 | 4 | conv | hybrid | 7 | 0.65564 |
| 4 | 0 | attention | hybrid | 8 | 0.59588 |

Raw DICOM inference produced 25 valid submission rows from 1 local test study.

## Protocol

- Chest X-ray and mammography reuse the official benchmark cohorts from the initial study. The search ranks configurations on an inner patient-grouped part of training. Finalists train on the whole training set and compete against the incumbent and prespecified blends on the original validation set. Chest X-ray maximizes the mean of macro AUROC and AP. Mammography maximizes AUROC.
- The radiography search covers EfficientNet-B0 at 160 and 224 pixels, ResNet-50 at 224, and DINOv2 ViT-S at 224. Each profile has 12 combinations of head, learning rate, dropout, width, weight decay, and positive weighting. Successive halving advances 12 to 4 to 2 candidates with increasing epoch budgets. [Successive halving](https://proceedings.mlr.press/v51/jamieson16.html) · [DINOv2](https://arxiv.org/abs/2304.07193)
- Mammography uses one whole image and six overlapping tiles per available view. Radiography blends combine logits after class-weight correction. MRI blends combine probabilities.
- MRI uses five outer folds and a separate inner holdout in each. Outer and inner validation data are excluded from localizer fitting for inner training images. Local, convolutional, and attention heads use deep and texture descriptors with varied ordinal penalties and severe-stenosis weights. MRI blends must win on at least three folds and have a paired interval with upper bound below zero to replace the incumbent.
- MRI and chest X-ray select the refit epoch count on inner validation. The mammography search ranks configurations at each rung endpoint.
- The X-ray test cohorts were used in the first investigation and are reused benchmarks. This round locks its chosen models, blend weights, and calibration before scoring. The local MRI test has one study without available grades. Performance is assessed on the frozen folds. The local study verifies the inference format.

Experiment checkpoints, detailed timings, and individual predictions are stored in the local search directory. The repository contains training code and aggregate results.
