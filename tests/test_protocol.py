from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from medimage.chest_cohort import exact_pixel_groups, grouped_partitions
from medimage.common import assert_disjoint
from medimage.radiography_data import CHEST_LABELS
from medimage.radiography_evaluate import fit_calibration, patient_bootstrap
from medimage.spine_data import SIZE, project
from medimage.spine_localize import fit_localizer, roi_coordinates, world_from_pixels
from medimage.spine_metrics import check_probabilities, score, submission


def test_patient_split_guards() -> None:
    assert_disjoint(np.array([1, 1, 2]), np.array(['train', 'train', 'test']))
    with pytest.raises(ValueError):
        assert_disjoint(np.array([1, 1, 2]), np.array(['train', 'test', 'test']))


def test_exact_duplicate_pixels_group_patient_aliases_and_keep_official_test() -> None:
    total = 55
    frame = pd.DataFrame({'patient': [*range(54), 5], 'split': ['train'] * 53 + ['test', 'val']})
    for index, label in enumerate(CHEST_LABELS):
        frame[label] = (np.arange(total) % 3 == index % 3).astype(int)
    pixels = np.broadcast_to(np.arange(total, dtype=np.uint8)[:, None, None], (total, 16, 16)).copy()
    pixels[53] = pixels[5]
    pixels[3] = pixels[4]
    groups = exact_pixel_groups(pixels)
    partition, report = grouped_partitions(frame, groups)
    assert partition.loc[53, 'split'] == 'test'
    assert partition.loc[5, 'split'] == partition.loc[54, 'split'] == 'excluded'
    assert partition.loc[3, 'split'] == partition.loc[4, 'split']
    assert report['development_patients_excluded_for_test_alias'] == 1
    assert report['development_images_excluded_for_test_alias'] == 2


def test_patient_bootstrap_matches_identical_patient_per_replicate() -> None:
    truth = np.tile([0., 1., 1., 0.], 4)[:, None]
    prediction = np.tile([.3, .6, .4, .4], 4)[:, None]
    groups = np.repeat(np.arange(4), 4)
    report = patient_bootstrap(truth, prediction, groups, repeats=50)
    auc = roc_auc_score(truth, prediction)
    ap = average_precision_score(truth, prediction)
    np.testing.assert_allclose(report['macro_auroc_ci95'], [auc, auc])
    np.testing.assert_allclose(report['macro_auprc_ci95'], [ap, ap])


def test_calibration_is_monotone_and_finite() -> None:
    logits = np.linspace(-3, 3, 32)[:, None]
    labels = (logits > .6).astype(np.float64)
    fit, thresholds = fit_calibration(labels, logits)
    calibrated = expit(logits * fit[:, 0] + fit[:, 1])
    assert np.isfinite(calibrated).all() and np.all(np.diff(calibrated[:, 0]) > 0)
    assert .01 < thresholds[0] < .99


def test_geometry_roundtrip_and_nearest_axial_plane() -> None:
    planes = np.zeros((42, 3, 3), np.float32)
    valid = np.zeros(42, dtype=bool)
    valid[[18, 19, 20]] = True
    for slot, z in zip([18, 19, 20], [20., 16., 12.], strict=True):
        planes[slot] = np.array([[0, 0, z], [0, .6, 0], [-.5, 0, 0]])
    xy = np.array([[[40., 50.]]])
    point = world_from_pixels(xy, planes[19:20])
    estimated, residual = project(point, planes[19:20, None])
    np.testing.assert_allclose(estimated, xy)
    np.testing.assert_allclose(residual, 0, atol=2e-6)
    coordinates, metadata = roi_coordinates(np.tile(point[0], (30, 1)), np.ones(5), planes, valid)
    np.testing.assert_array_equal(coordinates[15, :, 0], [18, 19, 20])
    assert (coordinates[:15, :, 0] == -1).all()
    assert (metadata[15:, 0] == 1).all()


def test_localizer_fit_ignores_every_held_out_label() -> None:
    rng = np.random.default_rng(123)
    cohort = 30
    anchors = np.broadcast_to([[3., 8., 5.], [0., 1., 0.], [0., 0., 1.]], (cohort, 3, 3))
    pixel = rng.uniform(5, SIZE - 5, (cohort, 5, 2))
    points = np.tile(world_from_pixels(pixel, anchors), (1, 5, 1))
    kernel_points = rng.normal(size=(cohort, 4))
    squared = ((kernel_points[:, None] - kernel_points[None, :]) ** 2).sum(axis=-1)
    train = np.arange(23)
    first = fit_localizer(train, squared, anchors, points)
    points[23:] = -1024
    second = fit_localizer(train, squared, anchors, points)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_metric_masks_missing_and_weights_any_severe() -> None:
    labels = np.zeros((2, 25), dtype=int)
    labels[0, 0], labels[1, 0] = 2, -1
    probability = np.broadcast_to([.60, .25, .15], (2, 25, 3)).copy()
    probability[1, 0] = [.01, .01, .98]
    details = score(labels, probability)
    any_severe = -(4 * np.log(.15) + np.log(.85)) / 5
    spinal = -(4 * np.log(.15) + 8 * np.log(.60)) / 12
    assert details['any_severe_spinal'] == pytest.approx(any_severe)
    assert details['spinal'] == pytest.approx(spinal)
    assert len(submission(np.array([120, 264]), probability)) == 50
    probability[0, 0, 1] = -5
    with pytest.raises(ValueError):
        check_probabilities(probability)
