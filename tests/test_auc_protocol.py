from pathlib import Path

import cv2
import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from medimage.auc_mammo_data import Geometry
from medimage.auc_selection import cross_fit_calibration, label_weights, read_scores
from medimage.auc_train import ResidentPixels
from medimage.radiography_evaluate import patient_bootstrap


def test_patient_bootstrap_accepts_integer_benchmark_labels():
    labels = np.array([[0], [1], [0], [1]], dtype=np.int8)
    scores = np.array([[.1], [.8], [.2], [.5]])
    baseline = np.array([[.2], [.7], [.3], [.4]])
    result = patient_bootstrap(labels, scores, np.arange(4), repeats=40, reference=baseline)
    assert result['macro_auroc_ci95'] == [1., 1.]
    assert result['paired_auroc_delta_ci95'] == [0., 0.]


def test_mammography_mask_uses_the_same_crop_flip_and_canvas():
    source = np.arange(32, dtype=np.uint8).reshape(4, 8)
    geometry = Geometry((4, 8), (1, 3, 2, 6), True, (2, 4), canvas=(4, 8))
    transformed = geometry.transform(source, cv2.INTER_NEAREST)
    np.testing.assert_equal(transformed[:2, :4], source[1:3, 2:6][:, ::-1])
    assert not transformed[2:].any()
    assert not transformed[:, 4:].any()


def test_resident_store_has_no_held_out_rows():
    values = np.arange(24, dtype=np.uint8).reshape(6, 2, 2)
    resident = ResidentPixels(values, np.array([0, 2, 5]))
    np.testing.assert_equal(resident[np.array([5, 0])], values[[5, 0]])
    with pytest.raises(ValueError, match='outside the resident'):
        resident[np.array([0, 1])]


def test_validation_reader_rejects_reordered_predictions(tmp_path: Path):
    file = tmp_path / 'scores.npz'
    np.savez_compressed(file, rows=np.array([1, 3]), logits=np.array([[.1], [.2]]))
    with pytest.raises(ValueError, match='misaligned'):
        read_scores(file, np.array([3, 1]), num_labels=1)


def test_grouped_calibration_is_independent_of_its_held_out_labels():
    truth = np.tile(np.asarray([[0, 1], [1, 0], [0, 0], [1, 1]]), (12, 1))
    group = np.repeat(np.arange(12), 4)
    rng = np.random.default_rng(359)
    scores = {'a': truth + rng.normal(0, .6, truth.shape), 'b': rng.normal(size=truth.shape)}
    folds = list(GroupKFold(5).split(truth, groups=group))
    _, check = folds[0]
    changed = truth.copy()
    changed[check] = 1 - changed[check]
    baseline = cross_fit_calibration(truth, scores, folds)
    altered = cross_fit_calibration(changed, scores, folds)
    for key in scores:
        np.testing.assert_array_equal(baseline[key][check], altered[key][check])
    weights = label_weights(truth, scores, {'a': .5, 'b': .5}, 24)
    np.testing.assert_allclose(np.sum(list(weights.values()), axis=0), 1)
    assert np.all(np.asarray(weights['a']) > .5)
