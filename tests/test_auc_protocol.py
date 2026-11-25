import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from sklearn.model_selection import GroupKFold

from medimage.auc_mammo_data import Geometry
from medimage.auc_mammo_mil import BreastMIL
from medimage.auc_mammo_texture import TEXTURE_DIMENSIONS, breast_features
from medimage.auc_predict import prediction_trace
from medimage.auc_selection import (
    cross_fit_calibration, fit_pairwise_differences, fit_pairwise_weights, label_weights, read_scores, stable_candidate,
)
from medimage.auc_train import ResidentPixels
from medimage.mammo_rank_balance import ranking_pairs
from medimage.radiography_evaluate import patient_bootstrap
from medimage.spine_native import reproject_regions
from medimage.visual_explain import patch_grid, soften


def test_patient_bootstrap_accepts_integer_benchmark_labels():
    labels = np.array([[0], [1], [0], [1]], dtype=np.int8)
    scores = np.array([[.1], [.8], [.2], [.5]])
    baseline = np.array([[.2], [.7], [.3], [.4]])
    result = patient_bootstrap(labels, scores, np.arange(4), repeats=40, reference=baseline)
    assert result['macro_auroc_ci95'] == [1., 1.]
    assert result['paired_auroc_delta_ci95'] == [0., 0.]


def test_pairwise_fusion_prefers_the_better_ranker_on_its_fit_rows():
    labels = np.array([[0], [1], [0], [1]], dtype=np.int8)
    model_logits = np.array([[-2, 2], [2, -2], [-1, 1], [1, -1]], dtype=float)
    weights = fit_pairwise_weights(labels, model_logits, strength=.1)
    assert weights[0] > weights[1]
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(), 1.)


def test_lesion_balanced_ranking_retains_global_pairs_and_equalizes_added_mass():
    truth = np.array([1, 1, 1, 1, 0, 0, 0], dtype=bool)
    types = np.array(['calc', 'calc', 'calc', 'mass', 'calc', 'calc', 'mass'])
    logits = np.array([[1, -1], [2, 0], [0, 1], [2, -2], [0, 0], [0, 1], [1, 0]], dtype=float)
    pairs, uniform = ranking_pairs(logits, truth, types, 0.)
    same_pairs, balanced = ranking_pairs(logits, truth, types, .5)
    np.testing.assert_array_equal(pairs, same_pairs)
    np.testing.assert_allclose(balanced.sum(), 1.)
    np.testing.assert_allclose((balanced - .5 * uniform).reshape(4, 3)[:3, :2].sum(), .25)
    np.testing.assert_allclose((balanced - .5 * uniform).reshape(4, 3)[3, 2], .25)
    np.testing.assert_allclose(fit_pairwise_differences(pairs, .12, pair_weights=uniform),
                               fit_pairwise_weights(truth[:, None], logits, .12), atol=1e-10)


def test_group_bagging_produces_normalized_model_weights():
    labels = np.tile([[0], [1]], (20, 1))
    rng = np.random.default_rng(602)
    scores = {name: labels + rng.normal(0, .45, labels.shape) for name in ('a', 'b', 'c')}
    candidate = stable_candidate(labels, scores, np.arange(len(labels)), {key: key for key in scores})
    assert candidate['selection_folds'] == 15
    assert candidate['metrics']['macro_auroc'] > .7
    np.testing.assert_allclose(sum(candidate['weights'].values()), 1.)


@pytest.mark.parametrize('contextual', [False, True])
def test_breast_attention_ignores_unavailable_projections(contextual):
    model = BreastMIL(24, .2, contextual=contextual).eval()
    rng = torch.Generator().manual_seed(64)
    bags = torch.randn(2, 2, 13, 520, generator=rng)
    mask = torch.zeros(2, 2, 13, dtype=torch.bool)
    mask[0, 0, :4] = True
    mask[1, 1, :5] = True
    with torch.inference_mode():
        baseline = model(bags, mask)
        perturbed = model(bags.masked_fill(~mask[..., None], 10.), mask)
    assert torch.isfinite(baseline).all()
    torch.testing.assert_close(baseline, perturbed)


def test_prediction_trace_respects_per_finding_weights_and_calibration():
    lock = {'task': 'chest', 'labels': ['first', 'second'],
            'candidates': {'a': {'image_shape': [48, 48]}, 'b': {}},
            'selected': {'method': 'calibrated-label-logit', 'weights': {'a': .5, 'b': .5},
                         'label_weights': {'a': [.25, .9], 'b': [.75, .1]}},
            'component_calibration': {'a': [[2, .5], [1, -.5]], 'b': [[1, 0], [.5, 0]]},
            'ensemble_calibration': [[1, 0], [1, 0]], 'thresholds': [.3, .6]}
    logits = {'a': np.array([[1., 2.]]), 'b': np.array([[0., -2.]])}
    trace = prediction_trace(lock, logits, np.array([.65, .78]), ['frontal'], 'hash')
    np.testing.assert_allclose(trace['components']['a']['weighted_logit'], [.625, 1.35])
    np.testing.assert_allclose(trace['components']['b']['weighted_logit'], [0., -.1])
    np.testing.assert_allclose(trace['fused_logit'], [.625, 1.25])


def test_blur_probe_changes_only_its_reported_window():
    image = np.full((96, 100), 80, dtype=np.uint8)
    image[46:51, 48:53] = 230
    boxes = patch_grid(image, 'chest')
    # Coordinates also appear verbatim in public JSON.
    json.dumps(boxes)
    box = next((x0, y0, x1, y1) for x0, y0, x1, y1 in boxes if x0 <= 49 < x1 and y0 <= 48 < y1)
    changed = soften(image, [box])[0]
    x0, y0, x1, y1 = box
    allowed = np.zeros(image.shape, dtype=bool)
    allowed[y0:y1, x0:x1] = True
    assert (image[allowed] != changed[allowed]).any()
    np.testing.assert_array_equal(image[~allowed], changed[~allowed])


def test_spine_grid_lift_preserves_physical_crop():
    source = np.array([[[[1., 2., 0.], [2., 0., 0.], [0., 3., 0.]]]])
    dest = np.array([[[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]]])
    region = np.array([[[[0., 4., 5., 8., 12.], [-1., 0., 0., 0., 0.]]]])
    lifted = reproject_regions(region, source, dest)
    np.testing.assert_allclose(lifted[0, 0, 0], [0., 9., 17., 16., 36.])
    np.testing.assert_array_equal(lifted[0, 0, 1], region[0, 0, 1])


def test_texture_bank_marks_the_missing_view_and_responds_to_bright_spots():
    initial = np.full((96, 64), 65, dtype=np.uint8)
    study = {'pixels': np.stack([initial, initial])[None], 'views': np.array([[True, False]])}
    before = breast_features(study, [0], workers=1)
    study['pixels'][0, 0, 19:21, 22:24] = 245
    after = breast_features(study, [0], workers=1)
    assert before.shape == after.shape == (1, TEXTURE_DIMENSIONS * 3 + 2)
    np.testing.assert_array_equal(after[0, -2:], [1., 0.])
    assert not np.array_equal(before[0, :TEXTURE_DIMENSIONS], after[0, :TEXTURE_DIMENSIONS])


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
