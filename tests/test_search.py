from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from medimage.radiography_data import CHEST_LABELS
from medimage.radiography_evaluate import patient_bootstrap
from medimage.search_adapt import HighResolutionReference
from medimage.search_evaluate import candidate_plan
from medimage.search_heads import FeatureClassifier, RECIPES, inner_split
from medimage.search_spine import RECIPES as SPINE_RECIPES, SpineFeatureHead, loss_for


def test_inner_search_never_uses_original_validation_or_test() -> None:
    group = np.repeat(np.arange(70), 2)
    frame = pd.DataFrame({'pixel_cluster': group, 'split': np.where(group < 50, 'train', np.where(group < 60, 'val', 'test'))})
    for index, label in enumerate(CHEST_LABELS):
        frame[label] = (group % (index % 3 + 2) == 0).astype(int)
    train, valid = inner_split(frame, 'chest', 3803)
    assert len(train) + len(valid) == 100
    assert frame.iloc[train].split.eq('train').all()
    assert frame.iloc[valid].split.eq('train').all()
    assert set(frame.iloc[train].pixel_cluster).isdisjoint(frame.iloc[valid].pixel_cluster)


@pytest.mark.parametrize('index', [0, 2, 8])
def test_missing_mammography_tiles_do_not_influence_logits(index) -> None:
    torch.manual_seed(4)
    model = FeatureClassifier(32, 14, 1, RECIPES[index]).eval()
    features = torch.randn(2, 14, 32)
    mask = torch.tensor([[1] * 7 + [0] * 7, [0] * 7 + [1] * 7], dtype=torch.bool)
    updated = torch.where(mask.unsqueeze(-1), features, torch.randn_like(features) * 100)
    with torch.inference_mode():
        torch.testing.assert_close(model(features, mask), model(updated, mask))


@pytest.mark.parametrize('index', [1, 5, 9])
def test_spine_search_head_forward_and_masked_objective(index) -> None:
    torch.set_num_threads(2)
    recipe = SPINE_RECIPES[index]
    module = SpineFeatureHead(recipe)
    result = module(torch.randn(3, 30, 832), torch.ones(3, 30, 5))
    assert result.shape == (3, 25, 3)
    result.retain_grad()
    labels = torch.randint(0, 3, (3, 25))
    labels[0, 15] = -1
    loss_for(result, labels, recipe).backward()
    assert torch.isfinite(result.grad).all()
    assert torch.all(result.grad[0, 15] == 0)


def test_candidate_selection_is_deterministic_and_paired_bootstrap_is_zero_for_identical_predictions() -> None:
    truth = np.tile([0., 1., 1., 0.], 12)[:, None]
    score = np.tile([.3, .6, .4, .4], 12)[:, None]
    groups = np.repeat(np.arange(12), 4)
    candidates = candidate_plan({'incumbent-global': score, 'strong': score}, truth, 'mammo')
    assert candidates[0]['name'] == 'incumbent-global'
    result = patient_bootstrap(truth, score, groups, repeats=20, reference=score)
    assert result['paired_auroc_delta_ci95'] == [0., 0.]
    assert result['paired_auprc_delta_ci95'] == [0., 0.]


def test_high_resolution_reference_accepts_larger_input_with_trainable_final_block() -> None:
    torch.set_num_threads(2)
    model = HighResolutionReference().train()
    assert not model.encoder.training
    assert not next(model.encoder[0].parameters()).requires_grad
    assert all(weight.requires_grad for weight in model.encoder[-1].parameters())
    pixels = torch.randint(0, 255, (2, 224, 224), dtype=torch.uint8)
    with torch.inference_mode():
        output = model(pixels, torch.ones(2, 5, dtype=torch.bool))
    assert output.shape == (2, 14) and torch.isfinite(output).all()
