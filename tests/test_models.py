from __future__ import annotations

import numpy as np
import pytest
import torch

from medimage.radiography_models import build_model
from medimage.spine_model import ROI_SIZE, SpineClassifier, crop_regions, severity_loss


@pytest.mark.parametrize(('task', 'variant'), [('chest', 'global'), ('chest', 'attention'),
                                             ('mammo', 'global'), ('mammo', 'attention')])
def test_radiography_model_eval_handles_single_projection(task, variant) -> None:
    torch.set_num_threads(2)
    model = build_model(task, variant, pretrained=False).eval()
    shapes = {'chest': (1, 160, 160), 'mammo': (1, 2, 512, 320)}
    images = torch.zeros(shapes[task], dtype=torch.uint8)
    views = torch.tensor([[False, True]]) if task == 'mammo' else torch.tensor([[True]])
    with torch.inference_mode():
        first = model(images, views)
        second = model(images, views)
    assert first.shape == (1, 1 if task == 'mammo' else 14)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_roi_cropping_keeps_center_and_masks_missing() -> None:
    volume = np.zeros((42, 160, 160), dtype=np.uint8)
    volume[5, 40:90, 50:100] = 233
    coordinates = np.zeros((30, 3, 5), dtype=np.float32)
    coordinates[:, :, 0] = -1
    coordinates[3, 1] = [5, 74, 64, 42, 42]
    crops = crop_regions(volume, coordinates)
    assert crops[3, 1, ROI_SIZE // 2, ROI_SIZE // 2] == 233
    assert crops[3, 0].max() == 0
    assert crops[4].max() == 0


def test_spine_ordinal_loss_masks_missing_targets() -> None:
    logits = torch.zeros(2, 25, 3, requires_grad=True)
    labels = torch.zeros((2, 25), dtype=torch.long)
    labels[0, 12] = -1
    loss = severity_loss(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.all(logits.grad[0, 12] == 0)
    assert torch.any(logits.grad[1, 12] != 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a CUDA GPU')
def test_spine_amp_forward_backward() -> None:
    torch.set_num_threads(2)
    model = SpineClassifier(pretrained=False).cuda()
    images = torch.randint(0, 256, (2, 30, 3, ROI_SIZE, ROI_SIZE), device='cuda', dtype=torch.uint8)
    information = torch.ones(2, 30, 5, device='cuda')
    labels = torch.randint(0, 3, (2, 25), device='cuda')
    with torch.autocast('cuda', dtype=torch.float16):
        z = model(images, information)
    loss = severity_loss(z, labels)
    assert z.shape == (2, 25, 3) and torch.isfinite(loss)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    scaler.scale(loss).backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
