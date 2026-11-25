from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from medimage.spine_metrics import GROUPS

ROI_SIZE = 96


def crop_regions(volume: np.ndarray, coords: np.ndarray, jitter: np.ndarray | None = None,
                 size: int = ROI_SIZE) -> np.ndarray:
    if coords.shape != (30, 3, 5):
        raise ValueError('Expected 30 three-slice MRI regions')
    pixels = np.zeros((30, 3, size, size), dtype=np.uint8)
    for index, regions in enumerate(coords):
        delta = (0, 0) if jitter is None else jitter[index]
        for channel, (slot, x, y, width, height) in enumerate(regions):
            if slot < 0:
                continue
            sx, sy = width / size, height / size
            ox = x + delta[0] * width - (size - 1) * sx / 2
            oy = y + delta[1] * height - (size - 1) * sy / 2
            matrix = np.array([[sx, 0, ox], [0, sy, oy]], dtype=np.float32)
            pixels[index, channel] = cv2.warpAffine(volume[int(slot)], matrix, (size, size),
                                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return pixels


class SpineClassifier(nn.Module):
    def __init__(self, pretrained: bool, spatial_detail: bool = False) -> None:
        super().__init__()
        convnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder = nn.Sequential(*list(convnet.children())[:-2]).to(memory_format=torch.channels_last)
        dimension = 160
        self.map = nn.Sequential(nn.Linear(512, dimension), nn.LayerNorm(dimension), nn.GELU(), nn.Dropout(.1))
        self.detail = None
        if spatial_detail:
            self.detail = nn.Sequential(nn.Conv2d(512, 64, 1, bias=False), nn.GroupNorm(8, 64), nn.GELU(),
                                        nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(256, dimension))
            nn.init.zeros_(self.detail[-1].weight)
            nn.init.zeros_(self.detail[-1].bias)
        self.condition = nn.Embedding(6, dimension)
        self.level = nn.Embedding(5, dimension)
        self.quality = nn.Sequential(nn.Linear(5, dimension), nn.Tanh())
        self.register_buffer('condition_ids', torch.arange(6).repeat_interleave(5))
        self.register_buffer('level_ids', torch.arange(5).repeat(6))
        self.self_attention = nn.MultiheadAttention(dimension, 4, dropout=.10, batch_first=True)
        self.pre_attention = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 2 * dimension),
                                 nn.GELU(), nn.Dropout(.15), nn.Linear(2 * dimension, dimension))
        self.readout = nn.Sequential(nn.LayerNorm(dimension), nn.Dropout(.18), nn.Linear(dimension, 3))

    def forward(self, regions: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        batch, regions_count = regions.shape[:2]
        present = metadata[:, :, 0:1]
        x = (regions.flatten(0, 1).float() / 255. - .449) / .226
        if self.training:
            x = x * torch.empty_like(x[:, :1, :1, :1]).uniform_(.9, 1.1)
        encoded = self.encoder(x.contiguous(memory_format=torch.channels_last))
        visual = self.map(encoded.mean(dim=(2, 3)))
        if self.detail is not None:
            visual = visual + self.detail(encoded)
        image = visual.reshape(batch, regions_count, -1) * present
        token = image + self.condition(self.condition_ids) + self.level(self.level_ids) + self.quality(metadata)
        normalized = self.pre_attention(token)
        interaction, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        token = token + interaction
        token = token + self.mlp(token)
        return self.readout(token[:, :25])


def severity_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    scores = logits.float()
    observed = labels >= 0
    positions = labels.clamp_min(0)
    weight = torch.tensor([1., 2., 4.], device=logits.device)[positions] * observed
    nll = F.cross_entropy(scores.transpose(1, 2), positions, reduction='none')
    probabilities = scores.softmax(dim=-1)
    true_cdf = F.one_hot(positions, num_classes=3).cumsum(dim=-1)[..., :2]
    predicted_cdf = probabilities.cumsum(dim=-1)[..., :2]
    ordinal = (true_cdf - predicted_cdf).square().mean(dim=-1)
    local_loss = (nll + .10 * ordinal) * weight
    losses = [(local_loss[:, group].sum() / weight[:, group].sum().clamp_min(1)) for group in GROUPS]
    spinal_p = (probabilities[:, :5, 2] * observed[:, :5]).max(dim=1).values.clamp(1e-6, 1 - 1e-6)
    spinal_y = (labels[:, :5] == 2).any(dim=1).float()
    spinal_w = weight[:, :5].max(dim=1).values
    any_severe = F.binary_cross_entropy(spinal_p, spinal_y, weight=spinal_w, reduction='sum') / spinal_w.sum().clamp_min(1)
    return (sum(losses) + any_severe) / 4


def lateral_ranking_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Rank each labeled grade above other grades at the same lateral target."""
    scores = logits[:, 5:].float().log_softmax(dim=-1)
    actual = labels[:, 5:]
    known = actual >= 0
    positive = actual[..., None] == torch.arange(3, device=actual.device)
    negative = known[..., None] & ~positive
    compared = positive[:, None] & negative[None]
    difference = scores[:, None] - scores[None]
    count = compared.sum(dim=(0, 1))
    by_target = (F.softplus(-difference) * compared).sum(dim=(0, 1)) / count.clamp_min(1)
    return by_target.sum() / (count > 0).sum().clamp_min(1)
