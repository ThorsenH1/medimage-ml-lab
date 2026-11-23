from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def gray_encoder(pretrained: bool) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    rgb_conv = model.conv1
    model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    if pretrained:
        with torch.no_grad():
            model.conv1.weight.copy_(rgb_conv.weight.sum(dim=1, keepdim=True))
    return nn.Sequential(*list(model.children())[:-2]).to(memory_format=torch.channels_last)


def normalize(images: torch.Tensor) -> torch.Tensor:
    return (images.float().div(255) - .449) / .226


class ChestClassifier(nn.Module):
    def __init__(self, pretrained: bool, variant: str, labels: int = 14) -> None:
        super().__init__()
        self.encoder = gray_encoder(pretrained)
        self.global_head = nn.Sequential(nn.Dropout(.2), nn.Linear(512, labels))
        self.variant = variant
        if variant == 'attention':
            dimension = 128
            self.project = nn.Linear(512, dimension)
            self.query = nn.Parameter(torch.randn(labels, dimension) * .04)
            self.position = nn.Parameter(torch.randn(25, dimension) * .02)
            self.xattn = nn.MultiheadAttention(dimension, 4, dropout=.1, batch_first=True)
            self.interaction = nn.MultiheadAttention(dimension, 4, dropout=.1, batch_first=True)
            self.context_norm = nn.LayerNorm(dimension)
            self.condition_norm = nn.LayerNorm(dimension)
            self.class_readout = nn.Parameter(torch.randn(labels, dimension) * .02)

    def forward(self, images: torch.Tensor, views: torch.Tensor) -> torch.Tensor:
        x = normalize(images.unsqueeze(1)).contiguous(memory_format=torch.channels_last)
        if self.training:
            flip = torch.rand(len(x), 1, 1, 1, device=x.device) < .5
            x = torch.where(flip, x.flip((-1,)), x)
            x = x * torch.empty_like(x[:, :, :1, :1]).uniform_(.85, 1.15)
            x = x + torch.empty_like(x[:, :, :1, :1]).uniform_(-.15, .15)
        feature_map = self.encoder(x)
        logits = self.global_head(feature_map.mean(dim=(2, 3)))
        if self.variant == 'attention':
            tokens = F.adaptive_avg_pool2d(feature_map, (5, 5)).flatten(2).transpose(1, 2)
            tokens = self.project(tokens) + self.position
            queries = self.query.unsqueeze(0).expand(len(x), -1, -1)
            attended, _ = self.xattn(queries, tokens, tokens, need_weights=False)
            attended = self.context_norm(queries + attended)
            interactive, _ = self.interaction(attended, attended, attended, need_weights=False)
            context = self.condition_norm(attended + interactive)
            logits = logits + (context * self.class_readout).sum(dim=-1)
        return logits


class BreastClassifier(nn.Module):
    def __init__(self, pretrained: bool, variant: str) -> None:
        super().__init__()
        self.encoder = gray_encoder(pretrained)
        self.variant = variant
        self.project = nn.Sequential(nn.Linear(512, 160), nn.LayerNorm(160), nn.GELU(), nn.Dropout(.15))
        self.view_type = nn.Parameter(torch.randn(2, 160) * .02)
        if variant == 'attention':
            self.region = nn.Parameter(torch.randn(7, 160) * .02)
            self.tanh = nn.Linear(160, 96)
            self.sigmoid = nn.Linear(160, 96)
            self.patch_score = nn.Linear(96, 1, bias=False)
            self.view_attn = nn.MultiheadAttention(160, 4, dropout=.15, batch_first=True)
            self.view_norm = nn.LayerNorm(160)
        self.head = nn.Sequential(nn.LayerNorm(322), nn.Dropout(.30), nn.Linear(322, 128),
                                  nn.GELU(), nn.Dropout(.25), nn.Linear(128, 1))

    def forward(self, images: torch.Tensor, views: torch.Tensor) -> torch.Tensor:
        batch, nviews, height, width = images.shape
        x = normalize(images.reshape(batch * nviews, 1, height, width))
        if self.training:
            x = x * torch.empty_like(x[:, :, :1, :1]).uniform_(.82, 1.18)
            x = x + torch.empty_like(x[:, :, :1, :1]).uniform_(-.17, .17)
        global_image = F.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False, antialias=True)
        tiles = [global_image]
        if self.variant == 'attention':
            tiles.extend(F.interpolate(x[:, :, y:y + 192, column:column + 192], size=(160, 160),
                                       mode='bilinear', align_corners=False, antialias=True)
                         for y in (0, 160, 320) for column in (0, 128))
        tile_tensor = torch.stack(tiles, dim=1).reshape(-1, 1, 160, 160)
        valid = views.reshape(-1).repeat_interleave(len(tiles))
        encoded = self.encoder(tile_tensor[valid].contiguous(memory_format=torch.channels_last)).mean(dim=(2, 3))
        embedding = self.project(encoded)
        features = embedding.new_zeros(batch * nviews * len(tiles), embedding.shape[-1])
        features[valid] = embedding
        features = features.reshape(batch, nviews, len(tiles), -1)
        if self.variant == 'attention':
            features = features + self.region
            scores = self.patch_score(self.tanh(features).tanh() * self.sigmoid(features).sigmoid())
            aggregated = (scores.float().softmax(dim=2).to(features.dtype) * features).sum(dim=2)
            features = aggregated + features[:, :, 0]
        else:
            features = features[:, :, 0]
        features = features + self.view_type
        if self.variant == 'attention':
            context, _ = self.view_attn(features, features, features, key_padding_mask=~views, need_weights=False)
            features = self.view_norm(features + context)
        features = features.masked_fill(~views.unsqueeze(-1), 0)
        return self.head(torch.cat((features.flatten(1), views.to(features.dtype)), dim=1))


def build_model(task: str, variant: str, pretrained: bool) -> nn.Module:
    if task == 'chest':
        return ChestClassifier(pretrained, variant)
    if task == 'mammo':
        return BreastClassifier(pretrained, variant)
    raise ValueError(f'Unknown task {task}')
