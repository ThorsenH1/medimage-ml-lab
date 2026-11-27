"""Transfer the public CXR-CLIP Swin model trained on MIMIC-CXR and CheXpert."""
from __future__ import annotations

import re
from pathlib import Path

import timm
import torch
from torch import nn

from medimage.auc_models import legacy_weight_metadata
from medimage.common import checksum

CXR_CLIP_HASH = '19ad633804daa3fd6d48d26a93d3731356eed8dfc836102b95d1c5965c144608'


def convert_swin(source: dict) -> dict:
    converted = {}
    prefixes = {'embeddings.patch_embeddings.projection.': 'patch_embed.proj.',
                'embeddings.norm.': 'patch_embed.norm.', 'layernorm.': 'norm.'}
    blocks = {'layernorm_before': 'norm1', 'layernorm_after': 'norm2',
              'attention.output.dense': 'attn.proj', 'intermediate.dense': 'mlp.fc1',
              'output.dense': 'mlp.fc2'}
    for name, tensor in source.items():
        if name.endswith('.relative_position_index') or name.endswith('.attn_mask'):
            continue
        for old, new in prefixes.items():
            if name.startswith(old):
                converted[new + name.removeprefix(old)] = tensor
                break
        else:
            stage, category, part = re.fullmatch(r'encoder\.layers\.(\d+)\.(downsample|blocks)\.(.*)', name).groups()
            if category == 'downsample':
                converted[f'layers.{int(stage) + 1}.downsample.{part}'] = tensor
                continue
            block, component = part.split('.', 1)
            root = f'layers.{stage}.blocks.{block}.'
            if component == 'attention.self.relative_position_bias_table':
                converted[root + 'attn.relative_position_bias_table'] = tensor
                continue
            module, kind = component.rsplit('.', 1)
            if module == 'attention.self.query':
                base = name.removesuffix(f'query.{kind}')
                converted[root + f'attn.qkv.{kind}'] = torch.cat(
                    [source[base + f'{projection}.{kind}'] for projection in ('query', 'key', 'value')])
            elif module not in ('attention.self.key', 'attention.self.value'):
                converted[root + blocks[module] + '.' + kind] = tensor
    return converted


def load_image_tower(path: Path) -> dict:
    if checksum(path) != CXR_CLIP_HASH:
        raise ValueError('Expected the released Swin Tiny checkpoint trained on MIMIC-CXR and CheXpert')
    with torch.serialization.safe_globals(legacy_weight_metadata()):
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    prefix = 'image_encoder.image_encoder.'
    source = {key.removeprefix(prefix): value for key, value in saved['model'].items() if key.startswith(prefix)}
    if not source:
        raise ValueError('The released checkpoint does not contain the expected Swin image tower')
    return convert_swin(source)


class ChestCxrClip(nn.Module):
    def __init__(self, depth: int, dropout: float, weights: Path | None = None, *, resolution: int = 224) -> None:
        super().__init__()
        self.weight_sha256 = CXR_CLIP_HASH
        self.encoder = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0,
                                         img_size=resolution)
        if weights is not None:
            self.encoder.load_state_dict(load_image_tower(weights), strict=True)
        self.encoder.requires_grad_(False)
        self.depth = depth
        for stage in self.encoder.layers[-depth:]:
            stage.requires_grad_(True)
        self.encoder.norm.requires_grad_(True)
        channels = self.encoder.num_features
        self.head = nn.Sequential(nn.LayerNorm(channels * 2), nn.Dropout(dropout), nn.Linear(channels * 2, 14))

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.patch_embed.eval()
        for stage in self.encoder.layers[:-self.depth]:
            stage.eval()
        return self

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        image = pixels[:, None].float() / 255
        if self.training:
            shape = (len(image), 1, 1, 1)
            gain = torch.empty(shape, device=image.device).uniform_(.88, 1.12)
            bias = torch.empty(shape, device=image.device).uniform_(-.035, .035)
            image = (image * gain + bias).clamp_(0, 1)
            image = torch.where(torch.rand(shape, device=image.device) < .5, image.flip(-1), image)
        image = (2 * image - 1).expand(-1, 3, -1, -1)
        features = self.encoder.forward_features(image)
        return self.head(torch.cat((features.mean((1, 2)), features.amax((1, 2))), dim=-1))
