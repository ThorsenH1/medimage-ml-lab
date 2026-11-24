from __future__ import annotations

import pickle
import re
import typing
import warnings
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import timm
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from torchvision.models import densenet121
from torchvision.models.densenet import DenseNet, _DenseBlock, _DenseLayer, _Transition

from medimage.common import checksum
from medimage.radiography_models import ChestClassifier
from medimage.search_features import Encoder


CHEX_HASH = '1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4'
DENSE_HASH = 'a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395'
MAMMO_CLIP_HASH = '0baccb0cc19cd6093ff8252f7c2b7a6efbc32973498640abcb311210c46fd02c'
EFFICIENTNET_B3_HASH = '279d2a53898aa89dab43fd6bd7df9f706aea4cb9cf916223988bbcb8e5850469'


def chest_weights(path: Path, source: str) -> dict:
    if checksum(path) != (CHEX_HASH if source == 'chex' else DENSE_HASH):
        raise ValueError('The pretrained checkpoint differs from the documented source')
    if source == 'chex':
        # The published legacy module uses persistent module IDs. Resolve its
        # finite set of globals locally, after verifying the full artifact hash.
        allowed = {(value.__module__, value.__name__): value for value in
                   (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d,
                    nn.Linear, nn.Sequential, nn.ModuleDict, OrderedDict,
                    torch._utils._rebuild_parameter, torch._utils._rebuild_tensor_v2)}
        allowed.update({('torchxrayvision.models', value.__name__): value
                        for value in (DenseNet, _DenseBlock, _DenseLayer, _Transition)})
        allowed.update({('builtins', 'set'): set, ('__builtin__', 'set'): set})

        class CheckpointUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if (module, name) not in allowed:
                    raise pickle.UnpicklingError(f'Unexpected checkpoint global {module}.{name}')
                return allowed[module, name]

        safely_mapped = SimpleNamespace(__name__='chexpert_checkpoint', Unpickler=CheckpointUnpickler,
                                        load=pickle.load)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', torch.serialization.SourceChangeWarning)
            loaded = torch.load(path, map_location='cpu', weights_only=False, pickle_module=safely_mapped)
        state = loaded.state_dict() if isinstance(loaded, nn.Module) else loaded
    else:
        state = torch.load(path, map_location='cpu', weights_only=True)
    features = {re.sub(r'(norm|relu|conv)\.([12])\.', r'\1\2.', name.removeprefix('features.')): value
                for name, value in state.items() if name.startswith('features.')}
    if features['conv0.weight'].shape[1] == 3:
        features['conv0.weight'] = features['conv0.weight'].sum(dim=1, keepdim=True)
    return features


class UnusedConfiguration:
    """Inert representation of legacy configuration objects in the source archive."""


def plain_metadata(*_):
    return {}


def mammo_clip(weights: Path) -> nn.Module:
    if checksum(weights) != MAMMO_CLIP_HASH:
        raise ValueError('Mammo-CLIP checkpoint differs from the documented source')
    metadata_classes = ('omegaconf.listconfig.ListConfig', 'omegaconf.dictconfig.DictConfig',
                        'omegaconf.nodes.AnyNode', 'omegaconf.base.ContainerMetadata', 'omegaconf.base.Metadata')
    allowed = [(UnusedConfiguration, name) for name in metadata_classes]
    allowed += [typing.Any, list, dict, int, (plain_metadata, 'collections.defaultdict')]
    with torch.serialization.safe_globals(allowed):
        saved = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
    source = {name.removeprefix('image_encoder.'): value for name, value in saved['model'].items()
              if name.startswith('image_encoder.')}
    encoder = timm.create_model('tf_efficientnet_b5', pretrained=False, num_classes=0)
    block_map = []
    for stage, blocks in enumerate(encoder.blocks):
        for position, block in enumerate(blocks):
            block_map.append((stage, position, hasattr(block, 'conv_pwl')))
    converted = {}
    for name, value in source.items():
        if name.startswith(('_fc.', '_avg_pooling.')):
            continue
        if name.startswith('_blocks.'):
            _, index, component, attribute = name.split('.')
            stage, position, expanded = block_map[int(index)]
            components = {'_depthwise_conv': 'conv_dw', '_se_reduce': 'se.conv_reduce',
                          '_se_expand': 'se.conv_expand'}
            components.update({'_expand_conv': 'conv_pw', '_project_conv': 'conv_pwl',
                               '_bn0': 'bn1', '_bn1': 'bn2', '_bn2': 'bn3'} if expanded else
                              {'_project_conv': 'conv_pw', '_bn1': 'bn1', '_bn2': 'bn2'})
            destination = f'blocks.{stage}.{position}.{components[component]}.{attribute}'
        else:
            component, attribute = name.split('.')
            prefix = {'_conv_stem': 'conv_stem', '_bn0': 'bn1', '_conv_head': 'conv_head', '_bn1': 'bn2'}
            destination = f'{prefix[component]}.{attribute}'
        converted[destination] = value
    encoder.load_state_dict(converted)
    return encoder.to(memory_format=torch.channels_last)


class ChestDense(nn.Module):
    def __init__(self, source: str, stage: int, pretrained: Path | None = None) -> None:
        super().__init__()
        self.source = source
        self.encoder = densenet121().features
        self.encoder.conv0 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        if pretrained is not None:
            self.encoder.load_state_dict(chest_weights(pretrained, source))
        self.encoder.requires_grad_(False)
        for name, layer in self.encoder.named_children():
            if name in ('denseblock2', 'transition2') and stage <= 2 or (
                name in ('denseblock3', 'transition3') and stage <= 3
            ) or name in ('denseblock4', 'norm5'):
                layer.requires_grad_(True)
        self.head = nn.Sequential(nn.LayerNorm(2048), nn.Dropout(.22), nn.Linear(2048, 14))

    def train(self, mode: bool = True):
        super().train(mode)
        for layer in self.encoder.children():
            if not any(p.requires_grad for p in layer.parameters()):
                layer.eval()
        return self

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = pixels[:, None].float() / 255
        if self.training:
            shape = (len(x), 1, 1, 1)
            gain = torch.empty(shape, device=x.device).uniform_(.86, 1.14)
            bias = torch.empty(shape, device=x.device).uniform_(-.035, .035)
            x = (x * gain + bias).clamp_(0, 1)
            x = torch.where(torch.rand(shape, device=x.device) < .5, x.flip(-1), x)
        x = x * 2048 - 1024 if self.source == 'chex' else (x - .449) / .226
        features = self.encoder(x.contiguous(memory_format=torch.channels_last)).relu()
        return self.head(torch.cat((features.mean((-1, -2)), features.amax((-1, -2))), dim=-1))


class ChestReference(ChestClassifier):
    def __init__(self, depth: int) -> None:
        super().__init__(pretrained=False, variant='global')
        self.encoder.requires_grad_(False)
        self.encoder[-depth:].requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return super().forward(pixels, None)


class MammographySpatial(nn.Module):
    def __init__(self, backbone: str, resolution: int, weights: Path, depth: int, dropout: float,
                 pyramid: bool = False, initialize: bool = True, specialize: bool = False) -> None:
        super().__init__()
        if backbone == 'mammo-clip':
            self.encoder = (mammo_clip(weights) if initialize else
                            timm.create_model('tf_efficientnet_b5', pretrained=False, num_classes=0))
            self.weight_sha256 = MAMMO_CLIP_HASH
            mean = torch.full((1, 1, 1, 1), .3089279)
            std = torch.full((1, 1, 1, 1), .250535554)
            self.rgb = True
        elif backbone == 'effb3':
            self.encoder = timm.create_model('efficientnet_b3.ra2_in1k', pretrained=False, num_classes=0)
            self.weight_sha256 = EFFICIENTNET_B3_HASH
            if initialize:
                if checksum(weights) != self.weight_sha256:
                    raise ValueError('EfficientNet-B3 weights differ from the pinned ImageNet release')
                state = {name: value for name, value in load_file(weights).items()
                         if not name.startswith('classifier.')}
                self.encoder.load_state_dict(state)
            settings = timm.data.resolve_model_data_config(self.encoder)
            mean = torch.tensor(settings['mean'])[None, :, None, None]
            std = torch.tensor(settings['std'])[None, :, None, None]
            self.rgb = False
        else:
            foundation = Encoder(backbone, resolution, weights)
            self.encoder = foundation.model
            self.weight_sha256 = foundation.weight_sha256
            mean, std = foundation.mean, foundation.std
            self.rgb = False
        self.register_buffer('mean', mean, persistent=False)
        self.register_buffer('std', std, persistent=False)
        self.encoder.requires_grad_(False)
        if backbone in ('effb0', 'effb3', 'mammo-clip'):
            layers = [*self.encoder.blocks[-depth:], self.encoder.conv_head, self.encoder.bn2]
        elif backbone in ('r18', 'r50'):
            layers = [self.encoder.layer4] if depth == 1 else [self.encoder.layer3, self.encoder.layer4]
        else:
            raise ValueError('Expected a convolutional mammography encoder')
        for layer in layers:
            layer.requires_grad_(True)
        self.pyramid = pyramid
        if pyramid:
            feature_info = self.encoder.feature_info
            self.stages = tuple(i for i, entry in enumerate(feature_info) if entry['reduction'] in (8, 16))
            self.lateral = nn.ModuleList(nn.Conv2d(feature_info[index]['num_chs'], 128, 1)
                                         for index in self.stages)
        self.spatial = nn.Sequential(nn.Conv2d(self.encoder.num_features, 128, 1), nn.GroupNorm(8, 128), nn.SiLU())
        self.lesion_head = nn.Conv2d(128, 4, 1)
        self.view_head = nn.Sequential(nn.LayerNorm(128 * 3 + 12), nn.Linear(128 * 3 + 12, 128),
                                       nn.SiLU(), nn.Dropout(dropout))
        self.view_type = nn.Parameter(torch.zeros(2, 128))
        self.token_score = nn.Linear(128, 1)
        self.head = nn.Sequential(nn.LayerNorm(256), nn.Dropout(dropout), nn.Linear(256, 1))
        self.specialize = specialize
        if specialize:
            self.router = nn.Linear(256, 2)
            self.offset = nn.Linear(256, 2)
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)
        self.resolution = resolution

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, pixels: torch.Tensor, observed: torch.Tensor):
        batch, views, height, width = pixels.shape
        x = pixels.reshape(batch * views, 1, height, width).float() / 255
        target_size = (self.resolution, self.resolution * 5 // 8)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False, antialias=True)
        if self.training:
            shape = (len(x), 1, 1, 1)
            gain = torch.empty(shape, device=x.device).uniform_(.84, 1.18)
            exponent = torch.empty(shape, device=x.device).uniform_(.85, 1.17)
            x = (x.clamp_min(1e-6).pow(exponent) * gain).clamp_(0, 1)
        x = (x - self.mean) / self.std
        if self.rgb:
            x = x.expand(-1, 3, -1, -1)
        x = x.contiguous(memory_format=torch.channels_last)
        if self.pyramid:
            encoded, levels = self.encoder.forward_intermediates(x, indices=list(self.stages))
            feature = F.interpolate(self.spatial(encoded), size=levels[-1].shape[-2:], mode='bilinear',
                                    align_corners=False)
            for project, level in zip(self.lateral, levels, strict=True):
                feature = feature + F.adaptive_avg_pool2d(project(level), feature.shape[-2:])
        else:
            feature = self.spatial(self.encoder.forward_features(x))
        lesion = self.lesion_head(feature).float()
        values = feature.flatten(2)
        evidence = lesion.flatten(2)
        attention = (2 * evidence.sigmoid().amax(dim=1)).softmax(dim=-1).to(values.dtype)
        focused = (values * attention[:, None]).sum(dim=-1)
        top = evidence.topk(8, dim=-1).values
        summary = torch.cat((values.mean(-1), values.amax(-1), focused, top[..., 0],
                             top[..., :3].mean(-1), top.mean(-1)), dim=-1)
        token = self.view_head(summary).reshape(batch, views, -1) + self.view_type
        if self.training:
            drop = torch.rand_like(observed, dtype=torch.float32) < .12
            keep = observed & ~drop
            observed = torch.where(keep.any(1, keepdim=True), keep, observed)
        token_logits = self.token_score(token).squeeze(-1).masked_fill(~observed, -torch.inf)
        fused = (token * token_logits.softmax(dim=-1).unsqueeze(-1)).sum(dim=1)
        maximum = token.masked_fill(~observed[..., None], -torch.inf).amax(dim=1)
        embedding = torch.cat((fused, maximum), dim=-1)
        logits = self.head(embedding)
        lesion = lesion.reshape(batch, views, 4, *lesion.shape[-2:])
        if not self.specialize:
            return logits, lesion
        routing = self.router(embedding).float()
        experts = logits.float() + self.offset(embedding).float()
        probability = (routing.softmax(1) * experts.sigmoid()).sum(1, keepdim=True)
        return torch.logit(probability.clamp(1e-6, 1 - 1e-6)), lesion, routing, experts


def spatial_loss(logits: torch.Tensor, masks: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    target = F.adaptive_max_pool2d(masks.flatten(0, 1).float(), logits.shape[-2:]).reshape_as(logits)
    valid = available[:, :, None, None, None]
    positive = valid * target
    negative = valid * (1 - target)
    pos_loss = F.softplus(-logits) * (1 - logits.sigmoid()).square()
    neg_loss = F.softplus(logits) * logits.sigmoid().square()
    return ((positive * pos_loss).sum() / positive.sum().clamp_min(1)
            + (negative * neg_loss).sum() / negative.sum().clamp_min(1))


def align_augmentation(pixels: torch.Tensor, heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, views = pixels.shape[:2]
    angle = torch.empty(batch * views, device=pixels.device).uniform_(-.22, .22)
    scale = torch.empty_like(angle).uniform_(.82, 1.18)
    cosine, sine = angle.cos() * scale, angle.sin() * scale
    tx = torch.empty_like(angle).uniform_(-.08, .08)
    ty = torch.empty_like(angle).uniform_(-.08, .08)
    theta = torch.stack((cosine, sine, tx, -sine, cosine, ty), dim=-1).reshape(-1, 2, 3)
    image = pixels.flatten(0, 1)[:, None].float()
    label = heatmaps.flatten(0, 1).float()
    image_grid = F.affine_grid(theta, image.shape, align_corners=False)
    label_grid = F.affine_grid(theta, label.shape, align_corners=False)
    image = F.grid_sample(image, image_grid, align_corners=False).reshape_as(pixels)
    label = F.grid_sample(label, label_grid, align_corners=False).reshape_as(heatmaps) > .25
    return image, label
