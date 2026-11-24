from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from medimage.common import checksum, load_manifest, read_json, write_json
from medimage.radiography_data import CHEST_LABELS
from medimage.radiography_train import save_checkpoint, validation_metrics
from medimage.search_features import Encoder
from medimage.search_heads import DeviceBank, class_weights


PROFILES = ('effb0-224', 'dino-224')


@dataclass(frozen=True)
class FusionRecipe:
    width: int
    dropout: float
    learning_rate: float
    class_power: float
    ranking: float
    backbone_dropout: float


RECIPES = (
    FusionRecipe(96, .34, .0007, .28, 0., .12),
    FusionRecipe(128, .42, .0008, .35, .40, .17),
    FusionRecipe(160, .48, .0005, .50, .70, .20),
)


class FusionHead(nn.Module):
    def __init__(self, dimensions, recipe: FusionRecipe):
        super().__init__()
        width = recipe.width
        self.recipe = recipe
        self.branches = nn.ModuleList(nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width), nn.GELU())
                                      for dim in dimensions)
        self.position = nn.Parameter(torch.randn(5, width) * .015)
        self.project = nn.Sequential(nn.Linear(width * len(dimensions), width), nn.LayerNorm(width), nn.GELU())
        self.score = nn.Linear(width, 14)
        self.readout = nn.Parameter(torch.randn(14, width) * .025)
        self.global_head = nn.Sequential(nn.LayerNorm(width * 2), nn.Dropout(recipe.dropout),
                                         nn.Linear(width * 2, 14))

    def forward(self, banks):
        textures = [transform(values.float()) for transform, values in zip(self.branches, banks, strict=True)]
        if self.training:
            drop = torch.rand(len(textures), len(textures[0]), 1, 1, device=textures[0].device)
            observed = drop > self.recipe.backbone_dropout
            mask = observed | ~observed.any(dim=0, keepdim=True)
            textures = [values * view for values, view in zip(textures, mask, strict=True)]
        tokens = self.project(torch.cat(textures, dim=-1)) + self.position
        weights = self.score(tokens).float().softmax(dim=1)
        local = torch.einsum('bpc,bpw->bcw', weights, tokens.float())
        global_logits = self.global_head(torch.cat((tokens[:, 0], tokens[:, 1:].mean(1)), dim=-1))
        return global_logits + (local * self.readout).sum(-1)


def bank_for(root: Path, profiles: dict, manifest: str, rows):
    banks = [DeviceBank(root / name, manifest, 'chest', rows) for name in PROFILES]
    if [bank.info for bank in banks] != [profiles[key] for key in PROFILES]:
        raise ValueError('The fusion encoder profiles differ from the training provenance')
    return banks


def tensors_for(banks, rows):
    indices = torch.as_tensor(rows, device='cuda')
    return [bank.data[indices] for bank in banks]


@torch.inference_mode()
def cached_prediction(model, banks, rows, correction):
    model.eval()
    values = []
    for start in range(0, len(rows), 768):
        with torch.autocast('cuda', dtype=torch.float16):
            scores = model(tensors_for(banks, rows[start:start + 768]))
        values.append(scores.float().cpu().numpy())
    return np.concatenate(values) - correction


@torch.inference_mode()
def raw_prediction(model, config, input_pixels, weights, correction):
    vectors = []
    for profile in PROFILES:
        specification = config['profiles'][profile]
        encoder = Encoder(specification['backbone'], specification['size'], weights).cuda().eval()
        if encoder.weight_sha256 != specification['pretrained_sha256']:
            raise ValueError('The raw fusion encoder differs from the trained profile')
        with torch.autocast('cuda', dtype=torch.float16):
            vectors.append(encoder(torch.from_numpy(input_pixels).cuda(), spatial=True).half())
        del encoder
    with torch.autocast('cuda', dtype=torch.float16):
        return model(vectors).float().cpu().numpy() - correction


def fit(args):
    torch.set_num_threads(4)
    frame, audit = load_manifest(args.cohort)
    manifest = audit['manifest_sha256']
    train = np.flatnonzero(frame.split.eq('train').to_numpy())
    validation = np.flatnonzero(frame.split.eq('val').to_numpy())
    labels = frame[list(CHEST_LABELS)].to_numpy(np.float32, copy=True)
    truth = torch.from_numpy(labels).cuda()
    profiles = {name: read_json(args.features / name / 'features.json') for name in PROFILES}
    banks = bank_for(args.features, profiles, manifest, np.r_[train, validation])
    dimensions = [bank.shape[-1] for bank in banks]
    pixel_hash = checksum(args.pixels / 'pixels.json')
    for index, recipe in enumerate(RECIPES):
        directory = args.output / f'recipe-{index + 1}'
        directory.mkdir(parents=True, exist_ok=True)
        config = {'task': 'chest', 'backbone': 'chest-fusion', 'manifest': manifest, 'pixel_sha256': pixel_hash,
                  'profiles': profiles, 'dimensions': dimensions, 'resolution': 224,
                  'recipe': asdict(recipe), 'seed': args.seed + index, 'epochs': args.epochs}
        if (directory / 'result.json').is_file():
            if read_json(directory / 'result.json')['config'] != config:
                raise ValueError('The saved fusion recipe has changed')
            continue
        torch.manual_seed(config['seed'])
        model = FusionHead(dimensions, recipe).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate, weight_decay=.009, fused=True)
        scaler = torch.amp.GradScaler('cuda', init_scale=2048.)
        weight = class_weights(labels[train], recipe.class_power)
        correction = np.log(weight)
        weight_cuda = torch.from_numpy(weight).cuda()
        best, trace = -float('inf'), []
        for epoch in range(args.epochs):
            begin = time.monotonic()
            order = np.random.default_rng(config['seed'] + epoch).permutation(train)
            model.train()
            for start in range(0, len(order), 512):
                rows = order[start:start + 512]
                y = truth[rows]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(tensors_for(banks, rows)).float()
                    loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=weight_cuda)
                    if recipe.ranking:
                        labels_shuffled = y.roll(1, dims=0)
                        pairs = y * (1 - labels_shuffled) * weight_cuda
                        rank_loss = (F.softplus(logits.roll(1, dims=0) - logits) * pairs).sum()
                        loss = loss + recipe.ranking * rank_loss / pairs.sum().clamp_min(1)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scores = cached_prediction(model, banks, validation, correction)
            metrics = validation_metrics(labels[validation], scores)
            trace.append({'epoch': epoch + 1, 'seconds': round(time.monotonic() - begin, 2), **metrics})
            write_json(directory / 'history.json', {'config': config, 'epochs': trace})
            if metrics['macro_auroc'] > best:
                best = metrics['macro_auroc']
                save_checkpoint(directory / 'best.pt', {'config': config, 'model': model.state_dict(),
                                                        'prior_correction': correction.tolist(), 'epoch': epoch + 1})
                np.savez_compressed(directory / 'validation.partial.npz', rows=validation, logits=scores)
                (directory / 'validation.partial.npz').replace(directory / 'validation.npz')
            print(f'Fusion {index + 1}', trace[-1], flush=True)
        write_json(directory / 'result.json', {'config': config, 'epochs': trace,
                                               'best_sha256': checksum(directory / 'best.pt'),
                                               'validation_sha256': checksum(directory / 'validation.npz')})


def main():
    parser = argparse.ArgumentParser(description='Train joint chest feature heads with validation-only selection')
    for name in ('features', 'cohort', 'pixels', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--seed', type=int, default=4610)
    fit(parser.parse_args())


if __name__ == '__main__':
    main()
