from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from medimage.common import checksum, read_json, write_json
from medimage.radiography_data import CHEST_LABELS
from medimage.radiography_models import ChestClassifier
from medimage.radiography_train import save_checkpoint, validation_metrics
from medimage.search_features import Encoder, load_manifest, mammography_tiles
from medimage.search_heads import FeatureClassifier, Recipe, class_weights, objective


class AdaptedClassifier(nn.Module):
    def __init__(self, task: str, metadata: dict, recipe: Recipe, weights: Path) -> None:
        super().__init__()
        self.task = task
        self.profile = metadata
        name = metadata['backbone']
        self.encoder = Encoder(name, metadata['size'], weights)
        if metadata['pretrained_sha256'] != self.encoder.weight_sha256:
            raise ValueError('Adapted model encoder uses different foundation weights')
        self.encoder.requires_grad_(False)
        feature_model = self.encoder.model
        if name == 'r50':
            selected = [feature_model.layer4[-1]]
        elif name == 'effb0':
            selected = [feature_model.blocks[-1], feature_model.conv_head, feature_model.bn2]
        elif name == 'dino':
            selected = [*feature_model.blocks[-2:], feature_model.norm]
        else:
            raise ValueError('Unsupported adaptation backbone')
        for layer in selected:
            layer.requires_grad_(True)
        self.head = FeatureClassifier(self.encoder.dim, 5 if task == 'chest' else 14,
                                      14 if task == 'chest' else 1, recipe)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, pixels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.training:
            pixels = pixels.float()
            view = (len(pixels),) + (1,) * (pixels.ndim - 1)
            gain = torch.empty(view, device=pixels.device).uniform_(.9, 1.1)
            bias = torch.empty(view, device=pixels.device).uniform_(-9, 9)
            pixels = (pixels * gain + bias).clamp(0, 255)
            if self.task == 'chest':
                pixels = torch.where(torch.rand(view, device=pixels.device) < .5, pixels.flip(-1), pixels)
        if self.task == 'chest':
            values = self.encoder(pixels, spatial=True)
        else:
            size = self.profile['size']
            tiles = mammography_tiles(pixels, size).flatten(0, 2)
            valid = self.encoder(tiles[mask.reshape(-1)])
            values = valid.new_zeros((len(tiles), self.encoder.dim))
            values[mask.reshape(-1)] = valid
            values = values.reshape(len(pixels), 14, -1)
        return self.head(values, mask)


class HighResolutionReference(ChestClassifier):
    def __init__(self) -> None:
        super().__init__(pretrained=False, variant='global')
        self.encoder.requires_grad_(False)
        self.encoder[-1].requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self


def pixel_store(pixels_root: Path, cohort: Path, task: str):
    frame, audit = load_manifest(cohort)
    signature = read_json(pixels_root / 'pixels.json')
    if signature['manifest'] != audit['manifest_sha256'] or signature['task'] != task:
        raise ValueError('Adaptation pixel store belongs to a different cohort')
    pixels = np.load(pixels_root / 'pixels.npy', mmap_mode='r', allow_pickle=False)
    mask = (frame[['CC', 'MLO']].ne('').to_numpy().repeat(7, axis=1) if task == 'mammo'
            else np.ones((len(frame), 5), dtype=bool))
    return frame, audit, pixels, mask


@torch.inference_mode()
def logits_for(model: nn.Module, pixels, masks, rows, batch: int) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(rows), batch):
        take = rows[start:start + batch]
        with torch.autocast('cuda', dtype=torch.float16):
            z = model(torch.from_numpy(pixels[take]).cuda(), torch.from_numpy(masks[take]).cuda())
        chunks.append(z.float().cpu().numpy())
    result = np.concatenate(chunks)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite adapted predictions')
    return result


def load_adapted(path: Path, weights: Path) -> tuple[nn.Module, dict]:
    saved = torch.load(path, map_location='cpu', weights_only=True)
    config = saved['config']
    model = (HighResolutionReference() if config['profile'] == 'incumbent-224' else
             AdaptedClassifier(config['task'], config['base'], Recipe(**config['recipe']), weights)).cuda()
    model.load_state_dict(saved['model'])
    return model.eval(), saved


def adapt(task: str, cohort: Path, pixels_root: Path, search: Path, weights: Path, original: Path,
          *, epochs: int, batch: int) -> None:
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    frame, audit, pixels, masks = pixel_store(pixels_root, cohort, task)
    search_results = read_json(search / 'search.json')
    take = max(search_results['finalists'], key=lambda entry: entry['score'])
    profile = take['profile']
    head_path = search / 'final' / profile / str(take['seeds'][0]) / 'model.pt'
    starter = torch.load(head_path, map_location='cpu', weights_only=True)
    base = starter['config']['features']
    if base['manifest'] != audit['manifest_sha256']:
        raise ValueError('Adaptation starter and cohort disagree')
    recipe = Recipe(**take['recipe'])
    seed = 4413
    torch.manual_seed(seed)
    model = AdaptedClassifier(task, base, recipe, weights).cuda()
    model.head.load_state_dict(starter['model'])
    tune(task, model, model.head, frame, audit, pixels, masks, recipe, profile, base, head_path,
         search / 'adaptation', epochs, batch, seed, 2e-5, 1.6e-4)
    del model
    torch.cuda.empty_cache()
    if task == 'chest':
        reference = original / 'runs' / 'chest-global' / 'best.pt'
        recorded = torch.load(reference, map_location='cpu', weights_only=True)
        if recorded['config']['task'] != task or recorded['config']['manifest'] != audit['manifest_sha256']:
            raise ValueError('High-resolution reference starter came from a different cohort')
        torch.manual_seed(seed)
        higher = HighResolutionReference().cuda()
        higher.load_state_dict(recorded['model'])
        reference_recipe = Recipe('linear', 0, 1.2e-4, .003, .2, .5)
        tune(task, higher, higher.global_head, frame, audit, pixels, masks, reference_recipe, 'incumbent-224',
             {'task': 'chest', 'size': 224, 'parent': 'incumbent-global'}, reference,
             search / 'adaptation-reference', epochs, batch, seed, 3e-5, 1.2e-4)


def tune(task: str, model: nn.Module, head: nn.Module, frame, audit: dict, pixels, masks, recipe: Recipe,
         profile: str, base: dict, head_path: Path, directory: Path, epochs: int, batch: int,
         seed: int, backbone_lr: float, head_lr: float) -> None:
    backbone = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': backbone_lr}, {'params': head.parameters(), 'lr': head_lr}],
                                  weight_decay=.003, fused=True)
    grad = torch.amp.GradScaler('cuda', init_scale=4096.)
    train = np.flatnonzero(frame.split.eq('train').to_numpy())
    val = np.flatnonzero(frame.split.eq('val').to_numpy())
    labels = frame.loc[:, list(CHEST_LABELS) if task == 'chest' else ['positive']].to_numpy(np.float32, copy=True)
    target_weight = class_weights(labels[train], recipe.class_power)
    loss_weight = torch.from_numpy(target_weight).cuda()
    directory.mkdir(exist_ok=True)
    config = {'task': task, 'profile': profile, 'manifest': audit['manifest_sha256'], 'seed': seed,
              'epochs': epochs, 'batch': batch, 'recipe': asdict(recipe), 'base': base,
              'starter_sha256': checksum(head_path), 'code_sha256': checksum(Path(__file__)),
              'trainable_backbone': sum(parameter.numel() for parameter in backbone)}
    state_path = directory / 'last.pt'
    best_path = directory / 'best.pt'
    if (directory / 'result.json').is_file():
        done = read_json(directory / 'result.json')
        restored = torch.load(best_path, map_location='cpu', weights_only=True)
        stored_config = done['config']
        comparable = {name: value for name, value in config.items() if name != 'code_sha256'}
        if (any(stored_config.get(name) != value for name, value in comparable.items())
                or restored['config'] != stored_config or not (directory / 'validation.npz').is_file()):
            raise ValueError('Completed fine-tuning output differs from requested recipe')
        print(f'Validated completed {task} fine-tuning', flush=True)
        return
    history = []
    best = -float('inf')
    if state_path.exists():
        saved = torch.load(state_path, map_location='cpu', weights_only=True)
        if saved['config'] != config:
            raise ValueError('Cannot resume changed fine-tuning experiment')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        grad.load_state_dict(saved['scaler'])
        history, best = saved['history'], saved['best']
    for epoch in range(len(history), epochs):
        begin = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(seed + epoch)
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(train)
        total_loss = torch.zeros((), device='cuda')
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                z = model(torch.from_numpy(pixels[idx]).cuda(), torch.from_numpy(masks[idx]).cuda())
            loss = F.binary_cross_entropy_with_logits(z.float(), torch.from_numpy(labels[idx]).cuda(), pos_weight=loss_weight)
            grad.scale(loss).backward()
            grad.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.)
            grad.step(optimizer)
            grad.update()
            total_loss += loss.detach() * len(idx)
            if (start // batch + 1) % 180 == 0:
                print(f'{task}/{profile} adaptation epoch {epoch + 1}, images {start + len(idx)}/{len(order)}', flush=True)
        raw = logits_for(model, pixels, masks, val, batch)
        result = validation_metrics(labels[val], raw)
        metric = objective(task, result)
        train_loss = float(total_loss) / len(order)
        if not np.isfinite(train_loss):
            raise FloatingPointError('Nonfinite adapted train loss')
        record = {'epoch': epoch + 1, 'score': metric, 'metrics': result, 'train_loss': train_loss,
                  'seconds': round(time.monotonic() - begin, 2), 'peak_mib': round(torch.cuda.max_memory_reserved() / 2**20, 1)}
        history.append(record)
        if metric > best:
            best = metric
            save_checkpoint(best_path, {'config': config, 'model': model.state_dict(), 'weights': target_weight.tolist(), 'validation': record})
            np.savez_compressed(directory / 'validation.npz', rows=val, logits=raw - np.log(target_weight))
        save_checkpoint(state_path, {'config': config, 'model': model.state_dict(), 'history': history,
                                     'best': best, 'optimizer': optimizer.state_dict(), 'scaler': grad.state_dict()})
        write_json(directory / 'history.json', {'config': config, 'history': history})
        print(f'{task}/{profile} adaptation {record}', flush=True)
    saved = torch.load(best_path, map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model'])
    raw = logits_for(model, pixels, masks, val, batch)
    np.savez_compressed(directory / 'validation.npz', rows=val, logits=raw - np.log(target_weight))
    write_json(directory / 'result.json', {'name': f'adapted-{profile}', 'config': config, 'score': best,
                                          'validation': saved['validation'], 'completed_epochs': len(history)})


def main() -> None:
    parser = argparse.ArgumentParser(description='Fine-tune the strongest validation finalist on full images and local patches')
    parser.add_argument('task', choices=('chest', 'mammo'))
    for name in ('cohort', 'pixels', 'search', 'weights'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--original', type=Path, default=Path('artifacts'))
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError('Positive fine-tuning run sizes are required')
    adapt(args.task, args.cohort, args.pixels, args.search, args.weights, args.original,
          epochs=args.epochs, batch=args.batch_size)


if __name__ == '__main__':
    main()
