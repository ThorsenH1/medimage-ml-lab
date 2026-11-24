"""Learn spinal disc landmarks from the sagittal images in a frozen fold."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from medimage.common import checksum, write_json
from medimage.radiography_train import save_checkpoint
from medimage.spine_data import SIZE, STARTS, anchor_slot, project
from medimage.spine_localize import SpineCache, read_folds, world_from_pixels

MAP_SIZE = SIZE // 4


def conv(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                         nn.GroupNorm(8, out_channels), nn.SiLU())


class DiscHeatmap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(conv(3, 24, 2), conv(24, 24))
        self.level2 = nn.Sequential(conv(24, 40, 2), conv(40, 40))
        self.level3 = nn.Sequential(conv(40, 64, 2), conv(64, 64))
        self.level4 = nn.Sequential(conv(64, 96, 2), conv(96, 96), conv(96, 96))
        self.decode3 = nn.Sequential(conv(160, 64), conv(64, 64))
        self.decode2 = nn.Sequential(conv(104, 48), conv(48, 48), nn.Conv2d(48, 5, 1))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        low = self.level2(self.stem(pixels))
        mid = self.level3(low)
        high = self.level4(mid)
        up = F.interpolate(high, size=mid.shape[-2:], mode='bilinear', align_corners=False)
        mid = self.decode3(torch.cat((mid, up), dim=1))
        up = F.interpolate(mid, size=low.shape[-2:], mode='bilinear', align_corners=False)
        return self.decode2(torch.cat((low, up), dim=1))


def anchor_inputs(cache: SpineCache, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    planes = []
    geometry = []
    for row in rows:
        available = cache.valid[row]
        anchor = anchor_slot(available)
        view = np.searchsorted(STARTS[1:], anchor, side='right')
        first, last = STARTS[view:view + 2]
        candidates = np.flatnonzero(available[first:last]) + first
        adjacent = candidates[np.clip(np.searchsorted(candidates, anchor) + np.arange(-1, 2),
                                      0, len(candidates) - 1)]
        planes.append(cache.volumes[row, adjacent])
        geometry.append(cache.geometry[row, anchor])
    return np.stack(planes), np.stack(geometry)


def pixel_grid(device: torch.device) -> torch.Tensor:
    positions = (torch.arange(MAP_SIZE, device=device, dtype=torch.float32) + .5) * SIZE / MAP_SIZE - .5
    y, x = torch.meshgrid(positions, positions, indexing='ij')
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def landmark_loss(logits: torch.Tensor, coordinates: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    observed = torch.isfinite(coordinates).all(dim=-1)
    target = torch.nan_to_num(coordinates)
    separation = (target[..., None, :] - grid).square().sum(dim=-1)
    density = (-separation / (2 * 4.0 ** 2)).softmax(dim=-1)
    distribution = logits.float().flatten(2).log_softmax(dim=-1)
    heatmap = -(density * distribution).sum(dim=-1)
    estimate = distribution.exp() @ grid
    coordinate = F.smooth_l1_loss(estimate, target, reduction='none', beta=4).mean(dim=-1)
    return ((heatmap + .08 * coordinate) * observed).sum() / observed.sum().clamp_min(1)


def decode(logits: torch.Tensor, grid: torch.Tensor) -> np.ndarray:
    flat = logits.float().flatten(2)
    peak = grid[flat.argmax(dim=-1)]
    close = (peak[..., None, :] - grid).abs().amax(dim=-1) <= 12
    scores = flat.masked_fill(~close, -torch.inf).softmax(dim=-1)
    return (scores @ grid).cpu().numpy()


def augment(pixels: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    count = len(pixels)
    device = pixels.device
    scale = torch.empty(count, device=device).uniform_(.90, 1.10)
    offset = torch.empty(count, 2, device=device).uniform_(-.10, .10)
    angle = torch.empty(count, device=device).uniform_(-.12, .12)
    cosine, sine = angle.cos() * scale, angle.sin() * scale
    basis = torch.stack((cosine, -sine, sine, cosine), dim=-1).view(count, 2, 2)
    theta = torch.cat((basis, offset[..., None]), dim=-1)
    sampling = F.affine_grid(theta, pixels.shape, align_corners=False)
    pixels = F.grid_sample(pixels, sampling, align_corners=False)
    pixels = pixels * torch.empty_like(pixels[:, :1, :1, :1]).uniform_(.78, 1.22)
    centered = (2.0 * (targets + .5) / SIZE - 1) - offset[:, None]
    transformed = centered @ torch.linalg.inv(basis).transpose(-1, -2)
    targets = (transformed + 1) * SIZE / 2.0 - .5
    inside = ((targets >= 0) & (targets < SIZE)).all(dim=-1)
    return pixels, targets.masked_fill(~inside[..., None], float('nan'))


def mm_metrics(locations: np.ndarray, truth: np.ndarray, geometry: np.ndarray) -> dict:
    error = np.linalg.norm(world_from_pixels(locations, geometry) - world_from_pixels(truth, geometry), axis=-1)
    valid = np.isfinite(error)
    if not valid.any() or not np.isfinite(locations).all():
        raise ValueError('MRI landmarks contain no valid paired evidence')
    return {'observed': int(valid.sum()), 'median_mm': float(np.nanmedian(error)),
            'mean_mm': float(np.nanmean(error)), 'within_10mm': float((error[valid] <= 10).mean()),
            'median_mm_by_level': np.nanmedian(error, axis=0).tolist()}


def fit_heatmaps(inputs, xy, geometry, fit_rows, validation_rows, audit, output: Path) -> None:
    seed, epochs, batch, lr = (audit[key] for key in ('seed', 'epochs', 'batch', 'lr'))
    torch.manual_seed(seed)
    device = torch.device('cuda')
    axes = pixel_grid(device)
    model = DiscHeatmap().to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.008)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr / 12)
    scaler = torch.amp.GradScaler('cuda')
    pixels = torch.from_numpy(inputs)
    coordinates = torch.from_numpy(xy)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'best.pt').is_file() or (output / 'final.pt').is_file():
        raise ValueError('Use an unused output folder for a new heatmap fit')
    history = []
    best = float('inf')
    for epoch in range(epochs):
        model.train()
        losses = []
        shuffled = np.random.default_rng(seed + epoch).permutation(fit_rows)
        for indices in torch.as_tensor(shuffled).split(batch):
            image = pixels[indices].to(device).float().div_(255.)
            labels = coordinates[indices].to(device)
            image, labels = augment(image, labels)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                heatmaps = model(image.contiguous(memory_format=torch.channels_last))
            loss = landmark_loss(heatmaps, labels, axes)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.detach())
        schedule.step()
        record = {'epoch': epoch + 1, 'train_loss': float(torch.stack(losses).mean())}
        if not len(validation_rows):
            history.append(record)
            continue
        model.eval()
        predictions = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            for indices in torch.as_tensor(validation_rows).split(batch):
                image = pixels[indices].to(device).float().div_(255.)
                predictions.append(decode(model(image), axes))
        estimates = np.concatenate(predictions)
        report = mm_metrics(estimates, xy[validation_rows], geometry[validation_rows])
        record.update(report)
        history.append(record)
        if report['median_mm'] < best:
            best = report['median_mm']
            save_checkpoint(output / 'best.pt', {'state': model.state_dict(), 'audit': audit, 'best': record})
            np.savez_compressed(output / 'validation-points.npz',
                                rows=validation_rows, coordinates=estimates)
        write_json(output / 'validation.json', {'audit': audit, 'epochs': history})
        print(record, flush=True)
    if len(validation_rows):
        print(f'Best validation median localization error: {best:.3f} mm', flush=True)
    else:
        save_checkpoint(output / 'final.pt', {'state': model.state_dict(), 'audit': audit, 'history': history})


def observed_landmarks(cache: SpineCache, geometry: np.ndarray) -> np.ndarray:
    xy, _ = project(cache.points[:, :5], geometry[:, None])
    xy = xy.astype(np.float32)
    xy[~((xy >= 0) & (xy < SIZE)).all(axis=-1)] = np.nan
    return xy


def pilot(args: argparse.Namespace) -> None:
    torch.set_num_threads(4)
    cache = SpineCache.load(args.cache)
    folds = read_folds(args.folds, cache.studies)
    inner = (args.outer + 1) % 5
    rows = np.arange(len(folds))
    fit_rows = rows[~np.isin(folds, (args.outer, inner))]
    validation_rows = rows[folds == inner]
    inputs, geometry = anchor_inputs(cache, rows)
    xy = observed_landmarks(cache, geometry)
    audit = {'outer_excluded': args.outer, 'validation_fold': inner,
             'training_folds': sorted(set(folds[fit_rows].tolist())),
             'folds_sha256': checksum(args.folds), 'cache_sha256': checksum(args.cache / 'audit.json'),
             'code_sha256': checksum(Path(__file__)), 'seed': args.seed,
             'epochs': args.epochs, 'batch': args.batch, 'lr': args.lr}
    fit_heatmaps(inputs, xy, geometry, fit_rows, validation_rows, audit, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'folds', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--outer', type=int, choices=range(5), default=0)
    parser.add_argument('--epochs', type=int, default=18)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=9409)
    args = parser.parse_args()
    if min(args.epochs, args.batch) < 1 or args.lr <= 0:
        parser.error('Epochs, batch size and learning rate must be positive')
    pilot(args)


if __name__ == '__main__':
    main()
