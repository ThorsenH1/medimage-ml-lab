from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from medimage.auc_train import store, transfer
from medimage.common import checksum, read_json, write_json
from medimage.radiography_train import save_checkpoint, validation_metrics
from medimage.search_features import Encoder

PATCH = 128
SCALES = ((144, 72), (256, 112))
POOLING = ('peak', 'top3', 'top8', 'type_balanced')


class PatchNet(nn.Module):
    def __init__(self, weights: Path):
        super().__init__()
        foundation = Encoder('r18', PATCH, weights)
        self.encoder = foundation.model
        self.weight_sha256 = foundation.weight_sha256
        self.register_buffer('mean', foundation.mean, persistent=False)
        self.register_buffer('std', foundation.std, persistent=False)
        self.head = nn.Sequential(nn.LayerNorm(512), nn.Dropout(.2), nn.Linear(512, 5))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255
        if self.training:
            shape = (len(x), 1, 1, 1)
            x = x.clamp_min(1e-6).pow(torch.empty(shape, device=x.device).uniform_(.8, 1.22))
            x = (x * torch.empty(shape, device=x.device).uniform_(.8, 1.22)).clamp_(0, 1)
            x = torch.where(torch.rand(shape, device=x.device) < .5, x.flip(-1), x)
            x = torch.where(torch.rand(shape, device=x.device) < .5, x.flip(-2), x)
        normalized = ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
        features = self.encoder.forward_features(normalized)
        return features.mean((-1, -2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


def crop(image, y, x, width):
    h, w = image.shape
    r = width // 2
    x, y = np.clip(x - r, 0, w - width), np.clip(y - r, 0, h - width)
    return cv2.resize(image[y:y + width, x:x + width], (PATCH, PATCH), interpolation=cv2.INTER_AREA)


def training_patches(pixels, boxes, complete, rows, seed: int, repeats: int):
    rng = np.random.default_rng(seed)
    patches, labels = [], []
    for row in rows:
        for view in range(2):
            lesions = boxes[row, view][boxes[row, view, :, 0] >= 0]
            if not len(lesions):
                continue
            image = pixels[row, view]
            h, w = image.shape
            for category, x1, y1, x2, y2 in lesions:
                for base in rng.choice([144, 192, 256, 320], size=repeats):
                    side = int(np.clip(max(base, 1.1 * (max(x2 - x1, y2 - y1) + 1)), 128, 416))
                    dx, dy = rng.integers(-side // 6, side // 6 + 1, size=2)
                    patches.append(crop(image, (y1 + y2) // 2 + dy, (x1 + x2) // 2 + dx, side))
                    labels.append(int(category) + 1)
            if not complete[row, view]:
                continue
            left, top, right, bottom = lesions[:, 1:].T
            accepted = 0
            for _ in range(28):
                side = int(rng.choice([144, 192, 256, 320]))
                radius = side // 2
                y, x = int(rng.integers(radius, h - radius)), int(rng.integers(radius, w - radius))
                if np.any((x + radius >= left) & (x - radius <= right)
                          & (y + radius >= top) & (y - radius <= bottom)):
                    continue
                patch = crop(image, y, x, side)
                if (patch > 10).mean() < .50:
                    continue
                patches.append(patch)
                labels.append(0)
                accepted += 1
                if accepted >= len(lesions) * repeats:
                    break
    if len(set(labels)) != 5:
        raise ValueError('Expected background plus four training lesion subtypes')
    return np.stack(patches), np.asarray(labels)


def train_patch_epoch(model, arrays, boxes, train, seed, repeats, batch, optimizer, scaler):
    torch.manual_seed(seed)
    data, truth = training_patches(arrays['pixels'], boxes, arrays['annotation_available'],
                                   train, seed, repeats)
    order = np.random.default_rng(seed + 1000).permutation(len(truth))
    frequencies = np.bincount(truth, minlength=5) / len(truth)
    balance = torch.tensor((frequencies.mean() / frequencies) ** .45, device='cuda', dtype=torch.float32)
    total = torch.zeros((), device='cuda')
    model.train()
    for start in range(0, len(order), batch):
        rows = order[start:start + batch]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(transfer(data, rows)[:, None])
            loss = F.cross_entropy(logits, transfer(truth, rows), weight=balance, label_smoothing=.035)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.detach() * len(rows)
    return {'patches': len(truth), 'patch_class_counts': np.bincount(truth).tolist(),
            'train_loss': float(total / len(truth))}


@torch.inference_mode()
def scan(model: PatchNet, arrays, rows, batch: int = 96) -> dict[str, np.ndarray]:
    model.eval()
    scores = []
    for row in rows:
        image = transfer(arrays['pixels'], [row]).flatten(0, 1)[:, None].float()
        views = transfer(arrays['views'], [row]).flatten()
        per_scale = []
        for width, stride in SCALES:
            patches = F.unfold(image[views], width, padding=width // 2, stride=stride)
            patches = patches.transpose(1, 2).reshape(-1, 1, width, width)
            included = (patches > 10).float().mean((1, 2, 3)) > .15
            patches = patches[included]
            if not len(patches):
                continue
            x = F.interpolate(patches, size=(PATCH, PATCH), mode='bilinear', align_corners=False,
                              antialias=True)
            with torch.autocast('cuda', dtype=torch.float16):
                per_scale.extend(model(x[start:start + batch]).float().softmax(-1)
                                 for start in range(0, len(x), batch))
        if not per_scale:
            raise ValueError('A mammogram has no searchable tissue patches')
        probabilities = torch.cat(per_scale)
        malignant = probabilities[:, [2, 4]].sum(dim=1)
        strongest = malignant.topk(min(8, len(malignant))).values
        ranked = probabilities[:, [2, 4]].topk(min(3, len(malignant)), dim=0).values.mean(0)
        scores.append(torch.stack((strongest[0], strongest[:3].mean(), strongest.mean(),
                                   ranked.mean())).cpu().numpy())
    probability = np.stack(scores).clip(1e-6, 1 - 1e-6)
    logits = np.log(probability) - np.log1p(-probability)
    return {name: logits[:, index:index + 1] for index, name in enumerate(POOLING)}


def fit(args):
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    frame, audit, arrays, pixel_hash = store('mammo', args.cohort, args.pixels)
    labels = frame[['positive']].to_numpy(np.float32, copy=True)
    train = np.flatnonzero(frame.split.eq('train').to_numpy())
    validation = np.flatnonzero(frame.split.eq('val').to_numpy())
    boxes = np.load(args.pixels / 'lesions.npy', mmap_mode='r', allow_pickle=False)
    if (boxes[frame.split.ne('train').to_numpy()] >= 0).any():
        raise ValueError('Only training cases may supply lesion bounding boxes')
    model = PatchNet(args.weights).cuda().to(memory_format=torch.channels_last)
    config = {'task': 'mammo', 'backbone': 'patch-r18', 'seed': args.seed, 'epochs': args.epochs,
              'batch': args.batch, 'repeats': args.repeats, 'manifest': audit['manifest_sha256'],
              'pixel_sha256': pixel_hash, 'weight_sha256': model.weight_sha256}
    optimizer = torch.optim.AdamW([{'params': model.encoder.parameters(), 'lr': .00013},
                                  {'params': model.head.parameters(), 'lr': .0005}],
                                  weight_decay=.004, fused=True)
    scaler = torch.amp.GradScaler('cuda', init_scale=2048.)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'last.pt'
    history, best = [], -float('inf')
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if config != saved['config']:
            raise ValueError('Patch recipe changed since the last checkpoint')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        history, best = saved['history'], saved['best']
    for epoch in range(len(history), args.epochs):
        begin = time.monotonic()
        summary = {'epoch': epoch + 1, **train_patch_epoch(model, arrays, boxes, train, args.seed + epoch,
                                                          args.repeats, args.batch, optimizer, scaler)}
        checkpoint = {'config': config, 'model': model.state_dict(), 'epoch': epoch + 1}
        if (epoch + 1) % 2 == 0 or epoch + 1 == args.epochs:
            print(f'Patch epoch {epoch + 1}: scanning full validation mammograms', flush=True)
            logits = scan(model, arrays, validation)
            metrics = {kind: validation_metrics(labels[validation], score) for kind, score in logits.items()}
            pool = max(POOLING, key=lambda name: (metrics[name]['macro_auroc'], metrics[name]['macro_auprc']))
            summary.update(pooling=pool, validation=metrics)
            if metrics[pool]['macro_auroc'] > best:
                best = metrics[pool]['macro_auroc']
                save_checkpoint(args.output / 'best.pt', {**checkpoint, 'pooling': pool, 'validation': metrics[pool]})
                np.savez_compressed(args.output / 'validation.partial.npz', rows=validation, logits=logits[pool])
                (args.output / 'validation.partial.npz').replace(args.output / 'validation.npz')
        summary['seconds'] = round(time.monotonic() - begin, 2)
        history.append(summary)
        save_checkpoint(path, {**checkpoint, 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                               'history': history, 'best': best})
        write_json(args.output / 'history.json', {'config': config, 'epochs': history})
        print(summary, flush=True)
    write_json(args.output / 'result.json', {'config': config, 'epochs': history,
                                            'best_sha256': checksum(args.output / 'best.pt'),
                                            'validation_sha256': checksum(args.output / 'validation.npz')})


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a patch model and scan unannotated mammograms')
    for name in ('cohort', 'pixels', 'weights', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch', type=int, default=80)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=4497)
    args = parser.parse_args()
    if min(args.epochs, args.batch, args.repeats) < 1:
        parser.error('Use positive epoch, batch and patch counts')
    fit(args)


if __name__ == '__main__':
    main()
