from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from medimage.common import checksum, read_json, seed_everything, write_json
from medimage.common import load_manifest as load_frame
from medimage.radiography_data import CHEST_LABELS, SEED
from medimage.radiography_evaluate import validation_metrics
from medimage.radiography_models import build_model


class ImageCache(Dataset):
    def __init__(self, task: str, cache: Path, frame: pd.DataFrame, indices: np.ndarray) -> None:
        self.path = cache / 'images.npy'
        self.indices = indices
        self.labels = frame.loc[:, list(CHEST_LABELS) if task == 'chest' else ['positive']].to_numpy(np.float32)
        self.views = (frame[['CC', 'MLO']].ne('').to_numpy() if task == 'mammo'
                      else np.ones((len(frame), 1), dtype=np.bool_))
        self.images = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.images is None:
            self.images = np.load(self.path, mmap_mode='r', allow_pickle=False)
        row = self.indices[index]
        return self.images[row].copy(), self.views[row], self.labels[row]


def make_loader(task: str, cache: Path, frame: pd.DataFrame, split: str, batch_size: int,
                workers: int, *, generator=None, limit: int | None = None) -> DataLoader:
    indices = np.flatnonzero(frame.split.eq(split).to_numpy())
    if limit is not None:
        indices = np.random.default_rng(SEED).choice(indices, min(limit, len(indices)), replace=False)
    dataset = ImageCache(task, cache, frame, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0, generator=generator,
                      drop_last=(split == 'train' and len(dataset) % batch_size == 1))


@torch.inference_mode()
def predict_batches(model, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    chunks, truth = [], []
    for images, views, labels in loader:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(images.cuda(non_blocking=True), views.cuda(non_blocking=True))
        chunks.append(logits.float().cpu().numpy())
        truth.append(labels.numpy())
    result = np.concatenate(chunks)
    if not np.isfinite(result).all():
        raise FloatingPointError('Validation/inference logits are not finite')
    return result, np.concatenate(truth)


def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix('.partial')
    torch.save(state, tmp)
    os.replace(tmp, path)


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('This training profile requires a CUDA device')
    if args.epochs < 1 or args.batch_size < 2 or args.workers < 0:
        raise ValueError('Use at least one epoch, batch size two, and nonnegative workers')
    seed_everything(args.seed)
    torch.hub.set_dir(str(args.weights / 'hub'))
    weight_file = args.weights / 'hub' / 'checkpoints' / 'resnet18-f37072fd.pth'
    if not weight_file.is_file():
        raise FileNotFoundError(f'ImageNet transfer weights are required locally at {weight_file}')
    if not checksum(weight_file).startswith('f37072fd'):
        raise ValueError('Local ResNet18 ImageNet checkpoint failed its checksum')
    frame, audit = load_frame(args.cache)
    args.output.mkdir(parents=True, exist_ok=True)
    signature = {'task': args.task, 'variant': args.variant, 'epochs': args.epochs,
                 'batch_size': args.batch_size, 'seed': args.seed, 'limit': args.limit,
                 'manifest': audit['manifest_sha256'], 'revision': 1,
                 'code_sha256': {name: checksum(Path(__file__).with_name(name)) for name in
                                 ('radiography_models.py', 'radiography_train.py')}}
    generator = torch.Generator().manual_seed(args.seed + 1)
    train_loader = make_loader(args.task, args.cache, frame, 'train', args.batch_size, args.workers,
                               generator=generator, limit=args.limit)
    val_loader = make_loader(args.task, args.cache, frame, 'val', args.batch_size, args.workers,
                             limit=args.limit)
    target = train_loader.dataset.labels[train_loader.dataset.indices]
    positives = target.sum(axis=0)
    balancing = np.clip(np.sqrt((len(target) - positives + 1) / (positives + 1)), 1, 6)
    weights = torch.as_tensor(balancing, device='cuda', dtype=torch.float32)
    model = build_model(args.task, args.variant, pretrained=True).cuda()
    encoder = [parameter for parameter in model.encoder.parameters()]
    feature_ids = {id(parameter) for parameter in encoder}
    head = [parameter for parameter in model.parameters() if id(parameter) not in feature_ids]
    optimizer = torch.optim.AdamW([{'params': encoder, 'lr': 1.4e-4}, {'params': head, 'lr': 5e-4}], weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[1.4e-4, 5e-4],
                                                    total_steps=args.epochs * len(train_loader), pct_start=.15)
    scaler = torch.amp.GradScaler('cuda', init_scale=4096)
    last = args.output / 'last.pt'
    best = args.output / 'best.pt'
    history = []
    best_measure = -float('inf')
    if last.exists():
        checkpoint = torch.load(last, weights_only=False, map_location='cpu')
        if checkpoint['config'] != signature:
            raise ValueError('Run configuration differs from the checkpoint')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng'])
        generator.set_state(checkpoint['loader_rng'])
        random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        history = checkpoint['history']
        best_measure = checkpoint['best_measure']
    print(f'{args.task}/{args.variant}: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, '
          f'{len(history)}/{args.epochs} completed epochs on {torch.cuda.get_device_name()}', flush=True)
    for epoch in range(len(history), args.epochs):
        model.train()
        begin = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        summed_loss = torch.zeros((), device='cuda')
        skipped = 0
        for step, (images, views, labels) in enumerate(train_loader, 1):
            images, views, labels = (value.cuda(non_blocking=True) for value in (images, views, labels))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images, views)
            loss = F.binary_cross_entropy_with_logits(logits.float(), labels, pos_weight=weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale:
                scheduler.step()
            else:
                skipped += 1
            summed_loss += loss.detach() * len(images)
            if step % 150 == 0:
                print(f'epoch {epoch + 1}, step {step}/{len(train_loader)}, {time.monotonic() - begin:.1f}s', flush=True)
        logits, truth = predict_batches(model, val_loader)
        metrics = validation_metrics(truth, logits)
        train_loss = float(summed_loss.cpu()) / len(train_loader.dataset)
        if not np.isfinite(train_loss) or skipped == len(train_loader):
            raise FloatingPointError('Non-finite training loss or zero completed steps')
        result = {'epoch': epoch + 1, 'train_loss': train_loss, **metrics,
                  'seconds': round(time.monotonic() - begin, 2), 'skipped_fp16_steps': skipped,
                  'peak_reserved_mib': round(torch.cuda.max_memory_reserved() / 2**20, 1)}
        history.append(result)
        print(result, flush=True)
        measure = (metrics['macro_auroc'] if args.task == 'mammo' else
                   .5 * (metrics['macro_auroc'] + metrics['macro_auprc']))
        if measure > best_measure:
            best_measure = measure
            save_checkpoint(best, {'model': model.state_dict(), 'config': signature, 'validation': result})
        save_checkpoint(last, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                               'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                               'config': signature, 'history': history, 'best_measure': best_measure,
                               'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                               'loader_rng': generator.get_state(), 'python_rng': random.getstate(),
                               'numpy_rng': np.random.get_state()})
        write_json(args.output / 'history.json', {'config': signature, 'history': history})
    selected = torch.load(best, map_location='cpu', weights_only=True)['validation']
    write_json(args.output / 'result.json', {
        'config': signature, 'completed_epochs': len(history), 'best_validation': selected,
        'model_parameters': sum(parameter.numel() for parameter in model.parameters()),
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description='Fit an image classifier with validation-only checkpoint selection')
    parser.add_argument('task', choices=['chest', 'mammo'])
    parser.add_argument('--variant', choices=['global', 'attention'], default='attention')
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--limit', type=int)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
