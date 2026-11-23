from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset

from medimage.common import checksum, read_json, seed_everything, write_json
from medimage.radiography_train import save_checkpoint
from medimage.spine_localize import SpineCache, read_folds
from medimage.spine_metrics import check_probabilities, read_targets, score, submission
from medimage.spine_model import SpineClassifier, crop_regions, severity_loss


class RegionDataset(Dataset):
    def __init__(self, cache: SpineCache, indices: np.ndarray, region: np.ndarray,
                 metadata: np.ndarray, labels: np.ndarray, augment: bool = False) -> None:
        self.path = cache.path / 'volume.npy'
        self.indices = indices
        self.region = region
        self.metadata = metadata
        self.labels = labels
        self.augment = augment
        self.volumes = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.volumes is None:
            cv2.setNumThreads(1)
            self.volumes = np.load(self.path, mmap_mode='r', allow_pickle=False)
        row = self.indices[index]
        offset = np.random.uniform(-.07, .07, size=(30, 2)) if self.augment else None
        image = crop_regions(self.volumes[row], self.region[row], offset)
        return image, self.metadata[row], self.labels[row]


@torch.inference_mode()
def predict(model, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    outputs, truth = [], []
    with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False):
        for images, metadata, labels in loader:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                result = model(images.cuda(non_blocking=True), metadata.cuda(non_blocking=True))
            outputs.append(result.float().softmax(-1).cpu().numpy())
            truth.append(labels.numpy())
    probabilities = np.concatenate(outputs)
    check_probabilities(probabilities)
    return probabilities, np.concatenate(truth)


def _load_fold(path: Path, cache: SpineCache, folds: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved['studies'], cache.studies) or not np.array_equal(saved['fold_per_study'], folds):
            raise ValueError('Localization and frozen split are misaligned')
        roles = saved['trained_on']
        desired = (np.arange(5)[None] != folds[:, None]) & (np.arange(5)[None] != fold)
        if not np.array_equal(roles, desired):
            raise ValueError('Nested localization provenance does not hold out expected study folds')
        return saved['region'], saved['metadata']


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or args.epochs < 1 or args.batch_size < 2:
        raise ValueError('Spine training requires CUDA, positive epochs and a batch of at least two studies')
    seed = args.seed + args.fold
    seed_everything(seed)
    torch.hub.set_dir(str(args.weights / 'hub'))
    weight_path = args.weights / 'hub' / 'checkpoints' / 'resnet18-f37072fd.pth'
    if not weight_path.is_file() or not checksum(weight_path).startswith('f37072fd'):
        raise ValueError('A verified local ImageNet ResNet-18 checkpoint is required')
    cache = SpineCache.load(args.cache)
    labels = read_targets(args.root, cache.studies)
    folds = read_folds(args.folds, cache.studies)
    localization = read_json(args.localization / 'localization.json')
    if (localization['config']['folds_file_sha256'] != checksum(args.folds)
            or localization['config']['cache_sha256'] != checksum(args.cache / 'audit.json')):
        raise ValueError('Localization report came from another cohort')
    region_path = args.localization / f'fold-{args.fold}.npz'
    region, metadata = _load_fold(region_path, cache, folds, args.fold)
    train_rows, val_rows = np.flatnonzero(folds != args.fold), np.flatnonzero(folds == args.fold)
    if args.limit:
        rng = np.random.default_rng(seed)
        train_rows, val_rows = (np.sort(rng.choice(rows, min(args.limit, len(rows)), replace=False))
                                for rows in (train_rows, val_rows))
    train_set = RegionDataset(cache, train_rows, region, metadata, labels, augment=True)
    val_set = RegionDataset(cache, val_rows, region, metadata, labels)
    generator = torch.Generator().manual_seed(seed + 819)
    common = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, **common, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, **common)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {'seed': seed, 'fold': args.fold, 'epochs': args.epochs, 'batch_size': args.batch_size,
              'workers': args.workers, 'limit': args.limit, 'folds_sha256': checksum(args.folds),
              'region_sha256': checksum(region_path), 'cache_sha256': checksum(args.cache / 'audit.json'),
              'final_weights': 'arithmetic_mean_of_last_three_epoch_checkpoints',
              'code_sha256': {name: checksum(Path(__file__).with_name(name)) for name in
                              ('spine_train.py', 'spine_model.py', 'spine_metrics.py')}}
    model = SpineClassifier(pretrained=True).cuda()
    averaged = AveragedModel(model, device='cpu', use_buffers=True)
    backbone = list(model.encoder.parameters())
    backbone_ids = {id(parameter) for parameter in backbone}
    head = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    optimizer = torch.optim.AdamW([{'params': backbone}, {'params': head}], weight_decay=3e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[1.9e-4, 6e-4],
                                                    total_steps=args.epochs * len(train_loader), pct_start=.18)
    scaler = torch.amp.GradScaler('cuda', init_scale=4096)
    history = []
    last = args.output / 'last.pt'
    if last.exists():
        checkpoint = torch.load(last, weights_only=False, map_location='cpu')
        if checkpoint['config'] != config:
            raise ValueError('Spine checkpoint configuration mismatch')
        model.load_state_dict(checkpoint['model'])
        averaged.load_state_dict(checkpoint['averaged'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng'])
        generator.set_state(checkpoint['loader_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])
        history = checkpoint['history']
    print(f'Spine fold {args.fold}, {len(train_set)} train / {len(val_set)} val, '
          f'{len(history)} of {args.epochs} epochs completed', flush=True)
    for epoch in range(len(history), args.epochs):
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        summed_loss = torch.zeros((), device='cuda')
        skipped = 0
        model.train()
        for step, (images, meta, target) in enumerate(train_loader, 1):
            images, meta, target = (value.cuda(non_blocking=True) for value in (images, meta, target))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(images, meta)
            loss = severity_loss(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.)
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale:
                scheduler.step()
            else:
                skipped += 1
            summed_loss += loss.detach() * len(images)
            if step % 75 == 0:
                print(f'Fold {args.fold} epoch {epoch + 1}, {step}/{len(train_loader)} batches', flush=True)
        train_loss = float(summed_loss.cpu()) / len(train_set)
        if not np.isfinite(train_loss) or skipped == len(train_loader):
            raise FloatingPointError('Non-finite spine training loss or zero completed optimizer steps')
        probs, targets = predict(model, val_loader)
        result = {'epoch': epoch + 1, 'train_loss': train_loss, **score(targets, probs),
                  'seconds': round(time.monotonic() - start, 2), 'skipped_fp16_steps': skipped,
                  'peak_reserved_mib': round(torch.cuda.max_memory_reserved() / 2**20, 1)}
        history.append(result)
        print(result, flush=True)
        if epoch >= args.epochs - 3:
            averaged.update_parameters(model)
        save_checkpoint(last, {'model': model.state_dict(), 'averaged': averaged.state_dict(),
                               'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                               'scheduler': scheduler.state_dict(), 'config': config, 'history': history,
                               'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                               'loader_rng': generator.get_state(), 'numpy_rng': np.random.get_state(),
                               'python_rng': random.getstate()})
        write_json(args.output / 'history.json', {'config': config, 'history': history})
    model.load_state_dict(averaged.module.state_dict())
    probabilities, truth = predict(model, val_loader)
    result = {'config': config, 'validation': score(truth, probabilities), 'training_studies': len(train_rows),
              'validation_studies': len(val_rows), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'parameters': sum(parameter.numel() for parameter in model.parameters()),
              'gpu': torch.cuda.get_device_name(), 'averaged_epoch_count': int(averaged.n_averaged.item())}
    if args.fold == 0 and not args.limit:
        oracle_path = args.localization / 'oracle-fold-0.npz'
        with np.load(oracle_path, allow_pickle=False) as oracle:
            if not np.array_equal(oracle['rows'], val_rows):
                raise ValueError('Localizer oracle rows do not match the fold')
            oracle_region, oracle_meta = region.copy(), metadata.copy()
            oracle_region[val_rows], oracle_meta[val_rows] = oracle['region'], oracle['metadata']
        oracle_set = RegionDataset(cache, val_rows, oracle_region, oracle_meta, labels)
        oracle_probabilities, oracle_truth = predict(model, DataLoader(oracle_set, **common))
        result['oracle_coordinates_sensitivity_only'] = score(oracle_truth, oracle_probabilities)
    save_checkpoint(args.output / 'final.pt', {'model': model.state_dict(), 'config': config})
    submission(cache.studies[val_rows], probabilities).to_csv(args.output / 'oof.csv', index=False)
    write_json(args.output / 'result.json', result)
    print(f"Spine fold {args.fold} fixed endpoint score {result['validation']['score']:.5f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a fixed-endpoint localized MR classifier for one outer fold')
    for name in ('root', 'cache', 'folds', 'localization', 'output', 'weights'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1801)
    parser.add_argument('--limit', type=int)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
