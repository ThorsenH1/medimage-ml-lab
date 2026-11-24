from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from medimage.auc_models import ChestDense, ChestReference, MammographySpatial, align_augmentation, spatial_loss
from medimage.common import checksum, read_json, write_json
from medimage.radiography_data import CHEST_LABELS
from medimage.radiography_train import save_checkpoint, validation_metrics
from medimage.search_features import load_manifest
from medimage.search_heads import class_weights


def store(task: str, cohort: Path, path: Path):
    frame, audit = load_manifest(cohort)
    signature_path = path / ('pixels.json' if task == 'chest' else 'audit.json')
    signature = read_json(signature_path)
    actual = signature.get('manifest') if task == 'chest' else signature['config']['cohort']
    if actual != audit['manifest_sha256']:
        raise ValueError('Pixels belong to a different cohort')
    fields = ['pixels'] if task == 'chest' else ['pixels', 'views', 'heatmaps', 'annotation_available']
    arrays = {key: np.load(path / f'{key}.npy', mmap_mode='r', allow_pickle=False) for key in fields}
    if any(len(value) != len(frame) for value in arrays.values()):
        raise ValueError('Pixel arrays differ from cohort size')
    if task == 'mammo':
        if not np.array_equal(arrays['views'], frame[['CC', 'MLO']].ne('').to_numpy()):
            raise ValueError('Cached mammography views disagree with the manifest')
        if arrays['annotation_available'][frame.split.ne('train').to_numpy()].any():
            raise ValueError('Invalid lesion availability for held-out mammograms')
    return frame, audit, arrays, checksum(signature_path)


def new_model(config: dict, weights: Path, *, initialize: bool = False):
    if config['task'] == 'chest':
        if config['backbone'] == 'chest-reference':
            return ChestReference(config['depth'])
        return ChestDense(config['backbone'], config['depth'], weights if initialize else None)
    return MammographySpatial(config['backbone'], config['resolution'], weights,
                              config['depth'], config['dropout'], config.get('pyramid', False), initialize)


class ResidentPixels:
    """Pack only development images into memory, preserving cohort row indices."""

    def __init__(self, array, rows):
        self.data = np.empty((len(rows), *array.shape[1:]), dtype=array.dtype)
        self.mapping = np.full(len(array), -1, dtype=np.int64)
        order = np.sort(rows)
        self.mapping[order] = np.arange(len(order))
        for start in range(0, len(order), 384):
            take = order[start:start + 384]
            self.data[start:start + len(take)] = array[take]

    def __getitem__(self, rows):
        locations = self.mapping[rows]
        if (locations < 0).any():
            raise ValueError('A row is outside the resident development partition')
        return self.data[locations]


def training_batches(pixels, training, rng, block, batch):
    if not block:
        order = rng.permutation(training)
        for start in range(0, len(order), batch):
            rows = order[start:start + batch]
            yield rows, transfer(pixels, rows)
        return
    block = max(batch, block // batch * batch)
    starts = np.arange(0, len(training), block)
    for start in rng.permutation(starts):
        rows = training[start:start + block]
        # A sequential read avoids seeking to a different disk region for each image.
        packed = np.asarray(pixels[rows[0]:rows[-1] + 1])[rows - rows[0]]
        shuffled = rng.permutation(len(rows))
        for offset in range(0, len(rows), batch):
            take = shuffled[offset:offset + batch]
            yield rows[take], transfer(packed, take)


def transfer(array, rows):
    return torch.from_numpy(np.ascontiguousarray(array[rows])).to('cuda', non_blocking=True)


def infer_batch(model, task, arrays, rows, tta):
    pixels = transfer(arrays['pixels'], rows)
    if task == 'mammo':
        return model(pixels, transfer(arrays['views'], rows))[0]
    logits = model(pixels).float()
    return (logits + model(pixels.flip(-1)).float()) * .5 if tta else logits


@torch.inference_mode()
def predict(model, task: str, arrays, rows: np.ndarray, batch: int, tta: bool,
            prior_correction: np.ndarray) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(rows), batch):
        with torch.autocast('cuda', dtype=torch.float16):
            logits = infer_batch(model, task, arrays, rows[start:start + batch], tta)
        chunks.append(logits.float().cpu().numpy())
    result = np.concatenate(chunks) - prior_correction
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite validation/inference prediction')
    return result


def fit(args) -> None:
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.seed)
    frame, audit, arrays, pixel_hash = store(args.task, args.cohort, args.pixels)
    training = np.flatnonzero(frame.split.eq('train').to_numpy())
    validation = np.flatnonzero(frame.split.eq('val').to_numpy())
    labels = frame.loc[:, list(CHEST_LABELS) if args.task == 'chest' else ['positive']].to_numpy(np.float32, copy=True)
    if args.task == 'chest' and arrays['pixels'].shape[1:] != (args.resolution, args.resolution):
        raise ValueError('Requested chest resolution differs from the pixel cache')
    config = {name: getattr(args, name) for name in (
        'task', 'backbone', 'depth', 'resolution', 'epochs', 'seed', 'batch', 'accumulate',
        'lr', 'head_lr', 'class_power', 'auxiliary', 'dropout', 'tta', 'focal_neg')}
    config.update(manifest=audit['manifest_sha256'], pixel_sha256=pixel_hash,
                  parent_sha256=checksum(args.parent) if args.parent else None)
    for option in ('pyramid', 'affine', 'block'):
        if getattr(args, option):
            config[option] = getattr(args, option)
    model = new_model(config, args.weights, initialize=args.parent is None).cuda()
    model.to(memory_format=torch.channels_last)
    if args.task == 'mammo':
        config['weight_sha256'] = model.weight_sha256
    elif args.backbone == 'chest-reference':
        config['weight_sha256'] = config['parent_sha256']
    else:
        config['weight_sha256'] = checksum(args.weights)
    if args.parent:
        starter = torch.load(args.parent, map_location='cpu', weights_only=True)
        fixed = ('task', 'manifest') if args.backbone == 'chest-reference' else (
            'task', 'backbone', 'manifest', 'weight_sha256')
        if any(config[key] != starter['config'].get(key) for key in fixed):
            raise ValueError('Warm start differs from the expected architecture, cohort or foundation')
        model.load_state_dict(starter['model'])
    params = [p for name, p in model.named_parameters() if name.startswith('encoder.') and p.requires_grad]
    head = [p for name, p in model.named_parameters() if not name.startswith('encoder.') and p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': params, 'lr': args.lr},
                                  {'params': head, 'lr': args.head_lr}], weight_decay=.003, fused=True)
    scaler = torch.amp.GradScaler('cuda', init_scale=2048.)
    target_weights = class_weights(labels[training], args.class_power)
    correction = np.log(target_weights) if not args.focal_neg else np.zeros_like(target_weights)
    y = torch.from_numpy(labels).cuda()
    positive_weights = torch.from_numpy(target_weights).cuda()
    args.output.mkdir(parents=True, exist_ok=True)
    last = args.output / 'last.pt'
    history, best = [], -float('inf')
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=True)
        if config != saved['config']:
            raise ValueError('Resume configuration differs from the previous run')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        history, best = saved['history'], saved['best']
    if args.resident:
        print('Packing only training and validation pixels into RAM', flush=True)
        arrays['pixels'] = ResidentPixels(arrays['pixels'], np.r_[training, validation])
    print(f'{args.task} {args.backbone} {args.resolution}: {len(training)} train / '
          f'{len(validation)} validation, batch {args.batch} x {args.accumulate}', flush=True)
    for epoch in range(len(history), args.epochs):
        started = time.monotonic()
        rng = np.random.default_rng(args.seed + epoch)
        torch.manual_seed(args.seed + epoch)
        model.train()
        n_batches = (len(training) + args.batch - 1) // args.batch
        steps = (n_batches + args.accumulate - 1) // args.accumulate
        factor = .92 ** epoch
        optimizer.param_groups[0]['lr'] = args.lr * factor
        optimizer.param_groups[1]['lr'] = args.head_lr * factor
        optimizer.zero_grad(set_to_none=True)
        losses = torch.zeros((), device='cuda')
        batches = training_batches(arrays['pixels'], training, rng, args.block, args.batch)
        for i, (rows, pixels) in enumerate(batches):
            truth = y[rows]
            with torch.autocast('cuda', dtype=torch.float16):
                if args.task == 'mammo':
                    heatmaps = transfer(arrays['heatmaps'], rows)
                    if args.affine:
                        pixels, heatmaps = align_augmentation(pixels, heatmaps)
                    score, spatial = model(pixels, transfer(arrays['views'], rows))
                    lesion_loss = spatial_loss(spatial, heatmaps,
                                               transfer(arrays['annotation_available'], rows))
                else:
                    score, lesion_loss = model(pixels), 0.
                error = F.binary_cross_entropy_with_logits(score.float(), truth, pos_weight=positive_weights,
                                                           reduction='none')
                if args.focal_neg:
                    weight = torch.where(truth > 0, 1., score.detach().float().sigmoid().pow(args.focal_neg))
                    error = error * weight
                loss = error.mean() + args.auxiliary * lesion_loss
            group = min(args.accumulate, n_batches - (i // args.accumulate) * args.accumulate)
            scaler.scale(loss / group).backward()
            if (i + 1) % args.accumulate == 0 or i + 1 == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params + head, max_norm=2.)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses += loss.detach() * len(rows)
            if (i + 1) % max(40, n_batches // 5) == 0:
                print(f'epoch {epoch + 1}: {i + 1}/{n_batches} batches, {time.monotonic() - started:.0f}s',
                      flush=True)
        logits = predict(model, args.task, arrays, validation, args.batch, args.tta, correction)
        metrics = validation_metrics(labels[validation], logits)
        summary = {'epoch': epoch + 1, 'train_loss': float(losses / len(training)), 'optimizer_steps': steps,
                   'seconds': round(time.monotonic() - started, 2), **metrics}
        history.append(summary)
        checkpoint = {'config': config, 'model': model.state_dict(), 'epoch': epoch + 1,
                      'prior_correction': correction.tolist(), 'validation': metrics}
        if metrics['macro_auroc'] > best:
            best = metrics['macro_auroc']
            save_checkpoint(args.output / 'best.pt', checkpoint)
            temporary = args.output / 'validation.partial.npz'
            np.savez_compressed(temporary, rows=validation, logits=logits)
            temporary.replace(args.output / 'validation.npz')
        save_checkpoint(last, {**checkpoint, 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                               'history': history, 'best': best})
        write_json(args.output / 'history.json', {'config': config, 'epochs': history, 'best_auroc': best})
        print(summary, flush=True)
    write_json(args.output / 'result.json', {'config': config, 'epochs': history,
                                            'best_sha256': checksum(args.output / 'best.pt'),
                                            'validation_sha256': checksum(args.output / 'validation.npz')})


def main() -> None:
    parser = argparse.ArgumentParser(description='Train high-resolution models on the locked radiography cohorts')
    parser.add_argument('task', choices=('chest', 'mammo'))
    parser.add_argument('--backbone', required=True,
                        choices=('chex', 'imagenet', 'chest-reference', 'effb0', 'r18', 'r50', 'mammo-clip'))
    for name in ('cohort', 'pixels', 'weights', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--resolution', type=int, required=True)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--seed', type=int, default=4403)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--lr', type=float, default=7e-5)
    parser.add_argument('--head-lr', type=float, default=4e-4)
    parser.add_argument('--class-power', type=float, default=.35)
    parser.add_argument('--auxiliary', type=float, default=0.)
    parser.add_argument('--dropout', type=float, default=.30)
    parser.add_argument('--focal-neg', type=float, default=0.)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--block', type=int, default=0, help='Shuffle within and among contiguous training blocks')
    parser.add_argument('--resident', action='store_true', help='Store development pixels in memory for faster shuffling')
    parser.add_argument('--affine', action='store_true', help='Apply aligned affine transforms to image and label')
    parser.add_argument('--pyramid', action='store_true', help='Fuse stride-8, -16, and -32 mammography features')
    args = parser.parse_args()
    if min(args.batch, args.epochs, args.accumulate, args.depth) < 1 or args.focal_neg < 0:
        parser.error('Invalid training size or focal exponent')
    if (args.task == 'chest') != (args.backbone in ('chex', 'imagenet', 'chest-reference')):
        parser.error('Choose a backbone intended for this task')
    if args.backbone == 'chest-reference' and args.parent is None:
        parser.error('Provide a trained chest reference checkpoint with --parent')
    if args.resident and args.block:
        parser.error('Use either resident development pixels or block shuffling')
    if (args.affine or args.pyramid) and args.task != 'mammo':
        parser.error('Affine targets and feature pyramids require mammography')
    fit(args)


if __name__ == '__main__':
    main()
