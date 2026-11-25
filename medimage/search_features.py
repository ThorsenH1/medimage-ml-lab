from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm

from medimage.common import checksum, load_manifest, read_json, write_json
from medimage.radiography_data import prepare_breast, read_image

BACKBONES = {
    'r18': 'resnet18.a1_in1k',
    'r50': 'resnet50.a1_in1k',
    'effb0': 'efficientnet_b0.ra_in1k',
    'dino': 'vit_small_patch14_dinov2.lvd142m',
}
MAMMO_SHAPE = (928, 576)
MAMMO_BOXES = tuple((y, x) for y in (0, 292, 584) for x in (0, 232))
MAMMO_TILE = 344


class Encoder(torch.nn.Module):
    def __init__(self, backbone: str, resolution: int, weight_root: Path) -> None:
        super().__init__()
        model_name = BACKBONES[backbone]
        directory = weight_root / ('models--timm--' + model_name)
        revision = (directory / 'refs' / 'main').read_text(encoding='utf-8').strip()
        self.weight_path = directory / 'snapshots' / revision / 'model.safetensors'
        self.weight_sha256 = checksum(self.weight_path)
        self.backbone = backbone
        self.resolution = resolution
        args = {'img_size': resolution} if backbone == 'dino' else {}
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0, **args)
        weights = load_file(self.weight_path)
        if backbone == 'dino' and weights['pos_embed'].shape != self.model.pos_embed.shape:
            from timm.layers import resample_abs_pos_embed

            weights['pos_embed'] = resample_abs_pos_embed(weights['pos_embed'], self.model.patch_embed.grid_size,
                                                          num_prefix_tokens=self.model.num_prefix_tokens)
        state = self.model.load_state_dict(weights, strict=False)
        if state.missing_keys or any(key.split('.')[0] not in {'fc', 'head', 'classifier'} for key in state.unexpected_keys):
            raise ValueError(f'Checkpoint state differs from expected {model_name} architecture: {state}')
        self.dim = self.model.num_features
        config = timm.data.resolve_model_data_config(self.model)
        self.register_buffer('mean', torch.tensor(config['mean'])[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(config['std'])[None, :, None, None], persistent=False)
        if backbone != 'dino':
            self.model.to(memory_format=torch.channels_last)

    def forward(self, pixels: torch.Tensor, spatial: bool = False) -> torch.Tensor:
        images = (pixels[:, None] if pixels.ndim == 3 else pixels).float().div(255.)
        if tuple(images.shape[-2:]) != (self.resolution, self.resolution):
            images = F.interpolate(images, size=(self.resolution,) * 2, mode='bilinear', align_corners=False, antialias=True)
        images = ((images - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
        features = self.model.forward_features(images)
        if self.backbone == 'dino':
            pooled = features[:, 0]
            h, w = self.model.patch_embed.grid_size
            patches = features[:, self.model.num_prefix_tokens:].transpose(1, 2).reshape(len(images), self.dim, h, w)
        else:
            patches = features
            pooled = patches.mean((-1, -2))
        if not spatial:
            return pooled
        spatial_tokens = F.adaptive_avg_pool2d(patches, 2).flatten(2).transpose(1, 2)
        return torch.cat([pooled[:, None], spatial_tokens], dim=1)


def mammography_tiles(images: torch.Tensor, resolution: int) -> torch.Tensor:
    size = (resolution, resolution)
    batch, views = images.shape[:2]
    flat = images.reshape(batch * views, 1, *images.shape[-2:]).float()
    tiles = [flat, *(flat[..., y:y + MAMMO_TILE, x:x + MAMMO_TILE] for y, x in MAMMO_BOXES)]
    comparable = [F.interpolate(tile, size=size, mode='bilinear', align_corners=False, antialias=True)[:, 0] for tile in tiles]
    return torch.stack(comparable, dim=1).reshape(batch, views, 7, *size)


def chest_pixels(image: np.ndarray, resolution: int) -> np.ndarray:
    return cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_AREA)


def chest_source(path: Path, resolution: int) -> np.ndarray:
    return chest_pixels(read_image(path, cv2.IMREAD_GRAYSCALE), resolution)


def prepare_pixels(task: str, raw: Path, cohort: Path, output: Path, workers: int, *, chest_size: int = 224) -> None:
    cv2.setNumThreads(1)
    frame, audit = load_manifest(cohort)
    output.mkdir(parents=True, exist_ok=True)
    shape = (len(frame), chest_size, chest_size) if task == 'chest' else (len(frame), 2, *MAMMO_SHAPE)
    signature = {'task': task, 'manifest': audit['manifest_sha256'], 'shape': list(shape), 'version': 1}
    state_path, complete_path = output / 'progress.json', output / 'pixels.json'
    if complete_path.is_file():
        if read_json(complete_path) != signature:
            raise ValueError('High-resolution cache belongs to a different cohort')
        print(f'Validated {task} high-resolution cache', flush=True)
        return
    start = 0
    if state_path.is_file():
        state = read_json(state_path)
        if state['signature'] != signature:
            raise ValueError('High-resolution preprocessing signature changed')
        start = state['rows']
    pixels = np.lib.format.open_memmap(output / 'pixels.npy', mode='r+' if start else 'w+', dtype=np.uint8, shape=shape)

    def read(row):
        if task == 'chest':
            return chest_source(raw / row.path, chest_size)
        return np.stack([prepare_breast(raw / name, MAMMO_SHAPE) if name else np.zeros(MAMMO_SHAPE, np.uint8)
                         for name in (row.CC, row.MLO)])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in tqdm(range(start, len(frame), 192), desc=f'Pixels {task}', mininterval=15):
            stop = min(offset + 192, len(frame))
            rows = frame.iloc[offset:stop].itertuples(index=False)
            for index, result in enumerate(pool.map(read, rows), offset):
                pixels[index] = result
            pixels.flush()
            write_json(state_path, {'signature': signature, 'rows': stop})
    write_json(complete_path, signature)
    state_path.unlink(missing_ok=True)
    print(f'{task} pixels ready {shape}', flush=True)


@torch.inference_mode()
def extract(task: str, cohort: Path, pixels_root: Path, output: Path, weight_root: Path,
            backbone: str, resolution: int, batch_size: int) -> None:
    torch.set_num_threads(4)
    frame, audit = load_manifest(cohort)
    high = read_json(pixels_root / 'pixels.json')
    if high['manifest'] != audit['manifest_sha256'] or high['task'] != task:
        raise ValueError('Pixel store and cohort disagree')
    images = np.load(pixels_root / 'pixels.npy', mmap_mode='r', allow_pickle=False)
    encoder = Encoder(backbone, resolution, weight_root).cuda().eval()
    positions = 5 if task == 'chest' else 14
    shape = (len(frame), positions, encoder.dim)
    signature = {'task': task, 'manifest': audit['manifest_sha256'], 'backbone': backbone, 'size': resolution,
                 'pretrained_sha256': encoder.weight_sha256, 'shape': list(shape), 'version': 1}
    output.mkdir(parents=True, exist_ok=True)
    state_path, done_path = output / 'progress.json', output / 'features.json'
    if done_path.is_file():
        if read_json(done_path) != signature:
            raise ValueError('Feature store provenance differs from requested encoder')
        print(f'Validated {task}/{backbone}-{resolution} embeddings', flush=True)
        return
    offset = 0
    if state_path.is_file():
        state = read_json(state_path)
        if state['signature'] != signature:
            raise ValueError('Embeddings preparation signature changed')
        offset = state['rows']
    result = np.lib.format.open_memmap(output / 'features.npy', dtype=np.float16,
                                      mode='r+' if offset else 'w+', shape=shape)
    mask = (frame[['CC', 'MLO']].ne('').to_numpy().repeat(7, axis=1) if task == 'mammo'
            else np.ones((len(frame), positions), dtype=np.bool_))
    torch.backends.cudnn.benchmark = False
    for start in tqdm(range(offset, len(frame), batch_size), desc=f'{task}/{backbone}-{resolution}', mininterval=30):
        end = min(start + batch_size, len(frame))
        raw = torch.from_numpy(images[start:end].copy()).cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            if task == 'chest':
                features = encoder(raw, spatial=True).float()
            else:
                tiles = mammography_tiles(raw, resolution).reshape(-1, resolution, resolution)
                use = torch.from_numpy(mask[start:end].reshape(-1)).cuda()
                valid = encoder(tiles[use]).float()
                flat = valid.new_zeros(len(tiles), encoder.dim)
                flat[use] = valid
                features = flat.reshape(len(raw), 14, encoder.dim)
        values = features.half().cpu().numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError('Pretrained embeddings are not finite')
        result[start:end] = values
        if (start // batch_size) % 64 == 63 or end == len(frame):
            result.flush()
            write_json(state_path, {'signature': signature, 'rows': end})
    np.save(output / 'mask.npy', mask, allow_pickle=False)
    write_json(done_path, signature)
    state_path.unlink(missing_ok=True)
    print(f'Finished {task}/{backbone}-{resolution} with shape {shape}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build high-resolution pixel and multi-backbone feature stores locally')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('pixels', 'extract'):
        stage = sub.add_parser(name)
        stage.add_argument('task', choices=('chest', 'mammo'))
        stage.add_argument('--cohort', type=Path, required=True)
        stage.add_argument('--output', type=Path, required=True)
    sub.choices['pixels'].add_argument('--root', type=Path, required=True)
    sub.choices['pixels'].add_argument('--workers', type=int, default=8)
    sub.choices['pixels'].add_argument('--chest-size', type=int, choices=(224, 320, 384, 448), default=224)
    feature = sub.choices['extract']
    for name in ('pixels', 'weights'):
        feature.add_argument(f'--{name}', type=Path, required=True)
    feature.add_argument('--backbone', choices=sorted(BACKBONES), required=True)
    feature.add_argument('--resolution', type=int, required=True)
    feature.add_argument('--batch-size', type=int, default=96)
    args = parser.parse_args()
    if args.command == 'pixels':
        prepare_pixels(args.task, args.root, args.cohort, args.output, args.workers, chest_size=args.chest_size)
    else:
        extract(args.task, args.cohort, args.pixels, args.output, args.weights, args.backbone, args.resolution, args.batch_size)


if __name__ == '__main__':
    main()
