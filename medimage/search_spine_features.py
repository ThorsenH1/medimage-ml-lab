from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from medimage.common import checksum, read_json, write_json
from medimage.search_features import Encoder
from medimage.spine_localize import (SpineCache, fit_localizer, image_features, make_regions, predict_keypoints,
                                     read_folds)
from medimage.spine_model import crop_regions

TEXTURE_DIM = 320
DEEP_DIM = 512


def feature_rows(encoder: Encoder, cache: SpineCache, rows: np.ndarray, regions: np.ndarray) -> np.ndarray:
    cv2.setNumThreads(1)
    patches = np.stack([crop_regions(cache.volumes[row], region) for row, region in zip(rows, regions, strict=True)])
    pixels = torch.from_numpy(patches).cuda()
    flat = pixels.flatten(0, 1)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        deep = encoder(flat).float()
        normalized = flat.float() / 255.
        cells = F.adaptive_avg_pool2d(normalized, 8).flatten(1)
        middle = normalized[:, 1:2]
        dx = F.adaptive_avg_pool2d(middle.diff(dim=-1).abs(), 8).flatten(1)
        dy = F.adaptive_avg_pool2d(middle.diff(dim=-2).abs(), 8).flatten(1)
        result = torch.cat([deep, cells, dx, dy], dim=-1).view(len(rows), 30, DEEP_DIM + TEXTURE_DIM)
    values = result.half().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite MRI ROI descriptors')
    return values


def inner_regions(cache: SpineCache, folds: np.ndarray, distances: np.ndarray, anchors: np.ndarray,
                  outer: int, regular: dict, fits: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inner = (outer + 1) % 5
    rows = np.flatnonzero(~np.isin(folds, [outer, inner]))
    output = regular['region'][rows].copy()
    metadata = regular['metadata'][rows].copy()
    provenance = np.zeros((len(rows), 5), dtype=bool)
    for source in sorted(set(folds[rows])):
        excluded = tuple(sorted({outer, inner, source}))
        eligible = np.flatnonzero(~np.isin(folds, excluded))
        if excluded not in fits:
            fits[excluded] = fit_localizer(eligible, distances, anchors, cache.points)
        model = fits[excluded]
        take = np.flatnonzero(folds[rows] == source)
        target = rows[take]
        xy, confidence, _ = predict_keypoints(distances[np.ix_(target, eligible)], model)
        output[take], metadata[take], _ = make_regions(cache, target, anchors, xy, confidence, model)
        provenance[np.ix_(take, np.unique(folds[eligible]))] = True
    expected = (~np.isin(np.arange(5), [outer, inner])[None]) & (np.arange(5)[None] != folds[rows, None])
    if not np.array_equal(provenance, expected):
        raise AssertionError('Inner MRI regions use an unavailable fold')
    return rows, output, metadata, provenance


def extract(cache_path: Path, folds_path: Path, localization: Path, weights: Path, output: Path, batch: int) -> None:
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    cache = SpineCache.load(cache_path)
    folds = read_folds(folds_path, cache.studies)
    loc = read_json(localization / 'localization.json')
    if loc['config']['folds_file_sha256'] != checksum(folds_path) or loc['config']['cache_sha256'] != checksum(cache_path / 'audit.json'):
        raise ValueError('MRI crop audit does not match the input stores')
    encoder = Encoder('r18', 96, weights).cuda().eval()
    features, anchors = image_features(cache)
    distances = cdist(features, features, 'sqeuclidean').astype(np.float32)
    all_rows = np.arange(len(folds))
    fits = {}
    with threadpool_limits(limits=4):
        for outer in range(5):
            directory = output / f'fold-{outer}'
            directory.mkdir(parents=True, exist_ok=True)
            source = localization / f'fold-{outer}.npz'
            signature = {'fold': outer, 'inner_validation_fold': (outer + 1) % 5, 'manifest': checksum(folds_path),
                         'crop_sha256': checksum(source), 'pretrained_sha256': encoder.weight_sha256, 'version': 1,
                         'encoder': 'r18', 'resolution': 96, 'deep_dim': DEEP_DIM, 'texture_dim': TEXTURE_DIM}
            complete = directory / 'features.json'
            if complete.is_file():
                if read_json(complete) != signature:
                    raise ValueError('MRI profile conflicts with existing features')
                continue
            with np.load(source, allow_pickle=False) as item:
                regular = dict(item)
            if not np.array_equal(regular['studies'], cache.studies) or not np.array_equal(regular['fold_per_study'], folds):
                raise ValueError('MRI crop ordering differs from frozen studies')
            expected = (np.arange(5)[None] != folds[:, None]) & (np.arange(5)[None] != outer)
            if not np.array_equal(regular['trained_on'], expected):
                raise ValueError('Final MRI features are not based on nested held-out ROIs')
            inner_rows, nested, inner_meta, provenance = inner_regions(cache, folds, distances, anchors, outer, regular, fits)
            np.savez_compressed(directory / 'geometry.npz', studies=cache.studies, inner_rows=inner_rows,
                                inner_meta=inner_meta, outer_meta=regular['metadata'], inner_provenance=provenance,
                                outer_provenance=regular['trained_on'])
            for name, rows, regions in (('outer', all_rows, regular['region']), ('inner', inner_rows, nested)):
                progress = directory / f'{name}-progress.json'
                start = 0
                if progress.is_file():
                    started = read_json(progress)
                    if started['signature'] != signature:
                        raise ValueError('Cannot resume changed MRI feature profile')
                    start = started['rows']
                tensor = np.lib.format.open_memmap(directory / f'{name}.npy', dtype=np.float16,
                                                   mode='r+' if start else 'w+', shape=(len(rows), 30, DEEP_DIM + TEXTURE_DIM))
                for offset in tqdm(range(start, len(rows), batch), mininterval=30, desc=f'MR {outer}/{name}'):
                    end = min(offset + batch, len(rows))
                    tensor[offset:end] = feature_rows(encoder, cache, rows[offset:end], regions[offset:end])
                    if (offset // batch) % 32 == 31 or end == len(rows):
                        tensor.flush()
                        write_json(progress, {'signature': signature, 'rows': end})
                del tensor
            write_json(complete, signature)
            print(f'MRI fold {outer} features ready, inner train {len(inner_rows)} studies', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract leakage-safe MRI feature stores for inner tuning and final fits')
    for name in ('cache', 'folds', 'localization', 'weights', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=10)
    args = parser.parse_args()
    extract(args.cache, args.folds, args.localization, args.weights, args.output, args.batch_size)


if __name__ == '__main__':
    main()
