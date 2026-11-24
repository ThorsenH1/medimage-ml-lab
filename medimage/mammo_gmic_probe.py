"""Evaluate published external GMIC weights on development mammograms."""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.special import logit
from tqdm import tqdm

from medimage.auc_mammo_data import view_pixels
from medimage.common import checksum, load_manifest, write_json
from medimage.radiography_data import read_image
from medimage.radiography_evaluate import validation_metrics

SHAPE = (2944, 1920)
PERCENTILES = (.02, .03, .03, .05, .10)


def gmic_view(path: Path, projection: str) -> np.ndarray:
    _, geometry = view_pixels(path, SHAPE)
    source = read_image(path, cv2.IMREAD_GRAYSCALE)
    region = geometry.transform(source, cv2.INTER_AREA)
    height, width = geometry.size
    image = np.zeros(SHAPE, dtype=np.float32)
    top = (SHAPE[0] - height) // 2 if projection == 'CC' else 0
    image[top:top + height, :width] = region[:height, :width]
    image -= image.mean()
    image /= max(float(image.std()), 1e-5)
    return image


def model_for(source: Path, checkpoint: Path, number: int) -> torch.nn.Module:
    sys.path.insert(0, str(source.resolve()))
    try:
        from src.modeling.gmic import GMIC
    finally:
        sys.path.pop(0)
    config = {'device_type': 'gpu', 'gpu_number': 0, 'cam_size': (46, 30), 'K': 6,
              'crop_shape': (256, 256), 'post_processing_dim': 256, 'num_classes': 2,
              'use_v1_global': False, 'percent_t': PERCENTILES[number - 1]}
    model = GMIC(config)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state.pop('shared_rep_filter.weight', None)
    model.load_state_dict(state)
    return model.cuda().eval()


@torch.inference_mode()
def extract(args: argparse.Namespace) -> None:
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    frame, audit = load_manifest(args.cohort)
    rows = np.flatnonzero(frame.split.eq('val').to_numpy())
    if args.limit:
        rows = rows[:args.limit]
    selected = frame.iloc[rows]
    views = [(position, column, projection, args.root / image)
             for position, case in enumerate(selected.itertuples(index=False))
             for column, projection in enumerate(('CC', 'MLO'))
             if (image := getattr(case, projection))]
    checkpoint = args.source / 'models' / f'sample_model_{args.member}.p'
    model = model_for(args.source, checkpoint, args.member)
    values = np.zeros((len(rows), 2, 2), dtype=np.float32)
    available = np.zeros((len(rows), 2), dtype=bool)

    def prepare(view):
        return gmic_view(view[-1], view[2])

    with ThreadPoolExecutor(3) as threads:
        for start in tqdm(range(0, len(views), args.batch), desc='GMIC whole views', mininterval=12):
            chunk = views[start:start + args.batch]
            pixels = np.stack(list(threads.map(prepare, chunk)))
            output = model(torch.from_numpy(pixels[:, None]).cuda().contiguous(memory_format=torch.channels_last))
            output = output.float().cpu().numpy()
            if not np.isfinite(output).all():
                raise ValueError('The external mammography model produced nonfinite scores')
            for (position, column, _, _), scores in zip(chunk, output, strict=True):
                values[position, column] = scores
                available[position, column] = True
    if not available.any(axis=1).all():
        raise ValueError('Every breast must have at least one whole projection')
    truth = selected.positive.to_numpy()[:, None]
    members = logit(values[:, :, 1].clip(1e-6, 1 - 1e-6))
    mean = (members * available).sum(axis=1) / available.sum(axis=1)
    maximum = np.where(available, members, -np.inf).max(axis=1)
    result = {'cohort_sha256': audit['manifest_sha256'], 'member': args.member, 'weight_sha256': checksum(checkpoint),
              'partition': 'validation', 'breasts': len(rows), 'projections': len(views),
              'mean_logit': validation_metrics(truth, mean[:, None]),
              'max_logit': validation_metrics(truth, maximum[:, None])}
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / 'validation.npz', rows=rows, available=available, probabilities=values)
    write_json(args.output / 'result.json', result)
    print(result, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source', 'cohort', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--member', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--batch', type=int, default=3)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.batch < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Batch size and row limit must be positive')
    extract(args)


if __name__ == '__main__':
    main()
