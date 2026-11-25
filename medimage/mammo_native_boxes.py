"""Project training lesion boxes onto an existing, fixed native mammography cache."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from medimage.auc_mammo_data import MAX_LESIONS, lesion_paths
from medimage.auc_selection import checked, save_scores
from medimage.common import checksum, load_manifest, read_json
from medimage.mammo_native import PIXEL_FORMAT, native_view
from medimage.radiography_data import read_image


def boxes_for_view(geometry, records, root: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    boxes = np.full((MAX_LESIONS, 5), -1, dtype=np.int16)
    height, width = geometry.canvas
    gh, gw = height // 32, width // 32
    heatmaps = np.zeros((4, gh, gw), dtype=bool)
    complete = bool(records)
    if len(records) > MAX_LESIONS:
        raise ValueError('Too many masks to store training boxes for this projection')
    for index, (mask, category) in enumerate(records):
        if not mask:
            complete = False
            continue
        raw = (read_image(checked(root, mask), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        aligned = geometry.transform(raw, cv2.INTER_AREA) > .12
        y, x = np.nonzero(aligned)
        if not 4 <= len(y) < height * width * .40:
            complete = False
            continue
        boxes[index] = category, x.min(), y.min(), x.max(), y.max()
        heatmaps[category] |= aligned.reshape(gh, 32, gw, 32).any(axis=(1, 3))
    return boxes, heatmaps, complete


def prepare(args: argparse.Namespace) -> None:
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    frame, audit = load_manifest(args.cohort)
    cache_audit = args.pixels / 'audit.json'
    original = read_json(cache_audit)
    config = original['config']
    plan = read_json(args.plan)
    cohort_digest = audit['manifest_sha256']
    if (config['cohort'] != cohort_digest or plan['cohort_sha256'] != cohort_digest
            or config['acquisition_sha256'] != checksum(args.plan) or config['pixel_format'] != PIXEL_FORMAT
            or config['preprocess_sha256'] != checksum(Path(__file__).with_name('mammo_native.py'))):
        raise ValueError('Use the acquisition registry and preprocessor that produced the frozen native cache')
    arrays = {name: np.load(args.pixels / f'{name}.npy', mmap_mode='r', allow_pickle=False)
              for name in ('pixels', 'heatmaps', 'annotation_available')}
    pixels = arrays['pixels']
    if list(pixels.shape) != config['shape'] or pixels.shape[:2] != (len(frame), 2) or pixels.dtype != np.uint16:
        raise ValueError('Unexpected native mammography cache shape or precision')
    canvas = pixels.shape[-2:]
    indices = np.flatnonzero(frame.split.eq('train').to_numpy())
    development = frame.iloc[indices]
    views = set(development.CC) | set(development.MLO)
    views.discard('')
    registry = {entry['path']: entry for entry in plan['views'] if entry['split'] == 'train'}
    if registry.keys() != views:
        raise ValueError('The native registry and training-projection lists differ')
    masks, review = lesion_paths(args.project_root, views)
    if review != original['training_annotation_mapping']:
        raise ValueError('Training lesion mappings differ from the signed pixel cache')
    sources = ('dicom_info.csv', 'mass_case_description_train_set.csv', 'calc_case_description_train_set.csv')
    signature = json.dumps({'pixel_sha256': checksum(cache_audit), 'cohort_sha256': cohort_digest,
                            'code_sha256': checksum(Path(__file__)),
                            'source_csv_sha256': {name: checksum(args.project_root / 'csv' / name) for name in sources}},
                           sort_keys=True)
    shape = (len(frame), 2, MAX_LESIONS, 5)
    progress = args.output.with_name(f'{args.output.stem}.progress.npz')
    start, boxes = 0, np.full(shape, -1, dtype=np.int16)
    resume = args.output if args.output.exists() else progress
    if resume.exists():
        with np.load(resume, allow_pickle=False) as saved:
            if saved['signature'].item() != signature or not np.array_equal(saved['rows'], indices):
                raise ValueError('The annotation output belongs to a different cache or training split')
            start, boxes = saved['finished'].item(), saved['boxes']
        if not 0 <= start <= len(indices) or boxes.shape != shape:
            raise ValueError('Invalid annotation progress')
        if resume == args.output:
            if start != len(indices) or (boxes[frame.split.ne('train')] >= 0).any():
                raise ValueError('The completed annotation sidecar contains an incorrect split')
            print('Found the complete native training boxes', flush=True)
            return

    def one(item):
        index, row = item
        result = np.full((2, MAX_LESIONS, 5), -1, dtype=np.int16)
        for view, path in enumerate((row.CC, row.MLO)):
            records = masks.get(path, ())
            if not records:
                if arrays['annotation_available'][index, view] or arrays['heatmaps'][index, view].any():
                    raise ValueError('The cache uses annotations outside the selected training sources')
                continue
            entry = registry[path]
            series = entry['SeriesInstanceUID']
            files = list(checked(args.dicom / 'train', series).glob('*.dcm'))
            if len(files) != 1 or files[0].stat().st_size != round(entry['series_size_MB'] * 1_000_000):
                raise ValueError('Missing or incomplete training projection in the acquisition registry')
            aligned_pixels, geometry = native_view(files[0], canvas, series)
            rects, heatmap, complete = boxes_for_view(geometry, records, args.project_root)
            if (not np.array_equal(pixels[index, view], aligned_pixels)
                    or not np.array_equal(arrays['heatmaps'][index, view], heatmap)
                    or arrays['annotation_available'][index, view] != complete):
                raise ValueError('Projected boxes do not reproduce the frozen view and mask geometry')
            result[view] = rects
        return result

    rows = list(zip(indices, development.itertuples(index=False), strict=True))
    with ThreadPoolExecutor(args.workers) as pool:
        for position in range(start, len(rows), 48):
            stop = min(position + 48, len(rows))
            boxes[indices[position:stop]] = list(pool.map(one, rows[position:stop]))
            save_scores(progress, boxes=boxes, rows=indices, signature=signature, finished=stop)
            print(f'Checked morphology masks for {stop} / {len(rows)} native training breasts', flush=True)
    if (boxes[frame.split.ne('train')] >= 0).any():
        raise ValueError('A held-out breast received a training box')
    counts = np.bincount(boxes[..., 0][boxes[..., 0] >= 0], minlength=4)
    if (counts == 0).any():
        raise ValueError('The native proposal model requires every training lesion category')
    save_scores(args.output, boxes=boxes, rows=indices, signature=signature, finished=len(indices))
    progress.unlink(missing_ok=True)
    print(f'Native training boxes by category: {counts.tolist()}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort', 'plan', 'dicom', 'project-root', 'pixels', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or args.output.suffix != '.npz':
        parser.error('Use at least one worker and an .npz output path')
    prepare(args)


if __name__ == '__main__':
    main()
