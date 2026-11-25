"""Keep native DICOM intensity precision through mammography alignment and resizing."""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pydicom

from medimage.auc_mammo_data import lesion_paths, view_array
from medimage.common import checksum, load_manifest, read_json, write_json
from medimage.radiography_data import read_image

PIXEL_FORMAT = 'native-dicom-u16-v1'


def native_view(path: Path, canvas: tuple[int, int], expected_series: str | None = None):
    dataset = pydicom.dcmread(path)
    if expected_series is not None and str(dataset.SeriesInstanceUID) != expected_series:
        raise ValueError('DICOM series does not match the acquisition registry')
    raw = dataset.pixel_array
    if (raw.ndim != 2 or raw.dtype.kind not in ('u', 'i') or dataset.BitsStored < 12
            or dataset.PhotometricInterpretation not in ('MONOCHROME1', 'MONOCHROME2')):
        raise ValueError('Expected a high-bit-depth grayscale mammogram')
    pixels = raw.astype(np.float32)
    if dataset.PhotometricInterpretation == 'MONOCHROME1':
        pixels *= -1
    low, high = np.percentile(pixels[::4, ::4], [.5, 99.9])
    if not high > low:
        raise ValueError('DICOM mammogram has no usable intensity range')
    pixels -= low
    pixels *= 1. / (high - low)
    np.clip(pixels, 0., 1., out=pixels)
    display, geometry = view_array(np.rint(255 * pixels).astype(np.uint8), canvas)
    aligned = geometry.transform(pixels, cv2.INTER_AREA)
    # Retain native intensity precision and a smaller contribution from the usual contrast enhancement.
    values = .75 * aligned + .25 * (display / 255.)
    values[display == 0] = 0.
    return np.rint(65535 * values).astype(np.uint16), geometry


def projection(path, registry, root, project_root, canvas, labels, is_training):
    if not path:
        return (np.zeros(canvas, np.uint16), np.zeros((4, canvas[0] // 32, canvas[1] // 32), np.uint8), False)
    entry = registry[path]
    folder = root / entry['split'] / entry['SeriesInstanceUID']
    files = list(folder.glob('*.dcm'))
    if len(files) != 1 or files[0].stat().st_size != round(entry['series_size_MB'] * 1_000_000):
        raise ValueError('Missing or incomplete DICOM in the frozen acquisition registry')
    pixels, geometry = native_view(files[0], canvas, entry['SeriesInstanceUID'])
    gh, gw = canvas[0] // 32, canvas[1] // 32
    heatmaps = np.zeros((4, gh, gw), dtype=np.uint8)
    lesions = labels.get(path, []) if is_training else []
    annotated = bool(lesions)
    for mask, category in lesions:
        if not mask:
            annotated = False
            continue
        values = (read_image(project_root / mask, cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        mapped = geometry.transform(values, cv2.INTER_AREA) > .12
        coverage = int(mapped.sum())
        if not 4 <= coverage < pixels.size * .4:
            annotated = False
            continue
        heatmaps[category] |= mapped.reshape(gh, 32, gw, 32).any(axis=(1, 3))
    return pixels, heatmaps, annotated


def prepare(args: argparse.Namespace) -> None:
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    frame, audit = load_manifest(args.cohort)
    plan = read_json(args.plan)
    if plan['cohort_sha256'] != audit['manifest_sha256']:
        raise ValueError('DICOM registry and frozen breast cohort differ')
    registry = {row['path']: row for row in plan['views']}
    expected = {path: row.split for row in frame.itertuples(index=False) for path in (row.CC, row.MLO) if path}
    if (len(registry) != len(plan['views']) or {key: row['split'] for key, row in registry.items()} != expected
            or len({row['SeriesInstanceUID'] for row in registry.values()}) != len(registry)
            or any(re.fullmatch(r'[0-9]+(?:\.[0-9]+)+', row['SeriesInstanceUID']) is None for row in registry.values())):
        raise ValueError('DICOM registry must cover each manifest projection exactly once with its original split')
    training = {path for path, split in expected.items() if split == 'train'}
    masks, review = lesion_paths(args.project_root, training)
    shape = (args.height, args.height * 5 // 8)
    signed = {'cohort': audit['manifest_sha256'], 'shape': [len(frame), 2, *shape], 'pixel_format': PIXEL_FORMAT,
              'acquisition_sha256': checksum(args.plan), 'preprocess_sha256': checksum(Path(__file__))}
    args.output.mkdir(parents=True, exist_ok=True)
    audit_file, progress_file = args.output / 'audit.json', args.output / 'progress.json'
    if audit_file.exists():
        if read_json(audit_file)['config'] != signed:
            raise ValueError('Native pixel cache belongs to another acquisition or recipe')
        print('Found validated native mammography cache', flush=True)
        return
    progress = read_json(progress_file) if progress_file.exists() else {'config': signed, 'rows': 0}
    if progress['config'] != signed:
        raise ValueError('The partial native cache belongs to another acquisition or recipe')
    start, n = progress['rows'], len(frame)
    layout = {'pixels': ((n, 2, *shape), np.uint16),
              'heatmaps': ((n, 2, 4, shape[0] // 32, shape[1] // 32), np.uint8),
              'annotation_available': ((n, 2), np.bool_)}
    arrays = {name: np.lib.format.open_memmap(args.output / f'{name}.npy', mode='r+' if start else 'w+',
                                              shape=dimensions, dtype=dtype)
              for name, (dimensions, dtype) in layout.items()}

    def encode(row):
        return [projection(path, registry, args.dicom, args.project_root, shape, masks, row.split == 'train')
                for path in (row.CC, row.MLO)]

    with ThreadPoolExecutor(args.workers) as pool:
        for position in range(start, n, 16):
            stop = min(position + 16, n)
            values = pool.map(encode, frame.iloc[position:stop].itertuples(index=False))
            for index, views in enumerate(values, position):
                for array, pair in zip(arrays.values(), zip(*views, strict=True), strict=True):
                    array[index] = pair
            for array in arrays.values():
                array.flush()
            write_json(progress_file, {'config': signed, 'rows': stop})
            if position == start or stop == n or stop % 128 == 0:
                print(f'Aligned {stop} / {n} native breasts', flush=True)
    held = frame.split.ne('train').to_numpy()
    if arrays['annotation_available'][held].any() or arrays['heatmaps'][held].any():
        raise AssertionError('A held-out native mammogram received a lesion annotation')
    np.save(args.output / 'views.npy', frame[['CC', 'MLO']].ne('').to_numpy(), allow_pickle=False)
    write_json(audit_file, {'config': signed, 'training_annotation_mapping': review})
    progress_file.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort', 'plan', 'dicom', 'project-root', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--height', type=int, choices=(1024, 1536), default=1536)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('The worker count must be positive')
    prepare(args)


if __name__ == '__main__':
    main()
