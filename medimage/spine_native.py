"""Lift the frozen lumbar MRI slices and crop coordinates to a finer pixel grid."""
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pydicom

from medimage.common import checksum, read_json, write_json
from medimage.spine_data import STARTS, project, read_slice
from medimage.spine_localize import SpineCache, read_folds


def layout(header, size):
    scale = size / max(header.rows, header.columns)
    width, height = max(1, round(header.columns * scale)), max(1, round(header.rows * scale))
    left, top = (size - width) // 2, (size - height) // 2
    x_ratio, y_ratio = header.columns / width, header.rows / height
    origin = header.location((.5 - left) * x_ratio - .5, (.5 - top) * y_ratio - .5)
    geometry = np.stack((origin, header.column_step * x_ratio, header.row_step * y_ratio)).astype(np.float32)
    return (left, top, width, height), geometry


def pixels_for(header, box, size):
    data = pydicom.dcmread(header.path)
    image = data.pixel_array.astype(np.float32)
    if image.shape != (header.rows, header.columns) or not np.isfinite(image).all():
        raise ValueError('Unexpected raw MRI pixel shape')
    if str(data.get('PhotometricInterpretation', '')) == 'MONOCHROME1':
        image = -image
    if float(data.get('RescaleSlope', 1)) < 0:
        image = -image
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        raise ValueError('MRI slice has no dynamic range')
    normal = np.clip((image - low) * 255 / (high - low), 0, 255).astype(np.uint8)
    left, top, width, height = box
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[top:top + height, left:left + width] = cv2.resize(normal, (width, height), interpolation=cv2.INTER_AREA)
    return canvas


def lift_study(row, cache, series, source, size):
    view_count, previous_size, _ = cache.volumes.shape[1:]
    volumes = np.zeros((view_count, size, size), dtype=np.uint8)
    mapped = np.zeros((view_count, 3, 3), dtype=np.float32)
    for view, selected in enumerate(series[row]):
        slots = np.flatnonzero(cache.valid[row, STARTS[view]:STARTS[view + 1]]) + STARTS[view]
        if not len(slots):
            continue
        folder = source / str(cache.studies[row]) / str(selected)
        planes = []
        for file in folder.glob('*.dcm'):
            try:
                header = read_slice(file)
            except (ValueError, AttributeError, OSError, pydicom.errors.InvalidDicomError):
                continue
            planes.append(header)
        if not planes:
            raise ValueError('A selected series has no readable DICOM geometry')
        axis, sign = (2, -1) if view == 2 else (0, 1)
        planes.sort(key=lambda header: (sign * header.location(header.columns / 2, header.rows / 2)[axis],
                                        header.instance, header.path.name))
        count = min(STARTS[view + 1] - STARTS[view], len(planes))
        order = np.rint(np.linspace(0, len(planes) - 1, count)).astype(int)
        indices = np.rint(np.linspace(STARTS[view], STARTS[view + 1] - 1, count)).astype(int)
        expected = dict(zip(indices, order, strict=True))
        reference = np.stack([layout(plane, previous_size)[1] for plane in planes])
        for slot in slots:
            reference_plane = cache.geometry[row, slot]
            difference = np.abs(reference - reference_plane).max(axis=(1, 2))
            candidates = np.flatnonzero(difference <= .0001)
            if len(candidates) > 1:
                # Two instances can share a plane. Recover the one represented in the frozen cache.
                candidates = [i for i in candidates if np.array_equal(
                    pixels_for(planes[i], layout(planes[i], previous_size)[0], previous_size), cache.volumes[row, slot])]
                if expected.get(slot) in candidates:
                    candidates = [expected[slot]]
            if len(candidates) != 1:
                raise ValueError('Cannot match a unique native DICOM to a frozen slice')
            plane = planes[candidates[0]]
            box, mapped[slot] = layout(plane, size)
            volumes[slot] = pixels_for(plane, box, size)
    return row, volumes, mapped


def prepare(args):
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    previous = SpineCache.load(args.base)
    audit = read_json(args.base / 'audit.json')
    source = args.root / f"{audit['config']['split']}_images"
    series = np.load(args.base / 'series.npy', mmap_mode='r', allow_pickle=False)
    if series.shape != (len(previous.studies), 3):
        raise ValueError('Selected series and frozen studies are misaligned')
    args.output.mkdir(parents=True, exist_ok=False)
    n = len(previous.studies)
    volume = np.lib.format.open_memmap(args.output / 'volume.npy', mode='w+', dtype=np.uint8,
                                      shape=(n, previous.volumes.shape[1], args.size, args.size))
    geometry = np.lib.format.open_memmap(args.output / 'geometry.npy', mode='w+', dtype=np.float32,
                                        shape=previous.geometry.shape)
    with ThreadPoolExecutor(args.workers) as pool:
        for start in range(0, n, 2 * args.workers):
            indices = range(start, min(start + 2 * args.workers, n))
            for row, pixels, mapped in pool.map(lambda index: lift_study(index, previous, series, source, args.size), indices):
                volume[row], geometry[row] = pixels, mapped
            if start % (18 * args.workers) == 0:
                print(f'Prepared native MRI slices for {min(start + 2 * args.workers, n)} / {n} studies', flush=True)
    volume.flush()
    geometry.flush()
    del volume, geometry
    for file in ('studies.npy', 'series.npy', 'points.npy', 'valid.npy'):
        shutil.copyfile(args.base / file, args.output / file)
    config = {**audit['config'], 'size': args.size, 'reprojected_from': checksum(args.base / 'audit.json'),
              'preprocess_sha256': checksum(Path(__file__))}
    write_json(args.output / 'audit.json', {**audit, 'config': config})
    print(f'Prepared {n} native-grid MRI studies', flush=True)


def reproject_regions(regions, old_geometry, new_geometry):
    output = regions.copy()
    valid = regions[..., 0] >= 0
    studies = np.where(valid)[0]
    slots = regions[..., 0][valid].astype(int)
    original = old_geometry[studies, slots]
    destination = new_geometry[studies, slots]
    yx = regions[..., 1:3][valid][..., None]
    world = original[:, 0] + (yx * original[:, 1:]).sum(axis=1)
    coords, distance = project(world, destination)
    if not np.isfinite(coords).all() or np.max(distance, initial=0.) > .01:
        raise ValueError('Native MRI geometry disagrees with a frozen region plane')
    output[..., 1:3][valid] = coords
    scale = np.linalg.norm(original[:, 1:], axis=-1) / np.linalg.norm(destination[:, 1:], axis=-1)
    output[..., 3:][valid] = regions[..., 3:][valid] * scale
    return output


def reproject(args):
    earlier = SpineCache.load(args.base)
    native = SpineCache.load(args.native)
    old_hash = checksum(args.base / 'audit.json')
    config = read_json(args.native / 'audit.json')['config']
    stamp = read_json(args.reference / 'localization.json')
    if (config.get('reprojected_from') != old_hash or stamp['config']['cache_sha256'] != old_hash
            or stamp['config']['folds_file_sha256'] != checksum(args.folds)
            or not np.array_equal(earlier.studies, native.studies) or not np.array_equal(earlier.valid, native.valid)):
        raise ValueError('Native MRI cache disagrees with the frozen localization')
    folds = read_folds(args.folds, native.studies)
    args.output.mkdir(parents=True, exist_ok=False)
    for fold in range(5):
        with np.load(args.reference / f'fold-{fold}.npz', allow_pickle=False) as source:
            contents = dict(source)
        if not np.array_equal(contents['studies'], native.studies) or not np.array_equal(contents['fold_per_study'], folds):
            raise ValueError('Source fold does not cover the same studies')
        contents['region'] = reproject_regions(contents['region'], earlier.geometry, native.geometry)
        np.savez_compressed(args.output / f'fold-{fold}.npz', **contents)
    stamp['config'].update(cache_sha256=checksum(args.native / 'audit.json'),
                           reprojected_from_cache_sha256=old_hash,
                           original_localization_sha256=checksum(args.reference / 'localization.json'))
    write_json(args.output / 'localization.json', stamp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='phase', required=True)
    raw = stages.add_parser('prepare')
    crops = stages.add_parser('reproject')
    for stage, names in ((raw, ('root', 'base', 'output')), (crops, ('base', 'native', 'reference', 'folds', 'output'))):
        for name in names:
            stage.add_argument(f'--{name}', type=Path, required=True)
    raw.add_argument('--size', type=int, choices=(224, 256, 320), default=256)
    raw.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.phase == 'prepare':
        if args.workers < 1:
            parser.error('The worker count must be positive')
        prepare(args)
    else:
        reproject(args)


if __name__ == '__main__':
    main()
