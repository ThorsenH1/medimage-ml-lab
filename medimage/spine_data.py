from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

from medimage.common import checksum, read_json, write_json

LEVELS = ('l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1')
CONDITIONS = ('spinal_canal_stenosis', 'left_neural_foraminal_narrowing',
              'right_neural_foraminal_narrowing', 'left_subarticular_stenosis',
              'right_subarticular_stenosis')
TARGETS = tuple(f'{condition}_{level}' for condition in CONDITIONS for level in LEVELS)
SERIES = ('Sagittal T2/STIR', 'Sagittal T1', 'Axial T2')
DEPTHS = (9, 9, 24)
STARTS = np.r_[0, np.cumsum(DEPTHS)]
SIZE = 160
TARGET_VIEWS = np.repeat([0, 1, 1, 2, 2], 5)
SEVERITIES = ('normal_mild', 'moderate', 'severe')
LABEL_MAP = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}


@dataclass(frozen=True)
class Slice:
    path: Path
    instance: int
    origin: np.ndarray
    column_step: np.ndarray
    row_step: np.ndarray
    rows: int
    columns: int

    def location(self, x: float, y: float) -> np.ndarray:
        return self.origin + self.column_step * x + self.row_step * y


def read_slice(path: Path) -> Slice:
    tags = ['ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing', 'InstanceNumber', 'Rows', 'Columns']
    item = pydicom.dcmread(path, specific_tags=tags, stop_before_pixels=True)
    position = np.asarray(item.ImagePositionPatient, dtype=np.float64)
    orientation = np.asarray(item.ImageOrientationPatient, dtype=np.float64).reshape(2, 3)
    row_spacing, column_spacing = map(float, item.PixelSpacing)
    if (not np.isfinite(np.r_[position, orientation.ravel(), row_spacing, column_spacing]).all()
            or min(row_spacing, column_spacing, item.Rows, item.Columns) <= 0
            or not np.allclose(orientation @ orientation.T, np.eye(2), atol=.02)):
        raise ValueError('Invalid MRI geometry')
    return Slice(path, int(item.InstanceNumber), position, orientation[0] * column_spacing,
                 orientation[1] * row_spacing, int(item.Rows), int(item.Columns))


def read_pixels(header: Slice) -> tuple[np.ndarray, np.ndarray]:
    item = pydicom.dcmread(header.path)
    image = item.pixel_array.astype(np.float32)
    if image.shape != (header.rows, header.columns) or not np.isfinite(image).all():
        raise ValueError('Unexpected MRI pixel array')
    if str(getattr(item, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
        image = -image
    # Affine rescale is absorbed by percentile normalization unless its slope is negative.
    if float(getattr(item, 'RescaleSlope', 1)) < 0:
        image = -image
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        raise ValueError('MRI slice has no dynamic range')
    image = np.clip((image - low) * 255 / (high - low), 0, 255).astype(np.uint8)
    scale = SIZE / max(image.shape)
    width, height = max(1, round(header.columns * scale)), max(1, round(header.rows * scale))
    canvas = np.zeros((SIZE, SIZE), dtype=np.uint8)
    left, top = (SIZE - width) // 2, (SIZE - height) // 2
    canvas[top:top + height, left:left + width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    # OpenCV maps destination pixel centres to source pixel centres.
    x_ratio, y_ratio = header.columns / width, header.rows / height
    x_step, y_step = header.column_step * x_ratio, header.row_step * y_ratio
    origin = header.location((.5 - left) * x_ratio - .5, (.5 - top) * y_ratio - .5)
    return canvas, np.stack([origin, x_step, y_step]).astype(np.float32)


def project(point: np.ndarray, geometry: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project patient-space positions into a slice's continuous pixel plane."""
    offset = point - geometry[..., 0, :]
    x_step, y_step = geometry[..., 1, :], geometry[..., 2, :]
    xx = np.sum(x_step * x_step, axis=-1)
    yy = np.sum(y_step * y_step, axis=-1)
    xy = np.sum(x_step * y_step, axis=-1)
    rhs_x, rhs_y = np.sum(offset * x_step, axis=-1), np.sum(offset * y_step, axis=-1)
    determinant = xx * yy - xy * xy
    x = (rhs_x * yy - rhs_y * xy) / determinant
    y = (rhs_y * xx - rhs_x * xy) / determinant
    residual = offset - x[..., None] * x_step - y[..., None] * y_step
    return np.stack([x, y], axis=-1), np.linalg.norm(residual, axis=-1)


def anchor_slot(valid: np.ndarray) -> int:
    for view in (0, 1, 2):
        candidates = np.flatnonzero(valid[STARTS[view]:STARTS[view + 1]]) + STARTS[view]
        if len(candidates):
            return int(candidates[len(candidates) // 2])
    raise ValueError('Study has no decodable MR series')


def _series_key(headers: list[Slice], view: int) -> tuple[float, int]:
    centres = np.stack([info.location(info.columns / 2, info.rows / 2) for info in headers])
    if view == 2:
        return float(np.ptp(centres[:, 2])), len(headers)
    return float(len(headers)), len(headers)


def _make_study(study_path: Path, study_series: list[tuple[int, str]], coordinates: list[tuple]) -> tuple:
    volume = np.zeros((sum(DEPTHS), SIZE, SIZE), dtype=np.uint8)
    geometry = np.zeros((sum(DEPTHS), 3, 3), dtype=np.float32)
    valid = np.zeros(sum(DEPTHS), dtype=np.bool_)
    reference_points = np.full((25, 3), np.nan, dtype=np.float32)
    selected = np.full(3, -1, dtype=np.int64)
    issues = dict(header=0, pixels=0, unmatched_annotations=0, missing_view=0, extra_series=0)
    by_series = {}
    for series_id, _description in study_series:
        headers = []
        for path in (study_path / str(series_id)).glob('*.dcm'):
            try:
                headers.append(read_slice(path))
            except (ValueError, AttributeError, OSError, pydicom.errors.InvalidDicomError):
                issues['header'] += 1
        by_series[series_id] = headers
    for view, description in enumerate(SERIES):
        candidates = [(series_id, by_series[series_id]) for series_id, label in study_series
                      if label == description and by_series[series_id]]
        if not candidates:
            issues['missing_view'] += 1
            continue
        issues['extra_series'] += len(candidates) - 1
        series_id, headers = max(candidates, key=lambda row: (*_series_key(row[1], view), -row[0]))
        selected[view] = series_id
        # DICOM LPS coordinates define the same direction even if file order reverses.
        axis, sign = (2, -1) if view == 2 else (0, 1)
        headers = sorted(headers, key=lambda info: (sign * info.location(info.columns / 2, info.rows / 2)[axis],
                                                   info.instance, info.path.name))
        depth = min(DEPTHS[view], len(headers))
        sampled = np.rint(np.linspace(0, len(headers) - 1, depth)).astype(int)
        slots = np.rint(np.linspace(STARTS[view], STARTS[view + 1] - 1, depth)).astype(int)
        for slot, index in zip(slots, sampled, strict=True):
            try:
                volume[slot], geometry[slot] = read_pixels(headers[index])
                valid[slot] = True
            except (ValueError, RuntimeError, OSError, NotImplementedError, pydicom.errors.InvalidDicomError):
                issues['pixels'] += 1
        if not valid[STARTS[view]:STARTS[view + 1]].any():
            issues['missing_view'] += 1
    lookup = {(series_id, item.instance): item for series_id, items in by_series.items() for item in items}
    for series_id, instance, target, x, y in coordinates:
        header = lookup.get((series_id, instance))
        if header is None or not (0 <= x < header.columns and 0 <= y < header.rows):
            issues['unmatched_annotations'] += 1
        else:
            reference_points[target] = header.location(x, y)
    anchor_slot(valid)
    return volume, geometry, valid, reference_points, selected, issues


def prepare(root: Path, output: Path, split: str, workers: int, *, limit: int | None = None) -> None:
    cv2.setNumThreads(1)
    output.mkdir(parents=True, exist_ok=True)
    descriptions_path = root / f'{split}_series_descriptions.csv'
    coord_file = root / 'train_label_coordinates.csv'
    digests = {descriptions_path.name: checksum(descriptions_path)}
    if split == 'train':
        digests.update({path.name: checksum(path) for path in (coord_file, root / 'train.csv')})
    metadata = pd.read_csv(descriptions_path)
    if set(metadata.series_description) - set(SERIES) or metadata.duplicated(['study_id', 'series_id']).any():
        raise ValueError('MRI series inventory is not valid')
    series = {int(study): list(group[['series_id', 'series_description']].itertuples(index=False, name=None))
              for study, group in metadata.groupby('study_id', sort=True)}
    coordinates = {}
    if split == 'train':
        points = pd.read_csv(coord_file)
        key = points.condition.str.lower().str.replace(' ', '_') + '_' + points.level.str.lower().str.replace('/', '_')
        points['target'] = key.map({name: index for index, name in enumerate(TARGETS)})
        if points.target.isna().any() or points.duplicated(['study_id', 'target']).any():
            raise ValueError('MRI annotations contain unrecognized or duplicated targets')
        coordinates = {int(study): list(group[['series_id', 'instance_number', 'target', 'x', 'y']]
                                        .itertuples(index=False, name=None))
                       for study, group in points.groupby('study_id', sort=True)}
    studies = np.array(list(series)[:limit], dtype=np.int64)
    signature = {'split': split, 'upstream_sha256': digests, 'studies': len(studies),
                 'size': SIZE, 'depths': list(DEPTHS), 'preprocess_sha256': checksum(Path(__file__))}
    audit_path = output / 'audit.json'
    if audit_path.exists():
        audit = read_json(audit_path)
        if audit['config'] != signature:
            raise ValueError('Spine cache configuration differs from the source')
        print('Found validated spine cache', flush=True)
        return
    shapes = {'volume': ((len(studies), sum(DEPTHS), SIZE, SIZE), np.uint8),
              'geometry': ((len(studies), sum(DEPTHS), 3, 3), np.float32),
              'valid': ((len(studies), sum(DEPTHS)), np.bool_),
              'points': ((len(studies), 25, 3), np.float32),
              'series': ((len(studies), 3), np.int64)}
    state_path = output / 'cache-progress.json'
    state = read_json(state_path) if state_path.exists() else {'config': signature, 'written': 0, 'issues': {}}
    if state['config'] != signature:
        raise ValueError('MRI cache configuration changed during preparation')
    completed = state['written']
    arrays = {name: np.lib.format.open_memmap(output / f'{name}.npy', mode='r+' if completed else 'w+',
                                             dtype=dtype, shape=shape)
              for name, (shape, dtype) in shapes.items()}

    def load(study):
        return _make_study(root / f'{split}_images' / str(study), series[study], coordinates.get(study, []))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        progress = tqdm(range(completed, len(studies), 16), desc=f'Prepare spine {split}', mininterval=9)
        for offset in progress:
            end = min(offset + 16, len(studies))
            for index, row in enumerate(pool.map(load, studies[offset:end]), offset):
                for name, array in arrays.items():
                    array[index] = row[list(shapes).index(name)]
                for key, count in row[-1].items():
                    state['issues'][key] = state['issues'].get(key, 0) + count
            for array in arrays.values():
                array.flush()
            state['written'] = end
            write_json(state_path, state)
    np.save(output / 'studies.npy', studies, allow_pickle=False)
    write_json(audit_path, {'config': signature, 'issues': state['issues'],
                            'decoded_slices': int(arrays['valid'].sum()),
                            'annotated_targets': int(np.isfinite(arrays['points']).all(axis=-1).sum())})
    state_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare a geometry-preserving lumbar MRI cache')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    prepare(args.root, args.output, args.split, args.workers, limit=args.limit)


if __name__ == '__main__':
    main()
