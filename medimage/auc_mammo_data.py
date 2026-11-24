from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from medimage.common import checksum, read_json, write_json
from medimage.radiography_data import read_image
from medimage.search_features import load_manifest

HEIGHT, WIDTH = 1024, 640
MAX_LESIONS = 24


@dataclass(frozen=True)
class Geometry:
    source: tuple[int, int]
    bounds: tuple[int, int, int, int]
    flipped: bool
    size: tuple[int, int]
    fallback: bool = False
    canvas: tuple[int, int] = (HEIGHT, WIDTH)

    def transform(self, image: np.ndarray, interpolation: int) -> np.ndarray:
        if image.shape != self.source:
            image = cv2.resize(image, self.source[::-1], interpolation=interpolation)
        top, bottom, left, right = self.bounds
        region = image[top:bottom, left:right]
        if self.flipped:
            region = region[:, ::-1]
        height, width = self.size
        output = np.zeros(self.canvas, dtype=image.dtype)
        output[:height, :width] = cv2.resize(region, (width, height), interpolation=interpolation)
        return output


def view_pixels(path: Path, canvas: tuple[int, int] = (HEIGHT, WIDTH)) -> tuple[np.ndarray, Geometry]:
    raw = read_image(path, cv2.IMREAD_GRAYSCALE if canvas[0] > HEIGHT else cv2.IMREAD_REDUCED_GRAYSCALE_2)
    return view_array(raw, canvas)


def view_array(raw: np.ndarray, canvas: tuple[int, int]) -> tuple[np.ndarray, Geometry]:
    height, width = canvas
    h, w = raw.shape
    pw, ph = 150, 240
    preview = cv2.resize(raw, (pw, ph), interpolation=cv2.INTER_AREA)
    smooth = cv2.GaussianBlur(preview, (5, 5), 0)
    threshold, _ = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    fallback = False
    for fraction, floor in ((.70, 6.), (.40, 2.), (.10, 1.)):
        coarse = (smooth > max(floor, threshold * fraction)).astype(np.uint8)
        n, components, stats, _ = cv2.connectedComponentsWithStats(coarse, connectivity=8)
        if n > 1:
            component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            x, y, dx, dy, area = stats[component]
            if area >= .06 * pw * ph:
                break
        fallback = True
    else:
        x, y, dx, dy = 0, 0, pw, ph
        components, component = np.ones((ph, pw), dtype=np.uint8), 1
    top, bottom = max(0, int(h * y / ph) - 12), min(h, int(np.ceil(h * (y + dy) / ph)) + 12)
    left, right = max(0, int(w * x / pw) - 12), min(w, int(np.ceil(w * (x + dx) / pw)) + 12)
    component_mask = cv2.resize((components == component).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    component_mask = cv2.dilate(component_mask, np.ones((7, 7), dtype=np.uint8))
    patch = component_mask[top:bottom, left:right]
    wing = max(1, patch.shape[1] // 7)
    flipped = bool(patch[:, -wing:].sum() > patch[:, :wing].sum())
    scale = min(height / (bottom - top), width / (right - left))
    size = (max(1, round((bottom - top) * scale)), max(1, round((right - left) * scale)))
    geometry = Geometry((h, w), (top, bottom, left, right), flipped, size, fallback, canvas)
    selected = raw[component_mask > 0]
    lower, upper = np.percentile(selected, (1., 99.5))
    if upper <= lower:
        lower, upper = float(selected.min()), float(selected.max())
    if upper <= lower:
        raise ValueError('Mammogram has insufficient intensity range')
    normalized = np.clip((raw.astype(np.float32) - lower) * (255. / (upper - lower)), 0, 255).astype(np.uint8)
    normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(normalized)
    normalized[component_mask == 0] = 0
    return geometry.transform(normalized, cv2.INTER_AREA), geometry


def _uid(path: str) -> str:
    return PurePosixPath(path.strip()).parent.name


def lesion_paths(root: Path, training_images: set[str]) -> tuple[dict[str, list], dict]:
    columns = ['image_path', 'SeriesDescription', 'SeriesInstanceUID', 'Rows', 'Columns']
    metadata = pd.read_csv(root / 'csv' / 'dicom_info.csv', usecols=columns, keep_default_na=False)
    series = defaultdict(list)
    for item in metadata.itertuples(index=False):
        if item.image_path.startswith('CBIS-DDSM/jpeg/'):
            series[item.SeriesInstanceUID].append(item)
    # The held-out lesion tables are never opened by this pipeline.
    records, audit = defaultdict(list), Counter()
    for kind, category in (('mass', 0), ('calc', 2)):
        file = root / 'csv' / f'{kind}_case_description_train_set.csv'
        for row in pd.read_csv(file).to_dict('records'):
            candidates = [item for item in series[_uid(row['image file path'])]
                          if item.image_path.removeprefix('CBIS-DDSM/') in training_images]
            if not candidates:
                continue
            if len(candidates) != 1:
                raise ValueError('An annotated training view maps to multiple full images')
            whole = candidates[0]
            full = whole.image_path.removeprefix('CBIS-DDSM/')
            audit[f'{kind}_training_rows'] += 1
            alternatives = {item.image_path: item for key in ('ROI mask file path', 'cropped image file path')
                            for item in series[_uid(row[key])]}
            matches = []
            for entry in alternatives.values():
                relative = entry.image_path.removeprefix('CBIS-DDSM/')
                ratios = (entry.Rows / whole.Rows, entry.Columns / whole.Columns)
                if (entry.SeriesDescription.strip().lower() == 'roi mask images'
                        and min(ratios) > .8 and max(ratios) < 1.2
                        and abs(ratios[0] / ratios[1] - 1.) < .025 and (root / relative).is_file()):
                    matches.append(relative)
            if len(matches) == 1:
                positive = row['pathology'] == 'MALIGNANT'
                records[full].append((matches[0], category + int(positive)))
                audit[f'{kind}_matched'] += 1
            else:
                records[full].append((None, -1))
                audit[f'{kind}_ambiguous_or_absent'] += 1
    return records, dict(audit)


def prepare(root: Path, cohort: Path, output: Path, workers: int, height: int = HEIGHT) -> None:
    cv2.setNumThreads(1)
    width = height * 5 // 8
    grid_height, grid_width = height // 32, width // 32
    frame, audit = load_manifest(cohort)
    rows = len(frame)
    training = set(frame.loc[frame.split.eq('train'), 'CC']) | set(frame.loc[frame.split.eq('train'), 'MLO'])
    training.discard('')
    signature = {'cohort': audit['manifest_sha256'], 'shape': [rows, 2, height, width],
                 'version': 1, 'preprocess_sha256': checksum(Path(__file__))}
    output.mkdir(parents=True, exist_ok=True)
    final, progress = output / 'audit.json', output / 'progress.json'
    if final.is_file():
        if read_json(final)['config'] != signature:
            raise ValueError('Mammography spatial cache has another configuration')
        print('Validated the mammography spatial cache', flush=True)
        return
    masks, mapping_audit = lesion_paths(root, training)
    start = 0
    stats = Counter()
    if progress.is_file():
        record = read_json(progress)
        if record['config'] != signature:
            raise ValueError('Mammography preprocessing changed during a resumed run')
        start = record['rows']
        stats.update(record['statistics'])
    layout = {'pixels': ((rows, 2, height, width), np.uint8),
              'heatmaps': ((rows, 2, 4, grid_height, grid_width), np.uint8),
              'annotation_available': ((rows, 2), np.bool_),
              'lesions': ((rows, 2, MAX_LESIONS, 5), np.int16),
              'views': ((rows, 2), np.bool_)}
    arrays = {key: np.lib.format.open_memmap(output / f'{key}.npy', mode='r+' if start else 'w+', dtype=dtype, shape=shape)
              for key, (shape, dtype) in layout.items()}

    def sample(row):
        pixels = np.zeros((2, height, width), np.uint8)
        heatmaps = np.zeros((2, 4, grid_height, grid_width), np.uint8)
        boxes = np.full((2, MAX_LESIONS, 5), -1, np.int16)
        enabled = np.zeros(2, dtype=bool)
        observed = np.zeros(2, dtype=bool)
        errors = Counter()
        for view, path in enumerate((row.CC, row.MLO)):
            if not path:
                continue
            pixels[view], geometry = view_pixels(root / path, (height, width))
            errors['foreground_fallback_views'] += int(geometry.fallback)
            observed[view] = True
            if row.split != 'train':
                continue
            eligible = masks.get(path, [])
            if len(eligible) > MAX_LESIONS:
                raise ValueError('Increase the training-view annotation capacity')
            complete = bool(eligible)
            for index, (mask, category) in enumerate(eligible):
                if mask is None:
                    complete = False
                    continue
                raw_mask = (read_image(root / mask, cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
                region = geometry.transform(raw_mask, cv2.INTER_AREA) > .12
                ys, xs = np.nonzero(region)
                if len(xs) < 4 or len(xs) > height * width * .4:
                    errors['empty_or_oversized_mask'] += 1
                    complete = False
                    continue
                boxes[view, index] = [category, xs.min(), ys.min(), xs.max(), ys.max()]
                heatmap = region.reshape(grid_height, 32, grid_width, 32).any(axis=(1, 3))
                heatmaps[view, category] |= heatmap
                errors['mapped_lesions'] += 1
            enabled[view] = complete
            errors['supervised_views' if complete else 'incomplete_training_views'] += 1
        return (pixels, heatmaps, enabled, boxes, observed), errors

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for offset in tqdm(range(start, rows, 20), mininterval=20, desc='Mammography geometry'):
            end = min(offset + 20, rows)
            for index, (values, errors) in enumerate(executor.map(sample, frame.iloc[offset:end].itertuples(index=False)), offset):
                for array, value in zip(arrays.values(), values, strict=True):
                    array[index] = value
                stats.update(errors)
            for array in arrays.values():
                array.flush()
            write_json(progress, {'config': signature, 'rows': end, 'statistics': dict(stats)})
    held = frame.split.ne('train').to_numpy()
    if arrays['annotation_available'][held].any() or arrays['heatmaps'][held].any() or (arrays['lesions'][held] >= 0).any():
        raise AssertionError('A held-out mammogram received an annotation tensor')
    write_json(final, {'config': signature, 'mapping': mapping_audit, 'statistics': dict(stats),
                       'used_roi_tables': ['mass_case_description_train_set.csv', 'calc_case_description_train_set.csv']})
    progress.unlink(missing_ok=True)
    print({'mapping': mapping_audit, 'spatial': dict(stats)}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Align full mammograms with training-only lesion masks')
    for name in ('root', 'cohort', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--height', type=int, choices=(1024, 1536), default=1024)
    args = parser.parse_args()
    prepare(args.root, args.cohort, args.output, args.workers, args.height)


if __name__ == '__main__':
    main()
