"""Measure how often a frozen mammography component's strongest region overlaps a held-out mask."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from medimage.auc_mammo_data import _uid, view_pixels
from medimage.auc_selection import checked, load_component, pixels_location
from medimage.auc_train import store, transfer
from medimage.common import checksum, read_json, write_json
from medimage.radiography_data import read_image

KINDS = ('mass', 'calc')
METRICS = ('peak_overlap', 'tissue_center_overlap', 'uniform_tissue_expectation')


def test_masks(root: Path, test_views: set[str]) -> tuple[dict, dict, dict]:
    columns = ['image_path', 'SeriesDescription', 'SeriesInstanceUID', 'Rows', 'Columns']
    metadata = root / 'csv' / 'dicom_info.csv'
    table = pd.read_csv(metadata, usecols=columns, keep_default_na=False)
    table = table[table.image_path.str.startswith('CBIS-DDSM/')].copy()
    table['path'] = table.image_path.str.removeprefix('CBIS-DDSM/')
    series = {uid: list(rows.itertuples(index=False)) for uid, rows in table.groupby('SeriesInstanceUID')}
    references, counts = defaultdict(list), Counter()
    hashes = {metadata.name: checksum(metadata)}
    for kind in KINDS:
        file = root / 'csv' / f'{kind}_case_description_test_set.csv'
        hashes[file.name] = checksum(file)
        for row in pd.read_csv(file).to_dict('records'):
            original = series.get(_uid(row['image file path']), ())
            full = [entry for entry in original if entry.path in test_views]
            if not full:
                counts[f'{kind}_rows_outside_cohort'] += 1
                continue
            if len(full) != 1:
                raise ValueError('An official-test annotation matched multiple cohort projections')
            target = full[0]
            alternatives = {entry.path: entry for key in ('ROI mask file path', 'cropped image file path')
                            for entry in series.get(_uid(row[key]), ())}
            masks = []
            for relative, entry in alternatives.items():
                vertical, horizontal = entry.Rows / target.Rows, entry.Columns / target.Columns
                if (entry.SeriesDescription.strip().lower() == 'roi mask images'
                        and .8 < vertical < 1.2 and .8 < horizontal < 1.2
                        and abs(vertical / horizontal - 1.) < .025 and (root / relative).is_file()):
                    masks.append(relative)
            references[target.path].append((kind, masks[0] if len(masks) == 1 else None))
            counts[f'{kind}_unique_masks' if len(masks) == 1 else f'{kind}_unmatched_masks'] += 1
    return references, dict(counts), hashes


def grid_evidence(maps: np.ndarray, pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    evidence = maps.max(axis=0)
    if not np.isfinite(evidence).all():
        raise ValueError('Invalid lesion-head score')
    size = evidence.shape[::-1]
    tissue = cv2.resize((pixels > 0).astype(np.float32), size, interpolation=cv2.INTER_AREA) >= .25
    visible = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0
    indices = np.argwhere(tissue)
    if not len(indices):
        raise ValueError('No tissue cells in a mapped test projection')
    center = tuple(indices[np.square(indices - indices.mean(axis=0)).sum(axis=1).argmin()])
    peak = np.unravel_index(np.where(tissue, evidence, -np.inf).argmax(), evidence.shape)
    return np.array([visible[peak], visible[center], visible[tissue].mean()], dtype=np.float64)


def averages(rows: list, repeats: int) -> dict:
    patients, values = zip(*rows, strict=True)
    values = np.asarray(values)
    _, group = np.unique(patients, return_inverse=True)
    count = np.bincount(group)
    sums = np.column_stack([np.bincount(group, weights=metric) for metric in values.T])
    rng = np.random.default_rng(5714)
    draws = rng.integers(len(count), size=(repeats, len(count)))
    rates = sums[draws].sum(axis=1) / count[draws].sum(axis=1, keepdims=True)
    return {'projections': len(values), 'patients': len(count),
            **dict(zip(METRICS, values.mean(axis=0).tolist(), strict=True)),
            'peak_overlap_95': np.quantile(rates[:, 0], [.025, .975]).tolist(),
            'peak_minus_uniform_95': np.quantile(rates[:, 0] - rates[:, 2], [.025, .975]).tolist()}


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    lock = read_json(args.selection)
    candidate = lock['candidates'][args.component]
    if lock['task'] != 'mammo' or candidate['config'].get('auxiliary', 0) <= 0:
        raise ValueError('Choose a frozen mammography component trained with spatial supervision')
    frame, audit, arrays, signature = store('mammo', args.cohort, pixels_location(args, candidate['pixels']))
    if lock['manifest'] != audit['manifest_sha256'] or signature != candidate['config']['pixel_sha256']:
        raise ValueError('The audit requires the frozen cohort and pixel cache')
    if arrays['pixels'].dtype != np.uint8:
        raise ValueError('This audit expects the aligned JPEG cache')
    positions = np.flatnonzero(frame.split.eq('test'))
    held_out = frame.iloc[positions]
    test_views = (set(held_out.CC) | set(held_out.MLO)) - {''}
    references, mapping, input_hashes = test_masks(args.root, test_views)
    model, _ = load_component(args.work, candidate, args.weights, args.clip)
    canvas = arrays['pixels'].shape[-2:]
    observations, skipped = defaultdict(list), Counter()

    def align(item):
        slot, view, path = item
        pixels, geometry = view_pixels(checked(args.root, path), canvas)
        union = np.zeros((2, *canvas), dtype=bool)
        counts = Counter()
        for kind, mask in references[path]:
            if mask is None:
                continue
            raw = (read_image(checked(args.root, mask), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
            aligned = geometry.transform(raw, cv2.INTER_AREA) > .12
            if 4 <= aligned.sum() <= aligned.size * .4:
                union[KINDS.index(kind)] |= aligned
            else:
                counts[f'{kind}_invalid_aligned_masks'] += 1
        return slot, view, pixels, union, counts

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(positions), args.batch):
            rows = positions[start:start + args.batch]
            batch = frame.iloc[rows]
            items = [(slot, view, path) for slot, row in enumerate(batch.itertuples(index=False))
                     for view, path in enumerate((row.CC, row.MLO)) if path in references]
            with torch.autocast('cuda', dtype=torch.float16):
                maps = model(transfer(arrays['pixels'], rows), transfer(arrays['views'], rows))[1].cpu().numpy()
            for slot, view, pixels, union, counts in pool.map(align, items):
                skipped.update(counts)
                if not np.array_equal(pixels, arrays['pixels'][rows[slot], view]):
                    raise ValueError('The audit geometry does not reproduce the frozen input')
                patient = batch.iloc[slot].patient
                if union.any():
                    observations['all'].append((patient, grid_evidence(maps[slot, view], pixels, union.any(axis=0))))
                for kind, region in zip(KINDS, union, strict=True):
                    if region.any():
                        observations[kind].append((patient, grid_evidence(maps[slot, view], pixels, region)))
            if (start // args.batch + 1) % 25 == 0:
                print(f'Processed {min(start + args.batch, len(positions))} of {len(positions)} test breasts', flush=True)
    if not all(observations[key] for key in ('all', *KINDS)):
        raise ValueError('Both lesion types need matched masks to report the audit')
    report = {'selection_sha256': checksum(args.selection), 'checkpoint_sha256': candidate['checkpoint_sha256'],
              'mask_source_sha256': input_hashes, 'test_views': len(test_views), 'mapping': mapping,
              'invalid_aligned_masks': dict(skipped), 'bootstrap_repeats': args.bootstrap,
              'method': ('Maximum raw logit across all four lesion channels, choosing one cell with at least 25% tissue. '
                         'Any aligned mask pixel in that cell counts as an overlap. No flip pooling.'),
              'results': {key: averages(rows, args.bootstrap) for key, rows in observations.items()}}
    write_json(args.output, report)
    print(report['results'], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cohort', 'selection', 'work', 'weights', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    parser.add_argument('--clip', type=Path)
    parser.add_argument('--component', default='view-supervised')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.set_defaults(task='mammo')
    args = parser.parse_args()
    if min(args.batch, args.workers, args.bootstrap) < 1:
        parser.error('Use positive batch, worker and bootstrap counts')
    evaluate(args)


if __name__ == '__main__':
    main()
