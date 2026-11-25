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
METRICS = ('peak_overlap', 'tissue_center_overlap', 'uniform_tissue_expectation',
           'top_3_overlap', 'top_3_uniform_expectation', 'top_5_overlap', 'top_5_uniform_expectation')


def test_masks(root: Path, test_views: set[str], recover: bool = False) -> tuple[dict, dict, dict]:
    columns = ['image_path', 'SeriesDescription', 'SeriesInstanceUID', 'Rows', 'Columns', 'BitsStored',
               'SmallestImagePixelValue', 'LargestImagePixelValue']
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
            roi_series = _uid(row['ROI mask file path'])
            alternatives = {entry.path: entry for key in ('ROI mask file path', 'cropped image file path')
                            for entry in series.get(_uid(row[key]), ())}
            masks = []
            unlabeled = []
            for relative, entry in alternatives.items():
                vertical, horizontal = entry.Rows / target.Rows, entry.Columns / target.Columns
                description = entry.SeriesDescription.strip().lower()
                if not (root / relative).is_file():
                    continue
                if (description == 'roi mask images' and .8 < vertical < 1.2 and .8 < horizontal < 1.2
                        and abs(vertical / horizontal - 1.) < .025):
                    masks.append(relative)
                if (recover and kind == 'calc' and not description and entry.SeriesInstanceUID == roi_series
                        and str(entry.BitsStored) in ('8', '8.0')
                        and str(entry.SmallestImagePixelValue) in ('0', '0.0')
                        and str(entry.LargestImagePixelValue) in ('255', '255.0')):
                    if (entry.Rows, entry.Columns) == (target.Rows, target.Columns):
                        unlabeled.append(relative)
                    else:
                        counts['calc_unlabeled_dimension_mismatch'] += 1
            if len(masks) == 1:
                references[target.path].append((kind, masks[0], False))
                counts[f'{kind}_unique_masks'] += 1
            elif not masks and len(unlabeled) == 1:
                references[target.path].append((kind, unlabeled[0], True))
                counts['calc_unlabeled_candidates'] += 1
            else:
                counts[f'{kind}_unmatched_masks'] += 1
    return references, dict(counts), hashes


def plausible_unlabeled_mask(pixels: np.ndarray, expected_shape: tuple[int, int]) -> bool:
    if pixels.shape != expected_shape:
        return False
    histogram = cv2.calcHist([pixels], [0], None, [256], [0, 256]).ravel()
    extreme = histogram[:16].sum() + histogram[240:].sum()
    foreground = histogram[128:].sum() / pixels.size
    return extreme > .99 * pixels.size and 2e-7 <= foreground <= .25


def grid_points(maps: np.ndarray, pixels: np.ndarray) -> tuple:
    evidence = maps.max(axis=0)
    if not np.isfinite(evidence).all():
        raise ValueError('Invalid lesion-head score')
    size = evidence.shape[::-1]
    tissue = cv2.resize((pixels > 0).astype(np.float32), size, interpolation=cv2.INTER_AREA) >= .25
    indices = np.argwhere(tissue)
    if not len(indices):
        raise ValueError('No tissue cells in a mapped test projection')
    center = tuple(indices[np.square(indices - indices.mean(axis=0)).sum(axis=1).argmin()])
    top = indices[np.argsort(-evidence[tissue], kind='stable')[:5]]
    return tissue, center, top


def grid_evidence(points: tuple, mask: np.ndarray) -> np.ndarray:
    tissue, center, top = points
    visible = cv2.resize(mask.astype(np.float32), tissue.shape[::-1], interpolation=cv2.INTER_AREA) > 0
    count = int(tissue.sum())
    hits = int((tissue & visible).sum())
    reached = visible[top[:, 0], top[:, 1]]
    values = [reached[0], visible[center], hits / count]
    for size in (3, 5):
        draw = np.arange(min(size, count))
        uniform = 1 - np.prod((count - hits - draw).clip(min=0) / (count - draw))
        values.extend((reached[:size].any(), uniform))
    return np.asarray(values, dtype=np.float64)


def gallery(examples: dict, output: Path) -> None:
    from matplotlib.patches import Rectangle

    from medimage.portfolio_figures import plt, write_figure

    fig, axes = plt.subplots(4, 3, figsize=(10.1, 13.7), layout='constrained')
    fig.suptitle('Did the highest-scoring cell find the annotated cluster or mass?', x=.035,
                 ha='left', fontsize=15, fontweight='bold')
    for row, (kind, matched) in enumerate((kind, matched) for kind in KINDS for matched in (True, False)):
        pixels, mask, grid_shape, (peak_y, peak_x), view = examples[kind, matched]
        height, width = pixels.shape
        scale_x, scale_y = width / grid_shape[1], height / grid_shape[0]
        left, top = scale_x * peak_x - .5, scale_y * peak_y - .5
        y, x = np.nonzero(mask)
        windows = (None, (x.min(), y.min(), x.max(), y.max()), (left, top, left + scale_x, top + scale_y))
        name = 'Mass' if kind == 'mass' else 'Calcifications'
        outcome = 'overlap' if matched else 'miss'
        titles = (f'{name} · {outcome} · {view}', 'Dataset mask, zoomed in', 'Model cell, zoomed in')
        for axis, bounds, title in zip(axes[row], windows, titles, strict=True):
            axis.imshow(pixels, cmap='gray', vmin=0, vmax=255)
            axis.contour(mask, [.5], colors='#ffce52', linewidths=.9)
            axis.add_patch(Rectangle((left, top), scale_x, scale_y, fill=False,
                                     edgecolor='#54e4d0', linewidth=1.8))
            if bounds is not None:
                x0, y0, x1, y1 = bounds
                radius = max(74, .85 * max(x1 - x0, y1 - y0))
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                axis.set_xlim(cx - radius, cx + radius)
                axis.set_ylim(cy + radius, cy - radius)
            axis.set_title(title, loc='left', fontsize=10)
            axis.set_axis_off()
    fig.supxlabel('Gold outline = dataset mask    Turquoise square = the model’s top cell', fontsize=10)
    write_figure(fig, output)


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
    references, mapping, input_hashes = test_masks(args.root, test_views, args.recover_unlabeled_masks)
    model, _ = load_component(args.work, candidate, args.weights, args.clip)
    canvas = arrays['pixels'].shape[-2:]
    observations, skipped, examples = defaultdict(list), Counter(), {}

    def align(item):
        slot, view, path = item
        pixels, geometry = view_pixels(checked(args.root, path), canvas)
        union = np.zeros((2, *canvas), dtype=bool)
        counts = Counter()
        for kind, mask, recovered in references[path]:
            decoded = read_image(checked(args.root, mask), cv2.IMREAD_GRAYSCALE)
            if recovered and not plausible_unlabeled_mask(decoded, geometry.source):
                counts['calc_unlabeled_invalid_pixels'] += 1
                continue
            raw = (decoded > 127).astype(np.float32)
            aligned = geometry.transform(raw, cv2.INTER_AREA) > .12
            if 4 <= aligned.sum() <= aligned.size * .4:
                if recovered:
                    if (aligned & (pixels > 0)).sum() < .90 * aligned.sum():
                        counts['calc_unlabeled_misaligned'] += 1
                        continue
                    counts['calc_unlabeled_accepted'] += 1
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
                if not union.any():
                    continue
                points = grid_points(maps[slot, view], pixels)
                observations['all'].append((patient, grid_evidence(points, union.any(axis=0))))
                for kind, region in zip(KINDS, union, strict=True):
                    if region.any():
                        evidence = grid_evidence(points, region)
                        observations[kind].append((patient, evidence))
                        chosen = kind, bool(evidence[0])
                        if args.local_gallery and chosen not in examples:
                            examples[chosen] = (pixels, region, points[0].shape, points[2][0], ('CC', 'MLO')[view])
            if (start // args.batch + 1) % 25 == 0:
                print(f'Processed {min(start + args.batch, len(positions))} of {len(positions)} test breasts', flush=True)
    if not all(observations[key] for key in ('all', *KINDS)):
        raise ValueError('Both lesion types need matched masks to report the audit')
    report = {'selection_sha256': checksum(args.selection), 'checkpoint_sha256': candidate['checkpoint_sha256'],
              'mask_source_sha256': input_hashes, 'test_views': len(test_views), 'mapping': mapping,
              'mask_alignment_checks': dict(skipped), 'bootstrap_repeats': args.bootstrap,
              'include_unlabeled_masks': args.recover_unlabeled_masks,
              'method': ('Maximum raw logit across all four lesion channels, choosing one cell with at least 25% tissue. '
                         'Any aligned mask pixel in that cell counts as an overlap. No flip pooling. '
                         'Top-three and top-five figures allow a match in any ranked cell; uniform expectations '
                         'draw that many different cells at random.'),
              'results': {key: averages(rows, args.bootstrap) for key, rows in observations.items()}}
    write_json(args.output, report)
    if args.local_gallery:
        if len(examples) != 4:
            raise ValueError('The local gallery requires an overlap and a miss for each lesion type')
        gallery(examples, args.output.with_name('local-peak-gallery.png'))
    print(report['results'], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cohort', 'selection', 'work', 'weights', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    parser.add_argument('--clip', type=Path)
    parser.add_argument('--component', default='view-supervised')
    parser.add_argument('--recover-unlabeled-masks', action='store_true',
                        help='Check full-size, near-binary files in the official calcification mask series')
    parser.add_argument('--local-gallery', action='store_true',
                        help='Draw four actual held-out projections beside the local report')
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
