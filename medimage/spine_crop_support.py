"""Check held-out annotated landmarks against the physical MRI crops the graders read."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from medimage.common import checksum, read_json, write_json
from medimage.spine_data import CONDITIONS, LEVELS, STARTS, TARGETS, project
from medimage.spine_detail_infer import smaller_field
from medimage.spine_localize import SpineCache, read_folds
from medimage.spine_train import _load_fold

SLICE_TOLERANCE_MM = 5.0
SUBARTICULAR = slice(15, 25)


def wider_axial_stack(cache: SpineCache, coordinates: np.ndarray) -> np.ndarray:
    """Move the two neighboring subarticular slices out one acquisition step per side."""
    wider = coordinates.copy()
    size = cache.volumes.shape[-1]
    start, end = STARTS[2:4]
    for row, regions in enumerate(wider[:, SUBARTICULAR]):
        seen = np.flatnonzero(cache.valid[row, start:end]) + start
        if len(seen) < 2:
            continue
        current = regions[:, 1, 0].astype(int)
        valid = np.isin(current, seen)
        if not valid.any():
            continue
        position = np.searchsorted(seen, current[valid])
        destination = seen[np.clip(position[:, None] + [-2, 2], 0, len(seen) - 1)]
        geometry = cache.geometry[row]
        center = geometry[current[valid]]
        world = center[:, 0] + (regions[valid, 1, 1:3, None] * center[:, 1:]).sum(axis=1)
        planes = geometry[destination]
        xy, _ = project(world[:, None], planes)
        dimensions_mm = regions[valid, 1, 3:5] * np.linalg.norm(center[:, 1:], axis=-1)
        wider_fields = np.clip(dimensions_mm[:, None] / np.linalg.norm(planes[:, :, 1:], axis=-1), 18, size)
        for outside, slot in enumerate((0, 2)):
            regions[valid, slot, 0] = destination[:, outside]
            regions[valid, slot, 1:3] = xy[:, outside].clip(0, size - 1)
            regions[valid, slot, 3:5] = wider_fields[:, outside]
    return wider


def measure(cache: SpineCache, coordinates: np.ndarray) -> dict:
    annotated = np.isfinite(cache.points).all(axis=-1)
    rows, targets = np.nonzero(annotated)
    crops = coordinates[rows, targets]
    slots = crops[..., 0].astype(int)
    observed = (slots >= 0) & cache.valid[rows[:, None], slots.clip(min=0)]
    planes = cache.geometry[rows[:, None], slots.clip(min=0)].copy()
    planes[~observed] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    landmark = cache.points[rows, targets, None, :]
    xy, depth = project(landmark, planes)
    footprint = (np.abs(xy - crops[..., 1:3]) <= crops[..., 3:5] / 2).all(axis=-1)
    close = observed & (depth <= SLICE_TOLERANCE_MM)
    nearby = close.any(axis=-1)
    coverage = (close & footprint).any(axis=-1)
    available = observed.any(axis=-1)
    center = planes[..., 0, :] + (crops[..., 1:3, None] * planes[..., 1:, :]).sum(axis=-2)
    displacement = landmark - center
    nearest = np.where(observed, depth, np.inf).argmin(axis=-1)
    nearest_depth = depth[np.arange(len(rows)), nearest]
    nearest_delta = displacement[np.arange(len(rows)), nearest]
    center_error = np.linalg.norm(nearest_delta, axis=-1)

    def summarize(mask):
        located = mask & available
        observed_count = int(located.sum())
        return {'annotations': int(mask.sum()), 'studies': int(len(np.unique(rows[mask]))),
                'available': observed_count, 'covered': int(coverage[mask].sum()),
                'no_slice_within_tolerance': int((mask & ~nearby).sum()),
                'nearby_slice_but_outside_crop': int((mask & nearby & ~coverage).sum()),
                'covered_fraction': float(coverage[mask].mean()) if mask.any() else None,
                'median_plane_distance_mm': float(np.median(nearest_depth[located])) if observed_count else None,
                'median_center_error_mm': float(np.median(center_error[located])) if observed_count else None,
                'median_landmark_minus_center_patient_axes_mm':
                    np.median(nearest_delta[located], axis=0).tolist() if observed_count else None}

    groups = targets // len(LEVELS)
    return {'overall': summarize(np.ones(len(rows), dtype=bool)),
            'conditions': {name: summarize(groups == index) for index, name in enumerate(CONDITIONS)},
            'targets': {name: summarize(targets == index) for index, name in enumerate(TARGETS)}}


def figure(results: dict, output: Path) -> None:
    from medimage.portfolio_figures import BLUE, TEAL, plt, write_figure

    labels = ['Central canal', 'Left neural foramen', 'Right neural foramen',
              'Left subarticular', 'Right subarticular']
    rows = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.3), sharey=True, layout='constrained')
    fig.suptitle('How closely do the MRI crops match the annotated locations?',
                 x=.025, ha='left', fontsize=15, fontweight='bold')
    for axis, key, title, factor in zip(axes, ('median_center_error_mm', 'covered_fraction'),
                                        ('Median center error (mm)', 'Annotations covered (%)'), (1, 100), strict=True):
        reference = np.array([results['atlas']['conditions'][name][key] for name in CONDITIONS]) * factor
        learned = np.array([results['learned']['conditions'][name][key] for name in CONDITIONS]) * factor
        axis.hlines(rows, reference, learned, color='#90a3a6', linewidth=1.2)
        axis.scatter(reference, rows, label='Original atlas', s=35, color=BLUE)
        axis.scatter(learned, rows, label='Learned crops', s=40, color=TEAL, marker='D')
        axis.set_title(title, loc='left', fontsize=11)
        axis.grid(axis='x', color='#e3eaeb')
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(rows, labels)
    axes[0].invert_yaxis()
    axes[0].legend(loc='lower right', frameon=False)
    write_figure(fig, output)


def evaluate(args: argparse.Namespace) -> None:
    cache = SpineCache.load(args.cache)
    folds = read_folds(args.folds, cache.studies)
    fold_sha, cache_sha = checksum(args.folds), checksum(args.cache / 'audit.json')
    coordinate_sets = []
    digests = {}
    for name, directory in [('atlas', args.reference), ('learned', args.learned)]:
        audit = read_json(directory / 'localization.json')
        config = audit['config']
        if config['folds_file_sha256'] != fold_sha or config['cache_sha256'] != cache_sha:
            raise ValueError('The two localizers must use the same frozen study folds and physical image cache')
        coords = np.empty((len(folds), 30, 3, 5), dtype=np.float32)
        digests[name] = {}
        for fold in range(5):
            path = directory / f'fold-{fold}.npz'
            digest = checksum(path)
            if name == 'learned' and digest != audit['held_out_localization'][str(fold)]['region_sha256']:
                raise ValueError('The learned regions differ from their recorded cross-fit export')
            regions, _ = _load_fold(path, cache, folds, fold)
            held_out = folds == fold
            coords[held_out] = regions[held_out]
            digests[name][str(fold)] = digest
        coordinate_sets.append(coords)
    atlas, learned = coordinate_sets
    wider = wider_axial_stack(cache, learned)
    methods = {'atlas': atlas, 'learned': learned, 'narrower_sides': smaller_field(learned),
               'extra_axial_stack': np.concatenate((learned, wider), axis=2)}
    result = {name: measure(cache, regions) for name, regions in methods.items()}
    report = {'method': 'Landmark projects into one of the supplied crops within 5 mm of that slice plane',
              'slice_tolerance_mm': SLICE_TOLERANCE_MM, 'folds_sha256': fold_sha, 'cache_sha256': cache_sha,
              'region_file_hashes': digests, 'results': result}
    write_json(args.output, report)
    if args.figure:
        figure(result, args.figure)
    for name, details in result.items():
        print(name, details['overall'], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'folds', 'reference', 'learned', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--figure', type=Path)
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
