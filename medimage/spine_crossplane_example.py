"""Show both actual 64-pixel crops for one foraminal target in an MRI trace."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

from medimage.common import checksum, read_json, write_json
from medimage.portfolio_figures import plt, write_figure
from medimage.spine_crossplane import FORAMINA, PLANE_LIMIT_MM, axial_regions, crop_stack
from medimage.spine_data import TARGETS
from medimage.spine_localize import SpineCache


def run(args):
    cache = SpineCache.load(args.cache)
    trace = read_json(args.trace)
    recorded = trace['first_fold_regions']
    size = cache.volumes.shape[-1]
    if (trace['studies'] != cache.studies.tolist() or len(cache.studies) != 1
            or recorded.get('cache_size') != size or trace['targets'] != list(TARGETS)):
        raise ValueError('Use the native cache and trace from a single prediction')
    original = np.asarray(recorded['coordinates'][0], dtype=np.float32)[FORAMINA]
    axial, quality = axial_regions(original, cache.geometry[0], cache.valid[0], size)
    available = np.flatnonzero(quality[:, 0])
    if not len(available):
        raise ValueError('This prediction has no eligible second-plane crop')
    index = list(TARGETS[FORAMINA]).index(args.target) if args.target else int(available[0])
    if not quality[index, 0]:
        raise ValueError('The chosen target has no eligible second-plane crop')
    levels = ('L1–L2', 'L2–L3', 'L3–L4', 'L4–L5', 'L5–S1')
    side = 'Left' if index < 5 else 'Right'
    label = f'{side} neural foramen at {levels[index % 5]}'
    distance_mm = float(PLANE_LIMIT_MM * quality[index, 2])
    fig, panels = plt.subplots(1, 4, figsize=(11.65, 3.85))
    fig.subplots_adjust(left=.025, right=.975, bottom=.05, top=.73, wspace=.12)
    fig.text(.025, .96, 'Find the same area in two MRI planes', va='top', fontsize=18, fontweight='bold')
    fig.text(.025, .84, f'{label} · axial plane {distance_mm:.1f} mm from the estimated point', fontsize=11)
    for i, (region, view, color) in enumerate(((original[index], 'Sagittal', '#40dfc5'),
                                               (axial[index], 'Axial', '#ffac66'))):
        full, close = panels[2 * i:2 * i + 2]
        slot, x, y, width, height = region[1]
        full.imshow(cache.volumes[0, int(slot)], cmap='gray', vmin=0, vmax=255)
        full.add_patch(Rectangle((x - width / 2, y - height / 2), width, height, fill=False,
                                 edgecolor=color, linewidth=1.8))
        full.set_title(f'{view} plane', fontsize=11, loc='left')
        close.imshow(crop_stack(cache.volumes[0], region)[1], cmap='gray', vmin=0, vmax=255)
        close.set_title(f'{view} model crop', fontsize=11, loc='left')
        full.set_axis_off()
        close.set_axis_off()
    args.output.mkdir(parents=True, exist_ok=True)
    write_figure(fig, args.output / 'two-planes.png')
    write_json(args.output / 'two-planes.json', {'study': int(cache.studies[0]), 'target': TARGETS[FORAMINA][index],
                                               'source_trace_sha256': checksum(args.trace),
                                               'axial_plane_distance_mm': distance_mm,
                                               'sagittal_crop': original[index].tolist(), 'axial_crop': axial[index].tolist()})
    print(f'{label}: projected center is {distance_mm:.1f} mm from the selected axial plane', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('trace', 'cache', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--target', choices=TARGETS[FORAMINA])
    run(parser.parse_args())


if __name__ == '__main__':
    main()
