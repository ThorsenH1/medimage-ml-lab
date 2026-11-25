"""Draw the actual regions used for a synthetic lumbar MRI prediction."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

from medimage.common import read_json
from medimage.demo import PHANTOM_STUDY
from medimage.portfolio_figures import BLUE, TEAL, plt, write_figure
from medimage.spine_data import LEVELS, anchor_slot, project
from medimage.spine_localize import SpineCache
from medimage.spine_model import ROI_SIZE, crop_regions


def render_case(cache, case_index, regions, probabilities, output, annotated=False, reference_grades=None,
                crop_resolution=ROI_SIZE) -> None:
    if regions.shape != (25, 3, 5):
        raise ValueError('Expected one three-slice window per graded disc-level target')
    grade_names = ('Normal / mild', 'Moderate', 'Severe')
    if reference_grades is not None:
        reference_grades = np.asarray(reference_grades)
        if reference_grades.shape != (5,) or not np.isin(reference_grades, [-1, 0, 1, 2]).all():
            raise ValueError('Expected five reference grades in lumbar disc order')
        reference_grades = reference_grades.astype(np.int64, copy=False)
    packed = np.full((30, 3, 5), -1, dtype=np.float32)
    packed[:25] = regions
    crops = crop_regions(cache.volumes[case_index], packed, size=crop_resolution)
    slots = regions[:5, 1, 0].astype(int)
    if (slots < 0).any():
        raise ValueError('The example is missing a sagittal disc region')
    slot = anchor_slot(cache.valid[case_index])
    anchor = cache.geometry[case_index, slot]
    coordinate = regions[:5, 1, 1:3]
    geometry = cache.geometry[case_index, slots]
    centers = geometry[:, 0] + (coordinate[..., None] * geometry[:, 1:]).sum(axis=1)
    display, _ = project(centers, anchor)
    probabilities = np.asarray(probabilities)[:5]
    names = [level.replace('_', '–').upper() for level in LEVELS]

    fig = plt.figure(figsize=(11.7, 5.4), layout='constrained')
    fig.suptitle('Where the MRI model reads the spinal canal', x=.025, ha='left', fontsize=17, fontweight='bold')
    grid = fig.add_gridspec(5, 3, width_ratios=(2.4, .8, 2.6))
    overview = fig.add_subplot(grid[:, 0])
    slice_pixels = cache.volumes[case_index, slot]
    overview.imshow(slice_pixels, cmap='gray', vmin=0, vmax=255)
    overview.set_title('Model estimates and reference marks' if annotated else 'Estimated centers on a sagittal slice',
                       loc='left', fontsize=11)
    overview.set_xlim(-.5, slice_pixels.shape[1] - .5)
    overview.set_ylim(slice_pixels.shape[0] - .5, -.5)
    overview.set_axis_off()
    label_ys = np.linspace(.15, .85, 5) * slice_pixels.shape[0]
    for index, (x, y) in enumerate(display):
        if not (np.isfinite(x) and np.isfinite(y)):
            raise ValueError('Unable to project a selected region onto the display plane')
        overview.scatter([x], [y], s=36, color='#3df3d2', edgecolor='#0e3b49', linewidth=.9)
        overview.annotate(names[index], (x, y), xytext=(8, label_ys[index]), textcoords='data',
                          color='white', va='center', ha='left', fontsize=9, fontweight='bold',
                          bbox={'boxstyle': 'round,pad=.23', 'facecolor': '#16232c', 'edgecolor': 'none'},
                          arrowprops={'arrowstyle': '-', 'color': '#3df3d2', 'shrinkA': 3, 'shrinkB': 4})
    if annotated:
        gold, _ = project(cache.points[case_index, :5], anchor)
        overview.scatter(gold[:, 0], gold[:, 1], marker='+', s=68, color='#ffbb63', linewidth=1.7)
        overview.text(.5, -.035, 'Teal dot = model estimate     Gold cross = dataset mark',
                      transform=overview.transAxes, ha='center', fontsize=8)
    row = 3
    x, y = display[row]
    width, height = regions[row, 1, 3:5]
    overview.add_patch(Rectangle((x - width / 2, y - height / 2), width, height,
                                 fill=False, edgecolor='#3df3d2', linewidth=1.2, linestyle='--'))
    colors = [BLUE, '#cfad4a', TEAL]
    axes = []
    for row, name in enumerate(names):
        tile = fig.add_subplot(grid[row, 1])
        tile.imshow(crops[row, 1], cmap='gray', vmin=0, vmax=255)
        tile.set_ylabel(name, rotation=0, ha='right', va='center', fontsize=9)
        tile.set(xticks=[], yticks=[])
        tile.spines[:].set_visible(False)
        if row == 0:
            tile.set_title('Actual crop', fontsize=10)
        ax = fig.add_subplot(grid[row, 2])
        axes.append(ax)
        beginning = 0.
        for index, value in enumerate(probabilities[row]):
            ax.barh(0, value, left=beginning, height=.60, color=colors[index])
            if value >= .085:
                ax.text(beginning + value / 2, 0, f'{value:.2f}', ha='center', va='center',
                        color='#182725' if index == 1 else 'white', fontsize=9)
            beginning += value
        ax.set_xlim(0, 1)
        ax.set_ylim(-.65, .65)
        ax.set_yticks([])
        ax.set_xticks([] if row < 4 else [.0, .25, .5, .75, 1.])
        ax.spines[['top', 'left', 'right']].set_visible(False)
        if reference_grades is not None and reference_grades[row] >= 0:
            ax.text(.012, -.49, f'Dataset grade: {grade_names[reference_grades[row]]}', va='center', fontsize=8)
        if row == 0:
            ax.set_title('Three model scores at each level', loc='left', fontsize=11)
    axes[-1].set_xlabel('Normal / mild (blue)     Moderate (gold)     Severe (teal)', fontsize=9)
    write_figure(fig, output)


def render_lateral_zoom(cache, trace: dict, output: Path) -> None:
    target = trace['targets'].index('left_subarticular_stenosis_l4_l5')
    sampling = trace['first_fold_regions']
    wide = np.asarray(sampling['coordinates'][0][target], dtype=np.float32)
    narrow = np.asarray(sampling['lateral_zoom_coordinates'][0][target - 5], dtype=np.float32)
    packed = np.full((30, 3, 5), -1, dtype=np.float32)
    packed[:2] = wide, narrow
    pixels = cache.volumes[0]
    crops = crop_regions(pixels, packed, size=sampling.get('crop_resolution', ROI_SIZE))
    slot = int(wide[1, 0])
    if slot < 0 or slot != int(narrow[1, 0]):
        raise ValueError('The two views must share a valid axial slice')

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.8), layout='constrained',
                             gridspec_kw={'width_ratios': [1.2, .8, .8, 1.45]})
    fig.suptitle('Zooming in on the left side at L4–L5', x=.025, ha='left', fontsize=16, fontweight='bold')
    axes[0].imshow(pixels[slot], cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Synthetic axial slice')
    for row, color in zip((wide, narrow), ('#f6bf65', '#3df3d2'), strict=True):
        _, x, y, width, height = row[1]
        axes[0].add_patch(Rectangle((x - width / 2, y - height / 2), width, height,
                                    fill=False, edgecolor=color, linewidth=1.5))
    for axis, crop, title, color in zip(axes[1:3], crops[:2], ('Original crop', '80% field'),
                                        ('#f6bf65', '#3df3d2'), strict=True):
        axis.imshow(crop[1], cmap='gray', vmin=0, vmax=255)
        axis.set_title(title, color='#37474f')
        for side in axis.spines.values():
            side.set_color(color)
            side.set_linewidth(2.)
    for axis in axes[:3]:
        axis.set(xticks=[], yticks=[])

    probabilities = np.asarray(trace['blended_probabilities'])[0, target]
    axes[3].barh([0, 1, 2], probabilities, color=[BLUE, '#cfad4a', TEAL], height=.55)
    axes[3].set(yticks=[0, 1, 2], yticklabels=['Normal / mild', 'Moderate', 'Severe'], xlim=(0, 1.04))
    axes[3].set_title('Final ensemble')
    axes[3].spines[['top', 'left', 'right']].set_visible(False)
    axes[3].invert_yaxis()
    for row, value in enumerate(probabilities):
        axes[3].text(value + .025, row, f'{value:.2f}', va='center', fontsize=9)
    write_figure(fig, output)


def render(cache_path: Path, trace_path: Path, output: Path, zoom_output: Path | None = None) -> None:
    trace = read_json(trace_path)
    cache = SpineCache.load(cache_path)
    if trace['studies'] != [PHANTOM_STUDY] or cache.studies.tolist() != [PHANTOM_STUDY]:
        raise ValueError('The public figure expects the one generated MRI study')
    sampling = trace['first_fold_regions']
    if sampling.get('cache_size', cache.volumes.shape[-1]) != cache.volumes.shape[-1]:
        raise ValueError('The trace coordinates refer to a different image grid')
    regions = np.asarray(sampling['coordinates'][0], dtype=np.float32)
    render_case(cache, 0, regions, trace['blended_probabilities'][0], output,
                crop_resolution=sampling.get('crop_resolution', ROI_SIZE))
    if zoom_output is not None:
        render_lateral_zoom(cache, trace, zoom_output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'trace', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--zoom-output', type=Path)
    args = parser.parse_args()
    render(args.cache, args.trace, args.output, args.zoom_output)


if __name__ == '__main__':
    main()
