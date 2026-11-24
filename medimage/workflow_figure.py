"""Render measured inference steps for the public synthetic examples."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.special import expit

from medimage.auc_mammo_data import view_pixels
from medimage.common import read_json
from medimage.demo import PHANTOM_STUDY
from medimage.portfolio_figures import BLUE, TEAL, plt, write_figure
from medimage.radiography_data import read_image
from medimage.search_features import chest_source
from medimage.spine_localize import SpineCache


def image(axis, values, title: str) -> None:
    axis.imshow(values, cmap='gray', vmin=0, vmax=255)
    axis.set_title(title, loc='left', fontsize=10, fontweight='bold')
    axis.set_axis_off()


def frame(title: str, description: str):
    fig = plt.figure(figsize=(11.2, 4.35), layout='constrained')
    fig.suptitle(f'{title}\n{description}', ha='left', x=.025, fontsize=15, fontweight='bold')
    grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 1.65))
    return fig, grid


def bars(axis, labels, values, limits, label: str, threshold=None) -> None:
    y = np.arange(len(labels))
    axis.barh(y, values, height=.48, color=TEAL)
    if threshold is not None:
        axis.scatter(threshold, y, marker='|', s=240, linewidth=2, color=BLUE, zorder=3,
                     label='Validation threshold')
        axis.legend(loc='upper right', fontsize=8, frameon=False)
    for pos, val in zip(y, values, strict=True):
        marker_is_close = threshold is not None and val <= threshold[pos] <= val + .11 * limits[1]
        axis.text(val + .014, pos + .30 * marker_is_close, f'{val:.3f}', va='center', fontsize=9)
    axis.set_yticks(y, labels)
    axis.set_xlim(*limits)
    axis.set_xlabel(label, fontsize=9)
    axis.set_title('3. Read the model output', loc='left', fontsize=10, fontweight='bold')
    axis.spines[['top', 'right', 'left']].set_visible(False)
    axis.grid(axis='x', alpha=.22)
    axis.set_axisbelow(True)
    axis.tick_params(axis='y', length=0)
    axis.invert_yaxis()


def check_trace(trace):
    components = np.stack([value['weighted_logit'] for value in trace['components'].values()])
    fused = np.asarray(trace['fused_logit'])
    np.testing.assert_allclose(components.sum(axis=0), fused, atol=1e-8)
    slope, intercept = np.asarray(trace['calibration']).T
    probabilities = np.asarray(trace['probability'])
    np.testing.assert_allclose(expit(fused * slope + intercept), probabilities, atol=1e-8)
    return probabilities, np.asarray(trace['validation_threshold'])


def radiography(task: str, examples: Path, trace: dict, output: Path) -> None:
    probability, threshold = check_trace(trace)
    if task == 'chest':
        fig, grid = frame('Chest X-ray inference', 'One synthetic frontal image produces 14 finding scores')
        source = examples / 'phantom.png'
        image(fig.add_subplot(grid[:, 0]), read_image(source, cv2.IMREAD_GRAYSCALE), '1. Open the input')
        size = max(part['model_input_shape'][0] for part in trace['components'].values()
                   if part['model_input_shape'])
        processed = chest_source(source, size)
        image(fig.add_subplot(grid[:, 1]), processed, f'2. Resize to {size} × {size} pixels')
        show = [trace['labels'].index(key) for key in ('Atelectasis', 'Effusion', 'Pneumonia', 'Pneumothorax')]
        scores, thresholds = probability[show], threshold[show]
        names = [trace['labels'][index] for index in show]
        right = max(scores.max(), thresholds.max())
        bars(fig.add_subplot(grid[:, 2]), names, scores, (0, min(1.055, right + .18)),
             'Calibrated score  ·  four of 14 findings', thresholds)
    else:
        fig, grid = frame('Mammography inference', 'Two synthetic projections produce one breast-level score')
        shape = max(tuple(part['model_input_shape']) for part in trace['components'].values()
                    if part['model_input_shape'])
        for row, name in enumerate(('cc', 'mlo')):
            source = examples / f'phantom-{name}.png'
            raw = read_image(source, cv2.IMREAD_GRAYSCALE)
            processed, _ = view_pixels(source, shape)
            left = f'1. {name.upper()} input' if row == 0 else f'{name.upper()} input'
            middle = f'2. {name.upper()} processed' if row == 0 else f'{name.upper()} processed'
            image(fig.add_subplot(grid[row, 0]), raw, left)
            image(fig.add_subplot(grid[row, 1]), processed, middle)
        score_axis = fig.add_subplot(grid[0, 2])
        bars(score_axis, ['Malignancy'], probability, (0, 1.055), 'Calibrated score', threshold)
        score_axis.set_ylim(.85, -.85)
        weights_axis = fig.add_subplot(grid[1, 2])
        weights_axis.set_axis_off()
        weights_axis.text(0, .88, 'The eight-model blend', fontsize=10, fontweight='bold')
        weights_axis.text(0, .64, 'Each model reads the available projections.\n'
                                  'The trace records the weight and logit\n'
                                  'from each component before calibration.',
                          va='top', fontsize=9, linespacing=1.4)
    write_figure(fig, output)


def mri(examples: Path, cache_path: Path, trace: dict, output: Path) -> None:
    cache = SpineCache.load(cache_path)
    if cache.studies.tolist() != [PHANTOM_STUDY] or trace['studies'] != [PHANTOM_STUDY]:
        raise ValueError('The public MRI illustration expects the generated phantom')
    fold_predictions = np.asarray(trace['fold_probabilities'])
    reference = np.asarray(trace['reference_probabilities'])
    model = np.asarray(trace['new_model_probabilities'])
    blended = np.asarray(trace['blended_probabilities'])
    np.testing.assert_allclose(fold_predictions.mean(axis=0), model, atol=1e-8)
    np.testing.assert_allclose(trace['reference_fraction'] * reference + trace['new_model_fraction'] * model,
                               blended, atol=1e-8)
    target = trace['targets'].index('spinal_canal_stenosis_l4_l5')
    fig, grid = frame('Lumbar MRI inference', 'One synthetic study produces three probabilities for each of 25 targets')
    raw = read_image(examples / 'phantom.png', cv2.IMREAD_GRAYSCALE)
    image(fig.add_subplot(grid[:, 0]), raw, '1. Open the three planes')
    slot, x, y, width, height = trace['first_fold_regions']['coordinates'][0][target][1]
    pilot = cache.volumes[0, int(slot)]
    axis = fig.add_subplot(grid[:, 1])
    image(axis, pilot, '2. Locate L4–L5')
    from matplotlib.patches import Rectangle

    axis.add_patch(Rectangle((x - width / 2, y - height / 2), width, height,
                             fill=False, edgecolor='#52e9cc', linewidth=1.7))
    scores = blended[0, target]
    bars(fig.add_subplot(grid[:, 2]), ['Normal / mild', 'Moderate', 'Severe'], scores,
         (0, max(.7, scores.max() + .13)), 'Probability  ·  L4–L5 spinal canal stenosis')
    write_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('chest', 'mammo', 'spine'))
    for name in ('examples', 'trace', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    args = parser.parse_args()
    trace = read_json(args.trace)
    if trace['task'] != args.task:
        parser.error('The trace must match the requested task')
    if args.task == 'spine':
        if args.cache is None:
            parser.error('MRI requires the synthetic study pixel cache')
        mri(args.examples, args.cache, trace, args.output)
    else:
        radiography(args.task, args.examples, trace, args.output)


if __name__ == '__main__':
    main()
