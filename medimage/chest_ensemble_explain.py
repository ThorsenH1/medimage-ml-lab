"""Measure patch sensitivity through every image classifier in a locked chest ensemble."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

from medimage.auc_predict import calibrated_probabilities
from medimage.auc_selection import component_logits, load_component
from medimage.common import checksum, read_json, write_json
from medimage.demo import save_png, xray_phantom
from medimage.portfolio_figures import plt, write_figure
from medimage.radiography_data import read_image
from medimage.search_features import chest_pixels
from medimage.visual_explain import alter, impact_overlay, patch_grid, soften


def mark(axis, box, color):
    x0, y0, x1, y1 = box
    axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.8))


def annotated_zoom(axis, image, box, color):
    x0, y0, x1, y1 = box
    height, width = image.shape
    pad = max(x1 - x0, y1 - y0) * .65
    left, right = max(0, round(x0 - pad)), min(width, round(x1 + pad))
    top, bottom = max(0, round(y0 - pad)), min(height, round(y1 + pad))
    axis.imshow(image[top:bottom, left:right], cmap='gray', vmin=0, vmax=255,
                extent=(left - .5, right - .5, bottom - .5, top - .5))
    mark(axis, box, color)


def draw(label, original, image, boxes, probes, score, before, output):
    synthetic = before is not None
    count = 4 if boxes else 3
    fig, axes = plt.subplots(1, count, figsize=(count * 3.1, 4.65))
    fig.subplots_adjust(left=.025, right=.975, top=.77, bottom=.23, wspace=.14)
    subtitle = f'Ensemble score {score:.3%}'
    if synthetic:
        subtitle = f'Generated chest image • Ensemble score {before:.3%} before the edit and {score:.3%} after'
    fig.text(.025, .955, f'Which parts change the {label.lower()} score?', va='top', fontsize=17, fontweight='bold')
    fig.text(.025, .88, subtitle, va='top', fontsize=11, color='#273545')
    gray = dict(cmap='gray', vmin=0, vmax=255)
    axes[0].imshow(image, **gray)
    strongest = max(probes, key=lambda item: abs(item['score_drop']))
    zoom = boxes[0] if boxes else strongest['box_xyxy']
    x0, y0, x1, y1 = map(round, zoom)
    for box in boxes:
        mark(axes[0], box, '#40dfc5')
    if synthetic:
        axes[0].set_title('1. Find the added shadow', loc='left')
        axes[1].imshow(original[y0:y1, x0:x1], **gray)
        axes[1].set_title('2. Before, zoomed', loc='left')
        axes[2].imshow(image[y0:y1, x0:x1], **gray)
        axes[2].set_title('3. After, zoomed', loc='left')
    else:
        if not boxes:
            mark(axes[0], zoom, '#ffaa56')
        axes[0].set_title('1. Image with reference boxes' if boxes else '1. Find the strongest response', loc='left')
        annotated_zoom(axes[1], image, zoom, '#40dfc5' if boxes else '#ffaa56')
        axes[1].set_title('2. Reference region, zoomed' if boxes else '2. Response region, zoomed', loc='left')
        if boxes:
            annotated_zoom(axes[2], image, strongest['box_xyxy'], '#ffaa56')
            axes[2].set_title('3. Strongest response, zoomed', loc='left')
    heatmap = impact_overlay(axes[-1], image, probes, scale_floor=0., fade=True)
    for box in boxes:
        mark(axes[-1], box, '#40dfc5')
    mark(axes[-1], strongest['box_xyxy'], '#ffaa56')
    axes[-1].set_title(f'{count}. Blur one patch at a time', loc='left')
    scale_axis = axes[-1].inset_axes([.05, -.27, .90, .032])
    scale = ScalarMappable(norm=heatmap.norm, cmap=heatmap.cmap)
    colorbar = fig.colorbar(scale, cax=scale_axis, orientation='horizontal',
                            ticks=MaxNLocator(4), format=FuncFormatter(lambda value, _: f'{100 * value:+.2f}'))
    colorbar.set_label('Original minus blurred score (points)', fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    for axis in axes:
        axis.set_axis_off()
        axis.title.set_fontsize(10)
    write_figure(fig, output)


@torch.inference_mode()
def explain(args):
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    lock = read_json(args.selection)
    if lock['task'] != 'chest':
        raise ValueError('Use a frozen chest selection')
    position = lock['labels'].index(args.label)
    entries = {name: lock['candidates'][name] for name in lock['selected']['weights']}
    supported = {'biovil-t', 'convnext-chest', 'cxr-clip-swin', 'chest-reference', 'chex', 'imagenet'}
    if any(not entry.get('image_shape') or entry.get('config', {}).get('backbone') not in supported
           for entry in entries.values()):
        raise ValueError('This probe expects image-classifier components with a stored input size')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.image is None:
        xray_phantom(args.output)
        original = read_image(args.output / 'phantom.png', cv2.IMREAD_GRAYSCALE)
        image, box = alter('chest', original)
        save_png(args.output / 'phantom-edited.png', image)
        references = [box]
        images = [image, original]
    else:
        original = image = read_image(args.image, cv2.IMREAD_GRAYSCALE)
        references = args.reference_box
        images = [image]
    height, width = image.shape
    if any(not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height)
           for x0, y0, x1, y1 in references):
        raise ValueError('Reference boxes must be within the original image')
    boxes = patch_grid(image, 'chest')
    if not boxes:
        raise ValueError('The image contains no testable regions')

    first_probe = len(images)
    images.extend(soften(image, boxes))
    logits = {}
    for name, entry in entries.items():
        model, saved = load_component(args.work, entry, args.weights, None)
        resolution = entry['image_shape'][0]
        predictions = []
        # Use the normal single-image batch shape for every probe to keep FP16 kernel changes out of the score delta.
        for pixels in images:
            arrays = {'pixels': chest_pixels(pixels, resolution)[None]}
            predictions.append(component_logits(model, saved, arrays, np.zeros(1, dtype=np.int64), args.weights))
        logits[name] = np.concatenate(predictions)
        del model, saved
        torch.cuda.empty_cache()

    scores = calibrated_probabilities(lock, logits)[:, position]
    score = float(scores[0])
    before = float(scores[1]) if args.image is None else None
    records = [{'box_xyxy': box, 'score_after_blur': float(blurred), 'score_drop': score - float(blurred)}
               for box, blurred in zip(boxes, scores[first_probe:], strict=True)]
    evidence = {'label': args.label, 'selection_sha256': checksum(args.selection),
                'image_kind': 'deterministic generated phantom' if before is not None else 'supplied local image',
                'image_height_width': [height, width], 'coordinate_system': 'original image pixels, xyxy',
                'score_units': 'calibrated research score on a 0–1 scale', 'unblurred_score': score,
                'before_synthetic_edit': before, 'validation_threshold': lock['thresholds'][position],
                'reference_boxes_xyxy': references, 'probes': records,
                'components': {name: {'checkpoint_sha256': entry['checkpoint_sha256'],
                                      'input_shape': entry['image_shape'],
                                      'unblurred_logit': float(logits[name][0, position])}
                               for name, entry in entries.items()}}
    draw(args.label, original, image, references, records, score, before, args.output / 'ensemble-region-probe.png')
    write_json(args.output / 'ensemble-region-evidence.json', evidence)
    print(f'{args.label}: {score:.8f} across {len(entries)} components and {len(records)} local probes', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('selection', 'work', 'weights', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--reference-box', type=float, nargs=4, action='append', default=[],
                        metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    args = parser.parse_args()
    if args.reference_box and args.image is None:
        parser.error('Reference boxes require a supplied image')
    explain(args)


if __name__ == '__main__':
    main()
