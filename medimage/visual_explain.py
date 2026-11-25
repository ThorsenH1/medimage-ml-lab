"""Show a known synthetic image edit and measure a model's response to local blur."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib.patches import Rectangle

from medimage.auc_predict import arrays_for
from medimage.auc_selection import component_logits, load_component
from medimage.common import checksum, read_json, write_json
from medimage.demo import mammography_phantom, save_png, xray_phantom
from medimage.mammo_native import PIXEL_FORMAT
from medimage.portfolio_figures import plt, write_figure
from medimage.radiography_data import read_image


def alter(task: str, original: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = original.shape
    y, x = np.mgrid[0:height, 0:width]
    if task == 'chest':
        edit = 58 * np.exp(-(((x / width - .30) / .086) ** 2 + ((y / height - .69) / .056) ** 2))
        bounds = (.17, .60, .43, .78)
    else:
        offsets = ((-.018, .007), (-.006, -.016), (.003, .012), (.015, -.008), (.024, .002))
        edit = sum(130 * np.exp(-(((x / width - .51 - dx) / .0028) ** 2
                                  + ((y / height - .46 - dy) / .0022) ** 2))
                   for dx, dy in offsets)
        bounds = (.47, .423, .55, .497)
    pixels = np.clip(np.rint(original.astype(float) + edit), 0, 255).astype(np.uint8)
    return pixels, tuple(round(value * size) for value, size in zip(bounds, (width, height, width, height), strict=True))


def patch_grid(image: np.ndarray, task: str) -> list[tuple[int, int, int, int]]:
    height, width = image.shape
    ny, nx = (7, 7) if task == 'chest' else (8, 7)
    window_h = max(2, round(height / (ny + 1) * 1.25))
    window_w = max(2, round(width / (nx + 1) * 1.25))
    support = cv2.GaussianBlur(image, (0, 0), max(1, min(height, width) / 90))
    black = np.iinfo(image.dtype).max * (8 / 255)
    result = []
    for cy in np.linspace(window_h // 2, height - window_h // 2, ny).astype(int).tolist():
        for cx in np.linspace(window_w // 2, width - window_w // 2, nx).astype(int).tolist():
            left, top = max(0, cx - window_w // 2), max(0, cy - window_h // 2)
            right, bottom = min(width, left + window_w), min(height, top + window_h)
            if (support[top:bottom, left:right] > black).mean() >= .28:
                result.append((left, top, right, bottom))
    return result


def soften(image: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
    smooth = cv2.GaussianBlur(image, (0, 0), max(2, min(image.shape) / 22))
    changed = []
    for left, top, right, bottom in boxes:
        horizontal = np.hanning(2 + right - left)[1:-1]
        vertical = np.hanning(2 + bottom - top)[1:-1]
        weight = np.outer(vertical, horizontal)
        out = image.copy()
        view = np.s_[top:bottom, left:right]
        out[view] = np.rint(image[view] * (1 - weight) + smooth[view] * weight).astype(image.dtype)
        changed.append(out)
    return changed


def impact_overlay(axis, model_input, scores):
    axis.imshow(model_input, cmap='gray', vmin=0, vmax=np.iinfo(model_input.dtype).max)
    impact = np.zeros(model_input.shape, dtype=np.float32)
    overlap = np.zeros_like(impact)
    for row in scores:
        x0, y0, x1, y1 = row['box_xyxy']
        weight = np.outer(np.hanning(2 + y1 - y0)[1:-1], np.hanning(2 + x1 - x0)[1:-1])
        impact[y0:y1, x0:x1] += weight * row['score_drop']
        overlap[y0:y1, x0:x1] += weight
    impact = np.divide(impact, overlap, out=impact, where=overlap > 0)
    scale = max(.015, float(np.abs(impact).max()))
    visible = (overlap > .10) & (np.abs(impact) > .04 * scale)
    axis.imshow(np.ma.masked_where(~visible, impact), alpha=.72, cmap='RdBu_r', vmin=-scale, vmax=scale)
    axis.text(.5, -.065, 'Red lowered the score, blue raised it', transform=axis.transAxes,
              ha='center', fontsize=9)


def figure(label, original, edited, bounds, model_input, scores, output):
    fig, cells = plt.subplots(1, 4, figsize=(12.2, 4.15), layout='constrained')
    fig.suptitle(f'What changed the {label} model score?', x=.02, ha='left', fontsize=17, fontweight='bold')
    left, top, right, bottom = bounds
    gray = dict(cmap='gray', vmin=0, vmax=255)
    cells[0].imshow(edited, **gray)
    cells[0].add_patch(Rectangle((left, top), right - left, bottom - top, fill=False,
                                 edgecolor='#40dfc5', linewidth=1.8))
    cells[0].set_title('1. Find the edited region', loc='left')
    cells[1].imshow(original[top:bottom, left:right], **gray)
    cells[1].set_title('2. Before, zoomed', loc='left')
    cells[2].imshow(edited[top:bottom, left:right], **gray)
    cells[2].set_title('3. After, zoomed', loc='left')
    impact_overlay(cells[3], model_input, scores)
    cells[3].set_title('4. Blur one region at a time', loc='left')
    for axis in cells:
        axis.set_axis_off()
        axis.title.set_fontsize(10)
    write_figure(fig, output)


def single_figure(label, pixels, scores, output):
    fig, cells = plt.subplots(1, 3, figsize=(10.1, 4.1), layout='constrained')
    fig.suptitle(f'What changed the {label} model score?', x=.02, ha='left', fontsize=17, fontweight='bold')
    strongest = max(scores, key=lambda row: abs(row['score_drop']))
    left, top, right, bottom = strongest['box_xyxy']
    gray = dict(cmap='gray', vmin=0, vmax=np.iinfo(pixels.dtype).max)
    cells[0].imshow(pixels, **gray)
    cells[0].add_patch(Rectangle((left, top), right - left, bottom - top, fill=False,
                                 edgecolor='#40dfc5', linewidth=1.6))
    cells[0].set_title('Processed model input', fontsize=10)
    cells[1].imshow(pixels[top:bottom, left:right], **gray)
    cells[1].set_title('Zoom of the strongest response', fontsize=10)
    impact_overlay(cells[2], pixels, scores)
    cells[2].set_title('Blur one region at a time', fontsize=10)
    for axis in cells:
        axis.set_axis_off()
    write_figure(fig, output)


@torch.inference_mode()
def explain(args) -> None:
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    lock = read_json(args.selection)
    task = lock['task']
    position = lock['labels'].index(args.label)
    entry = lock['candidates'][args.component]
    if not entry.get('image_shape'):
        raise ValueError('Choose a single image-consuming model component')
    args.output.mkdir(parents=True, exist_ok=True)
    real = any((args.image, args.cc, args.mlo))
    pixel_format = entry['config'].get('pixel_format')
    if pixel_format == PIXEL_FORMAT and not real:
        raise ValueError('Supply at least one local DICOM projection for the native component')
    view_index = 0
    if real and task == 'chest':
        if args.image is None or args.cc or args.mlo:
            raise ValueError('A chest explanation takes one frontal --image')
        after_args = argparse.Namespace(image=args.image)
    elif real:
        if args.image or not (args.cc or args.mlo):
            raise ValueError('Mammography takes --cc and/or --mlo')
        after_args = argparse.Namespace(cc=args.cc, mlo=args.mlo, native_cc=args.cc, native_mlo=args.mlo)
        view_index = int(args.cc is None)
    elif task == 'chest':
        xray_phantom(args.output)
        source = args.output / 'phantom.png'
    else:
        mammography_phantom(args.output)
        source = args.output / 'phantom-cc.png'
    if not real:
        input_path = args.output / 'phantom-edited.png'
        original = read_image(source, cv2.IMREAD_GRAYSCALE)
        edited, bounds = alter(task, original)
        save_png(input_path, edited)
        if task == 'chest':
            before_args = argparse.Namespace(image=source)
            after_args = argparse.Namespace(image=input_path)
        else:
            paired = args.output / 'phantom-mlo.png'
            before_args = argparse.Namespace(cc=source, mlo=paired)
            after_args = argparse.Namespace(cc=input_path, mlo=paired)
        before = arrays_for(task, entry['image_shape'], before_args)
    plane = (0,) if task == 'chest' else (0, view_index)
    after = arrays_for(task, entry['image_shape'], after_args, pixel_format)
    pixels = after['pixels'][plane]
    boxes = patch_grid(pixels, task)
    if not boxes:
        raise ValueError('The phantom contains no testable regions')
    variants = soften(pixels, boxes)
    model, saved = load_component(args.work, entry, args.weights, None)
    edited_score = float(component_logits(model, saved, after, np.array([0]), args.weights)[0, position])
    records = []
    for start in range(0, len(variants), args.batch):
        stop = min(start + args.batch, len(variants))
        n = stop - start
        pool = {key: np.repeat(value, n, axis=0) for key, value in after.items()}
        if task == 'chest':
            pool['pixels'][:] = np.stack(variants[start:stop])
        else:
            pool['pixels'][:, view_index] = np.stack(variants[start:stop])
        scores = component_logits(model, saved, pool, np.arange(n), args.weights)[:, position]
        records.extend({'box_xyxy': box, 'score_after_blur': float(score),
                        'score_drop': edited_score - float(score)}
                       for box, score in zip(boxes[start:stop], scores, strict=True))
    stats = {'task': task, 'image_kind': 'supplied local image' if real else 'deterministic generated phantom',
             'label': args.label,
             'model_component': args.component, 'model_checkpoint_sha256': entry['checkpoint_sha256'],
             'selection_sha256': checksum(args.selection), 'occlusion_coordinate_system': 'preprocessed image',
             'projection': ('CC', 'MLO')[view_index] if task == 'mammo' else 'frontal',
             'component_score_units': 'log-odds before ensemble calibration',
             'unblurred_score': edited_score, 'probes': records}
    target = args.label.lower() if task == 'chest' else 'malignancy'
    output = args.output / 'where-model-looked.png'
    if real:
        single_figure(target, pixels, records, output)
    else:
        base_score = float(component_logits(model, saved, before, np.array([0]), args.weights)[0, position])
        stats.update(edit_box_xyxy_raw_pixels=bounds, before_edit=base_score,
                     after_edit=edited_score, score_change=edited_score - base_score)
        figure(target, original, edited, bounds, pixels, records, output)
    write_json(args.output / 'evidence.json', stats)
    print({key: stats.get(key) for key in ('task', 'model_component', 'label', 'before_edit', 'unblurred_score')}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('selection', 'work', 'weights', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--component', required=True)
    for name in ('image', 'cc', 'mlo'):
        parser.add_argument(f'--{name}', type=Path)
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error('Batch size must be positive')
    explain(args)


if __name__ == '__main__':
    main()
