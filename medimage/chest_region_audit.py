"""Compare the locked BioViL spatial head with held-out NIH bounding boxes."""
from __future__ import annotations

import argparse
import csv
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from medimage.auc_selection import checked, load_component, pixels_location
from medimage.auc_train import store, transfer
from medimage.common import checksum, read_json, write_json

BOX_IMAGE_SIZE = 1024
LOCAL_STRIDE = 16


def read_boxes(path: Path, frame, labels: list[str]) -> dict:
    by_name = {row.image: index for index, row in enumerate(frame.itertuples(index=False))}
    annotations = defaultdict(lambda: defaultdict(list))
    with path.open(newline='', encoding='utf-8-sig') as source:
        csv_rows = csv.reader(source)
        if next(csv_rows)[:2] != ['Image Index', 'Finding Label']:
            raise ValueError('Unexpected NIH bounding box header')
        for row in csv_rows:
            if not row or not row[0]:
                continue
            image, finding = row[:2]
            index = by_name[image]
            if frame.iloc[index].split != 'test':
                raise ValueError('This audit expects the box annotations to be official test images')
            label = labels.index('Infiltration' if finding.strip() == 'Infiltrate' else finding.strip())
            box = np.asarray(row[2:6], np.float64)
            if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= 0).any():
                raise ValueError('Invalid coordinates in the NIH bounding box release')
            annotations[index][label].append(box)
    if not annotations:
        raise ValueError('No boxes were found in the NIH release')
    return annotations


def check_original_size(path: Path) -> None:
    with path.open('rb') as raster:
        header = raster.read(24)
    if (header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR'
            or struct.unpack('>II', header[16:24]) != (BOX_IMAGE_SIZE, BOX_IMAGE_SIZE)):
        raise ValueError('The source radiograph does not match the annotation coordinate frame')


def grid_evidence(values: np.ndarray, boxes: list, input_size: int) -> tuple[float, float, float]:
    h, w = values.shape
    if h * LOCAL_STRIDE != input_size or w * LOCAL_STRIDE != input_size or not np.isfinite(values).all():
        raise ValueError('Unexpected BioViL layer-3 spatial grid')
    centers = (np.arange(w) * LOCAL_STRIDE + .5) * BOX_IMAGE_SIZE / input_size
    y, x = np.meshgrid(centers, centers, indexing='ij')
    included = np.zeros_like(values, dtype=bool)
    for left, top, width, height in boxes:
        included |= (left <= x) & (x < left + width) & (top <= y) & (y < top + height)
    peak = np.unravel_index(np.argmax(values), values.shape)
    center = int(np.argmin(np.abs(centers - BOX_IMAGE_SIZE / 2)))
    return float(included[peak]), float(included[center, center]), float(included.mean())


def summarise(rows: list, repeats: int) -> dict:
    labels, patients, measurements = zip(*rows, strict=True)
    values = np.asarray(measurements)
    _, inverse = np.unique(patients, return_inverse=True)
    group_count = np.bincount(inverse)
    by_group = np.column_stack([np.bincount(inverse, weights=column) for column in values.T])
    names = ('peak_in_box', 'center_in_box', 'uniform_grid_expectation')
    rng = np.random.default_rng(4852)
    draws = rng.integers(0, len(group_count), size=(repeats, len(group_count)))
    bootstrap = by_group[draws].sum(axis=1) / group_count[draws].sum(axis=1, keepdims=True)
    summary = dict(zip(names, values.mean(axis=0).tolist(), strict=True))
    summary.update(pairs=len(values), patients=len(group_count),
                   peak_in_box_95=np.quantile(bootstrap[:, 0], [.025, .975]).tolist(),
                   peak_minus_uniform_95=np.quantile(bootstrap[:, 0] - bootstrap[:, 2], [.025, .975]).tolist(),
                   peak_minus_center_95=np.quantile(bootstrap[:, 0] - bootstrap[:, 1], [.025, .975]).tolist())
    classes = {}
    for finding in sorted(set(labels)):
        group = values[np.asarray(labels) == finding]
        classes[finding] = {'pairs': len(group), **dict(zip(names, group.mean(axis=0).tolist(), strict=True))}
    return {'overall': summary, 'by_finding': classes}


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    torch.set_num_threads(4)
    lock = read_json(args.selection)
    candidate = lock['candidates'][args.component]
    if lock['task'] != 'chest' or candidate['config']['backbone'] != 'biovil-t':
        raise ValueError('The audit requires the locked BioViL chest component')
    frame, audit, arrays, signature = store('chest', args.cohort, pixels_location(args, candidate['pixels']))
    if audit['manifest_sha256'] != lock['manifest'] or signature != candidate['config']['pixel_sha256']:
        raise ValueError('Test cohort and pixel cache must agree with the frozen component')
    labels = lock['labels']
    annotations = read_boxes(args.boxes, frame, labels)
    indices = np.asarray(sorted(annotations))
    for relative in frame.iloc[indices].path:
        check_original_size(checked(args.root, relative))

    model, saved = load_component(args.work, candidate, args.weights, None)
    results, captured = [], []
    with model.local_head.register_forward_hook(lambda _model, _input, out: captured.append(out.detach())):
        for start in range(0, len(indices), args.batch):
            rows = indices[start:start + args.batch]
            with torch.autocast('cuda', dtype=torch.float16):
                model(transfer(arrays['pixels'], rows))
            maps = captured.pop().float().cpu().numpy()
            for index, score in zip(rows, maps, strict=True):
                patient = frame.iloc[index].patient
                for label, boxes in annotations[index].items():
                    evidence = grid_evidence(score[label], boxes, saved['config']['resolution'])
                    results.append((labels[label], patient, evidence))
            if (start // args.batch + 1) % 25 == 0:
                print(f'Scored {min(start + args.batch, len(indices))} of {len(indices)} annotated images', flush=True)

    report = {'selection_sha256': checksum(args.selection), 'checkpoint_sha256': candidate['checkpoint_sha256'],
              'box_sha256': checksum(args.boxes), 'images': len(indices),
              'annotations': sum(len(boxes) for labels in annotations.values() for boxes in labels.values()),
              'method': 'Layer-3 local score without flip pooling, highest location evaluated against any box for the finding',
              'bootstrap_repeats': args.bootstrap, **summarise(results, args.bootstrap)}
    write_json(args.output, report)
    print(report['overall'], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cohort', 'boxes', 'selection', 'work', 'previous', 'weights', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    parser.add_argument('--component', default='biovil384')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.set_defaults(task='chest')
    args = parser.parse_args()
    if args.batch < 1 or args.bootstrap < 1:
        parser.error('Use a positive batch size and bootstrap count')
    evaluate(args)


if __name__ == '__main__':
    main()
