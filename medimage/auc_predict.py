from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.special import expit

from medimage.auc_mammo_data import view_pixels
from medimage.auc_selection import component_logits, load_component, selected_scores
from medimage.common import checksum, read_json, write_json
from medimage.search_features import chest_source
from medimage.search_predict import radiography_logits


def arrays_for(task, shape, args):
    if task == 'chest':
        return {'pixels': chest_source(args.image, shape[0])[None]}
    images = (args.cc, args.mlo)
    pixels = np.stack([view_pixels(image, tuple(shape))[0] if image else np.zeros(shape, dtype=np.uint8)
                       for image in images])
    return {'pixels': pixels[None], 'views': np.asarray([[image is not None for image in images]], dtype=bool)}


def evaluate(args):
    torch.set_num_threads(4)
    lock = read_json(args.selection)
    task, selected = lock['task'], lock['selected']
    if task == 'chest':
        if args.image is None or args.cc is not None or args.mlo is not None:
            raise ValueError('Use a single frontal image for the selected chest model')
        projections = ['frontal']
    else:
        if args.image is not None or (args.cc is None and args.mlo is None):
            raise ValueError('Provide one or both full mammography projections')
        projections = [name for name, file in (('CC', args.cc), ('MLO', args.mlo)) if file]
    logits = {}
    for name in selected['weights']:
        component = lock['candidates'][name]
        if name == 'previous-locked':
            previous = args.previous / 'evaluation' / task / 'selection.json'
            if checksum(previous) != component['prior_selection_sha256']:
                raise ValueError('The previous inference plan differs from the selected model')
            settings = SimpleNamespace(image=args.image, cc=args.cc, mlo=args.mlo, weights=args.weights,
                                       search=args.previous / 'runs' / task, original=args.original)
            logits[name] = radiography_logits(settings, read_json(previous))[0][None]
        else:
            model, saved = load_component(args.work, component, args.weights, args.clip)
            pixels = arrays_for(task, component['image_shape'], args)
            if saved['config']['backbone'] == 'chest-fusion':
                from medimage.auc_fusion import raw_prediction

                correction = np.asarray(saved['prior_correction'], dtype=np.float32)
                logits[name] = raw_prediction(model, saved['config'], pixels['pixels'], args.weights, correction)
            else:
                logits[name] = component_logits(model, saved, pixels, np.zeros(1, dtype=np.int64))
            del model
        torch.cuda.empty_cache()
    combined = selected_scores(selected, logits, lock['component_calibration'])[0]
    calibration = np.asarray(lock['ensemble_calibration'])
    probabilities = expit(combined * calibration[:, 0] + calibration[:, 1])
    if not np.isfinite(probabilities).all():
        raise ValueError('Expected finite calibrated research scores')
    values = {label: {'probability': float(probability), 'validation_threshold': threshold,
                      'threshold_crossed': bool(probability >= threshold)}
              for label, probability, threshold in zip(lock['labels'], probabilities, lock['thresholds'], strict=True)}
    write_json(args.output, {'task': task, 'purpose': 'research and education', 'projections': projections,
                             'selection_sha256': checksum(args.selection), 'scores': values})


def main():
    parser = argparse.ArgumentParser(description='Reproduce the frozen radiography ensemble on raw input pixels')
    for name in ('selection', 'work', 'previous', 'original', 'weights', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    for name in ('image', 'cc', 'mlo', 'clip'):
        parser.add_argument(f'--{name}', type=Path)
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
