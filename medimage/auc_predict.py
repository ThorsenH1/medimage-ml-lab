from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.special import expit

from medimage.auc_mammo_data import view_pixels
from medimage.auc_selection import component_logits, load_component, selected_scores, weighted_components
from medimage.common import checksum, read_json, write_json
from medimage.mammo_native import PIXEL_FORMAT, native_view
from medimage.search_features import chest_source
from medimage.search_predict import radiography_logits


def arrays_for(task, shape, args, pixel_format=None):
    if task == 'chest':
        return {'pixels': chest_source(args.image, shape[0])[None]}
    if pixel_format == PIXEL_FORMAT:
        images, loader, dtype = (args.native_cc, args.native_mlo), native_view, np.uint16
    else:
        images, loader, dtype = (args.cc, args.mlo), view_pixels, np.uint8
    pixels = np.stack([loader(image, tuple(shape))[0] if image else np.zeros(shape, dtype=dtype)
                       for image in images])
    return {'pixels': pixels[None], 'views': np.asarray([[image is not None for image in images]], dtype=bool)}


def prediction_trace(lock, logits, probabilities, projections, checksum_value):
    selection = lock['selected']
    weights = selection.get('label_weights', selection['weights'])
    calibration = lock['component_calibration'] if selection['method'].startswith('calibrated-') else {}
    parts = dict(weighted_components(logits, weights, calibration))
    combined = selected_scores(selection, logits, lock['component_calibration'])[0]
    return {'task': lock['task'], 'projections': projections, 'selection_sha256': checksum_value,
            'blend_method': selection['method'], 'labels': lock['labels'],
            'components': {name: {'weight': weights[name], 'raw_logit': scores[0].tolist(),
                                  'weighted_logit': parts[name][0].tolist(),
                                  'model_input_shape': lock['candidates'][name].get('image_shape')}
                           for name, scores in logits.items()},
            'fused_logit': combined.tolist(), 'calibration': lock['ensemble_calibration'],
            'probability': probabilities.tolist(), 'validation_threshold': lock['thresholds']}


def evaluate(args):
    torch.set_num_threads(4)
    lock = read_json(args.selection)
    task, selected = lock['task'], lock['selected']
    native_inputs = (getattr(args, 'native_cc', None), getattr(args, 'native_mlo', None))
    if task == 'chest':
        if args.image is None or any((args.cc, args.mlo, *native_inputs)):
            raise ValueError('Use a single frontal image for the selected chest model')
        projections = ['frontal']
    else:
        formats = {lock['candidates'][name].get('config', {}).get('pixel_format') for name in selected['weights']}
        if formats - {None, PIXEL_FORMAT}:
            raise ValueError('The locked model uses an unsupported mammography representation')
        required = [(args.cc, args.mlo) if kind is None else native_inputs for kind in formats]
        supplied = {tuple(bool(file) for file in pair) for pair in required}
        if args.image is not None or len(supplied) != 1 or not all(map(any, required)):
            raise ValueError('Provide matching CC/MLO projections for all selected JPEG and DICOM components')
        projections = [name for name, file in zip(('CC', 'MLO'), required[0], strict=True) if file]
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
            pixels = arrays_for(task, component['image_shape'], args, saved['config'].get('pixel_format'))
            if saved['config']['backbone'] == 'chest-fusion':
                from medimage.auc_fusion import raw_prediction

                correction = np.asarray(saved['prior_correction'], dtype=np.float32)
                logits[name] = raw_prediction(model, saved['config'], pixels['pixels'], args.weights, correction)
            else:
                logits[name] = component_logits(model, saved, pixels, np.zeros(1, dtype=np.int64), args.weights)
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
    selection_sha256 = checksum(args.selection)
    write_json(args.output, {'task': task, 'purpose': 'research and education', 'projections': projections,
                             'selection_sha256': selection_sha256, 'scores': values})
    if args.trace:
        write_json(args.trace, prediction_trace(lock, logits, probabilities, projections, selection_sha256))


def main():
    parser = argparse.ArgumentParser(description='Reproduce the frozen radiography ensemble on raw input pixels')
    for name in ('selection', 'work', 'previous', 'original', 'weights', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    for name in ('image', 'cc', 'mlo', 'native-cc', 'native-mlo', 'clip'):
        parser.add_argument(f'--{name}', type=Path)
    parser.add_argument('--trace', type=Path, help='Write a separate JSON file for the scoring steps')
    args = parser.parse_args()
    if args.trace and args.trace.resolve() == args.output.resolve():
        parser.error('Use different output paths for scores and the trace')
    evaluate(args)


if __name__ == '__main__':
    main()
