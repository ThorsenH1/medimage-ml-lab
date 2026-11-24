from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from medimage.common import checksum, read_json, write_json
from medimage.radiography_data import MAMMO_SHAPE as ORIGINAL_MAMMO_SHAPE, prepare_breast, resize_chest
from medimage.radiography_models import build_model
from medimage.search_adapt import load_adapted
from medimage.search_features import MAMMO_SHAPE, Encoder, chest_source, mammography_tiles
from medimage.search_heads import FeatureClassifier, Recipe
from medimage.search_spine import SpineFeatureHead, SpineRecipe, probabilities
from medimage.search_spine_features import feature_rows
from medimage.spine_data import prepare
from medimage.spine_localize import SpineCache, inference_regions
from medimage.spine_metrics import apply_temperature, read_submission, submission


def checked_file(artifact: dict, original: Path, search: Path) -> Path:
    root = {'search': search, 'original': original}[artifact['base']].resolve()
    path = (root / artifact['path']).resolve()
    if not path.is_relative_to(root) or checksum(path) != artifact['sha256']:
        raise ValueError('Saved deployment artifacts changed or point outside their configured directory')
    return path


@torch.inference_mode()
def radiography_logits(args: argparse.Namespace, lock: dict) -> tuple[np.ndarray, list[str]]:
    task = lock['task']
    if task == 'chest':
        if args.image is None or args.cc is not None or args.mlo is not None:
            raise ValueError('The thorax selection expects one --image')
        size = chest_source(args.image, 224)[None]
        low = resize_chest(args.image)[None]
        mask = torch.ones((1, 5), device='cuda', dtype=torch.bool)
        old_mask = torch.ones((1, 1), device='cuda', dtype=torch.bool)
        projections = ['frontal']
    elif task == 'mammo':
        images = (args.cc, args.mlo)
        if args.image is not None or not any(images):
            raise ValueError('The mammography selection expects full --cc and/or --mlo images')
        size = np.stack([prepare_breast(path, MAMMO_SHAPE) if path else np.zeros(MAMMO_SHAPE, np.uint8) for path in images])[None]
        low = np.stack([prepare_breast(path) if path else np.zeros(ORIGINAL_MAMMO_SHAPE, np.uint8) for path in images])[None]
        old_mask = torch.tensor([[path is not None for path in images]], device='cuda', dtype=torch.bool)
        mask = old_mask.repeat_interleave(7, dim=1)
        projections = [view for view, path in zip(('CC', 'MLO'), images, strict=True) if path]
    else:
        raise ValueError('Deployment plan has an unsupported task')
    pixels = torch.from_numpy(size).cuda()
    values = np.zeros(len(lock['labels']), dtype=np.float64)
    for profile, ratio in lock['selected']['weights'].items():
        paths = [checked_file(artifact, args.original, args.search) for artifact in lock['checkpoints'][profile]]
        with torch.autocast('cuda', dtype=torch.float16):
            if profile == 'incumbent-global':
                checkpoint = torch.load(paths[0], map_location='cpu', weights_only=True)
                model = build_model(task, 'global', pretrained=False).cuda().eval()
                model.load_state_dict(checkpoint['model'])
                raw = model(torch.from_numpy(low).cuda(), old_mask)[0].float().cpu().numpy()
                values += ratio * (raw - lock['incumbent_logit_offset'])
                continue
            if profile.startswith('adapted-'):
                model, saved = load_adapted(paths[0], args.weights)
                raw = model(pixels, mask)[0].float().cpu().numpy()
                values += ratio * (raw - np.log(saved['weights']))
                continue
            info = lock['feature_profiles'][profile]
            encoder = Encoder(info['backbone'], info['size'], args.weights).cuda().eval()
            if encoder.weight_sha256 != info['pretrained_sha256']:
                raise ValueError('Pretrained source changed after final model selection')
            if task == 'chest':
                encoded = encoder(pixels, spatial=True).half()
            else:
                tile = mammography_tiles(pixels, info['size']).flatten(0, 2)
                base = encoder(tile[mask.reshape(-1)]).half()
                encoded = base.new_zeros(1, 14, encoder.dim)
                encoded[mask] = base
            for path in paths:
                saved = torch.load(path, map_location='cpu', weights_only=True)
                config = saved['config']
                if config['features'] != info or config['task'] != task:
                    raise ValueError('Head and pretrained profile are from different experiments')
                head = FeatureClassifier(encoder.dim, mask.shape[1], len(lock['labels']), Recipe(**config['recipe'])).cuda().eval()
                head.load_state_dict(saved['model'])
                raw = head(encoded, mask)[0].float().cpu().numpy()
                values += ratio / len(paths) * (raw - np.log(saved['loss_weights']))
    return values, projections


def radiography(args: argparse.Namespace) -> None:
    lock = read_json(args.selection)
    values, projections = radiography_logits(args, lock)
    task = lock['task']
    fitted = np.asarray(lock['slope_intercept'])
    probability = expit(values * fitted[:, 0] + fitted[:, 1])
    if not np.isfinite(probability).all():
        raise ValueError('Raw-image inference did not yield finite probabilities')
    write_json(args.output, {'task': task, 'purpose': 'research and education', 'model': lock['selected']['name'],
                             'selection_sha256': checksum(args.selection), 'projections': projections,
                             'probabilities': dict(zip(lock['labels'], probability.tolist(), strict=True))})
    print(f'Raw {task} image inference verified for {len(probability)} outputs', flush=True)


def spine(args: argparse.Namespace) -> None:
    decision = read_json(args.selection)
    prepare(args.root, args.cache, 'test', 4)
    cache = SpineCache.load(args.cache)
    rows = np.arange(len(cache.studies))
    encoder = Encoder('r18', 96, args.weights).cuda().eval()
    if encoder.weight_sha256 != decision['pretrained_sha256']:
        raise ValueError('MRI encoder evidence does not match the locked selection')
    selected = decision['selected']
    raw, calibrated = [], []
    for item in decision['folds']:
        coords, metadata = inference_regions(cache, args.localization, item['fold'])
        vectors = torch.from_numpy(feature_rows(encoder, cache, rows, coords)).cuda()
        information = torch.from_numpy(metadata).cuda()
        predictions = []
        for checkpoint in item['checkpoints']:
            path = checked_file({**checkpoint, 'base': 'search'}, args.search, args.search)
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['config']['fold'] != item['fold'] or saved['config']['recipe'] != item['recipe']:
                raise ValueError('MRI fold checkpoint does not match the locked configuration')
            model = SpineFeatureHead(SpineRecipe(**item['recipe'])).cuda()
            model.load_state_dict(saved['model'])
            predictions.append(probabilities(model, vectors, information, rows))
        prediction = np.mean(predictions, axis=0, dtype=np.float64)
        raw.append(prediction)
        calibrated.append(apply_temperature(prediction, np.asarray(item['temperature'])))
    fraction = selected['fraction']
    features = np.mean(calibrated if selected['calibrated'] else raw, axis=0)
    incumbent = read_submission(args.incumbent, cache.studies)
    probabilities_out = (1 - fraction) * incumbent + fraction * features
    template = pd.read_csv(args.root / 'sample_submission.csv')
    result = submission(cache.studies, probabilities_out)
    if set(result.row_id) != set(template.row_id) or list(result.columns) != list(template.columns):
        raise ValueError('MRI sample schema and newly trained prediction differ')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.set_index('row_id').loc[template.row_id].reset_index().to_csv(args.output, index=False)
    reloaded = read_submission(args.output, cache.studies)
    np.testing.assert_allclose(reloaded, probabilities_out, atol=1e-9)
    write_json(args.output.with_suffix('.json'), {'rows': len(result), 'studies': len(cache.studies),
                                                  'selected': selected['name'], 'selection_sha256': checksum(args.selection),
                                                  'checkpoint_folds': len(decision['folds'])})
    print(f'Raw MRI test inference verified for {len(result)} rows', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Load a locked search winner and predict from raw images')
    sub = parser.add_subparsers(dest='task', required=True)
    for task, options in (('radiography', ('original', 'image', 'cc', 'mlo')),
                          ('spine', ('root', 'cache', 'localization', 'incumbent'))):
        command = sub.add_parser(task)
        for name in ('selection', 'search', 'weights', 'output', *options):
            command.add_argument(f'--{name}', type=Path, required=name not in {'image', 'cc', 'mlo'})
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    if args.task == 'radiography':
        radiography(args)
    else:
        spine(args)


if __name__ == '__main__':
    main()
