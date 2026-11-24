from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from medimage.auc_train import new_model, predict, store
from medimage.common import checksum, read_json, write_json
from medimage.radiography_train import save_checkpoint, validation_metrics


def validate(args):
    frame, audit, arrays, pixel_sha = store('mammo', args.cohort, args.pixels)
    source = args.parent / 'best.pt'
    result = read_json(args.parent / 'result.json')
    if checksum(source) != result['best_sha256']:
        raise ValueError('The source checkpoint changed after its original validation')
    saved = torch.load(source, map_location='cpu', weights_only=True)
    config = saved['config']
    if config != result['config'] or config['manifest'] != audit['manifest_sha256']:
        raise ValueError('The source checkpoint and cohort differ')
    if config['task'] != 'mammo' or config['pixel_sha256'] != pixel_sha or config['tta']:
        raise ValueError('Expected a spatial mammography model using the same pixel cache without TTA')
    config = {**config, 'tta': True, 'tta_recipe': 'gentle-rotation-gamma', 'original_sha256': result['best_sha256']}
    model = new_model(config, args.weights).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(saved['model'])
    rows = np.flatnonzero(frame.split.eq('val').to_numpy())
    truth = frame.iloc[rows][['positive']].to_numpy(np.float32)
    scores = predict(model, 'mammo', arrays, rows, min(config['batch'], 2), True,
                     np.asarray(saved['prior_correction'], dtype=np.float32))
    metrics = validation_metrics(truth, scores)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / 'best.pt'
    save_checkpoint(checkpoint, {**saved, 'config': config, 'validation': metrics})
    np.savez_compressed(args.output / 'validation.partial.npz', rows=rows, logits=scores)
    validation = args.output / 'validation.npz'
    (args.output / 'validation.partial.npz').replace(validation)
    write_json(args.output / 'result.json', {'config': config, 'epochs': [{'epoch': saved['epoch'], **metrics}],
                                            'best_sha256': checksum(checkpoint),
                                            'validation_sha256': checksum(validation)})
    print({'source': args.parent.name, 'validation_tta': metrics}, flush=True)


def main():
    parser = argparse.ArgumentParser(description='Validate predetermined mammography augmentation on a trained model')
    for name in ('cohort', 'pixels', 'weights', 'parent', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    torch.set_num_threads(4)
    validate(parser.parse_args())


if __name__ == '__main__':
    main()
