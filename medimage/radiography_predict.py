from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from medimage.common import checksum, read_json, write_json
from medimage.radiography_data import MAMMO_SHAPE, prepare_breast, resize_chest
from medimage.radiography_models import build_model


@torch.inference_mode()
def predict(args: argparse.Namespace) -> None:
    calibration = read_json(args.calibration)
    if calibration['checkpoint_sha256'] != checksum(args.checkpoint):
        raise ValueError('Calibration was fitted for another checkpoint')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    task, variant = saved['config']['task'], saved['config']['variant']
    if task == 'chest':
        if args.image is None or args.cc is not None or args.mlo is not None:
            raise ValueError('A ChestX-ray14 checkpoint needs exactly one --image')
        images, views = resize_chest(args.image)[None], np.ones((1, 1), dtype=bool)
        input_views = ['frontal']
    elif task == 'mammo':
        if (args.cc is None and args.mlo is None) or args.image is not None:
            raise ValueError('A CBIS-DDSM checkpoint needs at least one full --cc or --mlo mammogram')
        paths = (args.cc, args.mlo)
        images = np.stack([prepare_breast(path) if path else np.zeros(MAMMO_SHAPE, np.uint8) for path in paths])[None]
        views = np.array([[path is not None for path in paths]], dtype=bool)
        input_views = [name for name, path in zip(('CC', 'MLO'), paths, strict=True) if path is not None]
    else:
        raise ValueError('Unrecognized checkpoint task')
    model = build_model(task, variant, pretrained=False).to(args.device)
    model.load_state_dict(saved['model'])
    model.eval()
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.device == 'cuda'):
        z = model(torch.from_numpy(images).to(args.device), torch.from_numpy(views).to(args.device))
    params = np.asarray(calibration['slope_intercept'])
    probability = expit(z[0].float().cpu().numpy() * params[:, 0] + params[:, 1])
    result = {'purpose': 'research and education', 'task': task, 'views': input_views,
              'checkpoint_sha256': calibration['checkpoint_sha256'],
              'probabilities': dict(zip(calibration['labels'], probability.tolist(), strict=True))}
    write_json(args.output, result)
    print(f'Wrote research probabilities for {len(result["probabilities"])} outputs')


def main() -> None:
    parser = argparse.ArgumentParser(description='Run local radiography inference with locked calibration')
    for name in ('checkpoint', 'calibration', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    for name in ('image', 'cc', 'mlo'):
        parser.add_argument(f'--{name}', type=Path)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    predict(parser.parse_args())


if __name__ == '__main__':
    main()
