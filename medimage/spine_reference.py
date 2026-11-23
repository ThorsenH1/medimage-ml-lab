from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch import nn
from torchvision.models import resnet18

from medimage.common import checksum, write_json
from medimage.spine_data import SERIES, SEVERITIES, read_slice
from medimage.spine_metrics import submission


class GlobalReference(nn.Module):
    """The checkpoint-compatible architecture of the historical 224px 2.5D model."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.encoder.fc = nn.Identity()
        self.view_embedding = nn.Parameter(torch.zeros(3, 512))
        self.head = nn.Sequential(nn.LayerNorm(1539), nn.Dropout(.25), nn.Linear(1539, 75))

    def forward(self, images: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        batch, views, depth, height, width = images.shape
        encoded = self.encoder(images.reshape(batch * views * depth, 1, height, width))
        pooled = encoded.reshape(batch, views, depth, 512).mean(dim=2)
        combined = (pooled + self.view_embedding.unsqueeze(0)) * present.unsqueeze(-1)
        return self.head(torch.cat([combined.flatten(1), present], dim=1)).reshape(batch, 25, 3)


def reference_pixels(path: Path) -> np.ndarray:
    item = pydicom.dcmread(path)
    pixels = item.pixel_array.astype(np.float32)
    slope = float(getattr(item, 'RescaleSlope', 1.) or 1.)
    intercept = float(getattr(item, 'RescaleIntercept', 0.) or 0.)
    pixels = pixels * slope + intercept
    if str(getattr(item, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
        pixels = float(np.nanmax(pixels) + np.nanmin(pixels)) - pixels
    valid = np.isfinite(pixels)
    if not valid.any():
        raise ValueError('Historical MRI reference contains no finite pixels')
    low, high = np.percentile(pixels[valid], [1, 99])
    normalized = np.clip((pixels - low) / (high - low), 0, 1) if high > low else np.zeros_like(pixels)
    normalized = normalized.astype(np.float32, copy=False)
    normalized[~valid] = 0
    height, width = pixels.shape
    scale = 224 / max(height, width)
    width, height = max(1, round(width * scale)), max(1, round(height * scale))
    x, y = (224 - width) // 2, (224 - height) // 2
    result = np.zeros((224, 224), dtype=np.uint8)
    resized = cv2.resize(normalized, (width, height), interpolation=cv2.INTER_AREA)
    result[y:y + height, x:x + width] = np.rint(resized * 255).astype(np.uint8)
    return result


def extract_views(root: Path, studies: np.ndarray, descriptions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    volumes = np.zeros((len(studies), 3, 5, 224, 224), dtype=np.uint8)
    present = np.zeros((len(studies), 3), dtype=np.float32)
    indexed = {int(study): frame for study, frame in descriptions.groupby('study_id')}
    for index, study in enumerate(studies):
        frame = indexed[int(study)]
        for view, description in enumerate(SERIES):
            ids = frame.loc[frame.series_description.eq(description), 'series_id']
            if ids.empty:
                continue
            candidates = [(int(uid), sorted((root / 'test_images' / str(study) / str(uid)).glob('*.dcm'))) for uid in ids]
            _, paths = max(candidates, key=lambda item: (len(item[1]), -item[0]))
            if not paths:
                raise FileNotFoundError('A required series exists in the CSV but has no DICOM files')
            headers = [read_slice(path) for path in paths]

            def position(header):
                normal = np.cross(header.column_step, header.row_step)
                return float(np.dot(header.origin, normal / np.linalg.norm(normal)))

            headers.sort(key=position)
            picks = np.rint(np.linspace(0, len(headers) - 1, 5)).astype(int)
            for i, position_index in enumerate(picks):
                volumes[index, view, i] = reference_pixels(headers[position_index].path)
            present[index, view] = 1.
    return volumes, present


@torch.inference_mode()
def predict_reference(root: Path, checkpoints: list[Path], *, batch_size: int,
                      trusted_legacy_format: bool) -> tuple[np.ndarray, np.ndarray]:
    if len(checkpoints) != 5 or len({path.resolve() for path in checkpoints}) != 5:
        raise ValueError('The historical reference requires five distinct fold checkpoints')
    models = []
    for path in checkpoints:
        saved = torch.load(path, map_location='cpu', weights_only=not trusted_legacy_format)
        model = GlobalReference().cuda().eval()
        model.load_state_dict(saved.get('model', saved))
        models.append(model)
    desc = pd.read_csv(root / 'test_series_descriptions.csv')
    studies = np.sort(desc.study_id.unique())
    predictions = []
    for start in range(0, len(studies), batch_size):
        current = studies[start:start + batch_size]
        images, masks = extract_views(root, current, desc.loc[desc.study_id.isin(current)])
        x = (torch.from_numpy(images).cuda().float().div(255.) - .45) / .225
        mask = torch.from_numpy(masks).cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            values = [model(x, mask).softmax(dim=-1).float().cpu().numpy() for model in models]
        predictions.append(np.mean(values, axis=0, dtype=np.float64))
    return studies, np.concatenate(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description='Reproduce raw-DICOM inference for the historical RSNA global reference')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs=5, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--trusted-legacy-format', action='store_true',
                        help='Accept legacy training checkpoints only from a trusted local experiment')
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('Inference batch size must be positive')
    studies, values = predict_reference(args.root, args.checkpoints, batch_size=args.batch_size,
                                        trusted_legacy_format=args.trusted_legacy_format)
    frame = submission(studies, values)
    template = pd.read_csv(args.root / 'sample_submission.csv')
    if (list(template.columns) != ['row_id', *SEVERITIES] or template.row_id.duplicated().any()
            or set(frame.row_id) != set(template.row_id)):
        raise ValueError('Historical baseline output differs from competition row schema')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.set_index('row_id').loc[template.row_id].reset_index().to_csv(args.output, index=False)
    write_json(args.output.with_suffix('.json'), {'studies': len(studies), 'rows': len(frame),
                                                 'checkpoint_sha256': [checksum(path) for path in args.checkpoints]})
    print(f'Validated {len(frame)} global-reference rows from raw DICOM', flush=True)


if __name__ == '__main__':
    main()
