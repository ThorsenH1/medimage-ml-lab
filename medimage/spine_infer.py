from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from medimage.common import checksum, read_json, write_json
from medimage.spine_data import SEVERITIES, prepare
from medimage.spine_localize import SpineCache, inference_regions
from medimage.spine_metrics import read_submission, submission
from medimage.spine_model import SpineClassifier
from medimage.spine_train import RegionDataset, predict


def infer(args: argparse.Namespace) -> None:
    begin = time.monotonic()
    prepare(args.root, args.cache, 'test', args.workers)
    cache = SpineCache.load(args.cache)
    decision = read_json(args.assessment)
    selected = decision['selected_for_local_inference']
    modes = {'localized', 'global_reference', 'equal_weight_blend'}
    if selected not in modes:
        raise ValueError('Assessment selected an unsupported inference configuration')
    if selected in {'global_reference', 'equal_weight_blend'} and args.baseline_submission is None:
        raise ValueError('Selected configuration requires the baseline full-cohort submission')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_rows = np.arange(len(cache.studies))
    missing_labels = np.full((len(all_rows), 25), -1, dtype=np.int64)
    weights_audit = []
    model = SpineClassifier(pretrained=False).cuda()
    predictions = []
    for fold in range(5):
        model_path = args.runs / f'fold-{fold}' / 'final.pt'
        result_path = args.runs / f'fold-{fold}' / 'result.json'
        if read_json(result_path) != decision['fold_experiments'][fold]:
            raise ValueError('Inference fold differs from the assessed fold')
        saved = torch.load(model_path, map_location='cpu', weights_only=True)
        if saved['config'] != decision['fold_experiments'][fold]['config']:
            raise ValueError('Inference checkpoint differs from assessment configuration')
        model.load_state_dict(saved['model'])
        coords, metadata = inference_regions(cache, args.localization, fold)
        dataset = RegionDataset(cache, all_rows, coords, metadata, missing_labels)
        probabilities, _ = predict(model, DataLoader(dataset, batch_size=args.batch_size, pin_memory=True))
        predictions.append(probabilities.astype(np.float64))
        weights_audit.append({'fold': fold, 'model_sha256': checksum(model_path),
                              'localizer_sha256': checksum(args.localization / f'localizer-{fold}.npz'),
                              'missing_primary_regions': int((metadata[:, :25, 0] == 0).sum()),
                              'fallback_keypoints': int((metadata[:, :5, 1] == 0).sum())})
    localized = np.mean(predictions, axis=0)
    if selected == 'localized':
        chosen = localized
    else:
        global_reference = read_submission(args.baseline_submission, cache.studies)
        chosen = global_reference if selected == 'global_reference' else (global_reference + localized) / 2.
    template = pd.read_csv(args.root / 'sample_submission.csv')
    if list(template.columns) != ['row_id', *SEVERITIES] or template.row_id.duplicated().any():
        raise ValueError('Invalid sample submission schema')
    for path, probabilities in ((args.output.with_suffix('.localized.csv'), localized), (args.output, chosen)):
        table = submission(cache.studies, probabilities).set_index('row_id')
        if set(table.index) != set(template.row_id):
            raise ValueError('Inference study IDs differ from the provided competition template')
        tmp = path.with_suffix('.partial')
        table.loc[template.row_id].reset_index().to_csv(tmp, index=False)
        emitted = pd.read_csv(tmp)
        if not emitted.row_id.equals(template.row_id):
            raise AssertionError('Written submission row order differs from the template')
        reconstructed = read_submission(tmp, cache.studies)
        np.testing.assert_allclose(reconstructed, probabilities, rtol=1e-6, atol=1e-7)
        os.replace(tmp, path)
    report = {'configuration': selected, 'study_count': len(cache.studies), 'rows': len(template),
              'assessment_sha256': checksum(args.assessment), 'gpu': torch.cuda.get_device_name(),
              'test_cache_issues': read_json(args.cache / 'audit.json')['issues'], 'folds': weights_audit,
              'seconds': round(time.monotonic() - begin, 2)}
    if args.baseline_submission:
        report['baseline_submission_sha256'] = checksum(args.baseline_submission)
    write_json(args.output.with_suffix('.json'), report)
    print(f'Validated {len(template)} raw-MRI submission rows ({selected})', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Decode raw MRI, localize, classify and validate RSNA submission rows')
    for name in ('root', 'cache', 'assessment', 'localization', 'runs', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--baseline-submission', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=10)
    infer(parser.parse_args())


if __name__ == '__main__':
    main()
