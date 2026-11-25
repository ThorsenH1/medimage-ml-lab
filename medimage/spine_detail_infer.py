"""Lock and run the equal blend of the reference and new spatial MRI graders."""
from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from medimage.common import checksum, read_json, write_json
from medimage.spine_data import SEVERITIES, TARGETS, prepare
from medimage.spine_heatmap import DiscHeatmap, anchor_inputs, observed_landmarks
from medimage.spine_heatmap_crossfit import anatomy_offsets, predict_heatmaps
from medimage.spine_localize import SpineCache, inference_regions, make_regions, read_folds
from medimage.spine_metrics import read_submission, submission
from medimage.spine_model import ROI_SIZE, SpineClassifier
from medimage import spine_native
from medimage.spine_train import RegionDataset, predict


def lock(args):
    cache = SpineCache.load(args.train_cache)
    folds = read_folds(args.folds, cache.studies)
    audit = read_json(args.learned / 'localization.json')
    comparison = read_json(args.comparison)
    config = audit['config']
    fold_hash = checksum(args.folds)
    base_digest = checksum(args.train_cache / 'audit.json')
    if (config['folds_file_sha256'] != fold_hash or comparison['folds_sha256'] != fold_hash
            or config['cache_sha256'] != base_digest):
        raise ValueError('The deployment bundle must match the frozen training study splits')
    grading_digest = base_digest
    sampling = {}
    if args.native_train_cache:
        native_file = args.native_train_cache / 'audit.json'
        native = read_json(native_file)['config']
        grade_config = read_json(args.native_regions / 'localization.json')['config']
        grading_digest = checksum(native_file)
        if (native.get('reprojected_from') != base_digest or grade_config['cache_sha256'] != grading_digest
                or grade_config['folds_file_sha256'] != fold_hash
                or grade_config['original_localization_sha256'] != checksum(args.learned / 'localization.json')
                or native['preprocess_sha256'] != checksum(Path(spine_native.__file__))):
            raise ValueError('Native grading cache and cross-fitted regions do not share frozen geometry')
        sampling = {'native_size': native['size'], 'preprocess_sha256': native['preprocess_sha256']}
    _, geometry = anchor_inputs(cache, np.arange(len(folds)))
    xy = observed_landmarks(cache, geometry)
    entries = []
    for fold in range(5):
        key = str(fold)
        folder = args.runs / f'fold-{fold}'
        grade = comparison['folds'][key]
        learned = audit['held_out_localization'][key]
        recorded = read_json(folder / 'result.json')['config']
        region_digest = (checksum(args.native_regions / f'fold-{fold}.npz')
                         if sampling else learned['region_sha256'])
        crop_resolution = recorded.get('resolution', ROI_SIZE)
        if (checksum(folder / 'final.pt') != grade['checkpoint_sha256']
                or checksum(folder / 'oof.csv') != grade['prediction_sha256']
                or recorded['fold'] != fold or recorded['region_sha256'] != region_digest
                or recorded['cache_sha256'] != grading_digest or (not sampling and crop_resolution != ROI_SIZE)):
            raise ValueError('Stored model artifacts do not match the measured grading result')
        fit = np.flatnonzero(folds != fold)
        center = args.learned / 'models' / f'exclude-{fold}' / 'final.pt'
        center_hash = learned['checkpoints'][key]
        if checksum(center) != center_hash:
            raise ValueError('The held-out localizer differs from the cross-fit report')
        entry = {'fold': fold, 'heatmap_sha256': center_hash, 'grader_sha256': grade['checkpoint_sha256'],
                 'reference_localizer_sha256': checksum(args.reference / f'localizer-{fold}.npz'),
                 'offsets': anatomy_offsets(cache, xy, geometry, fit).tolist()}
        if sampling:
            entry['crop_resolution'] = crop_resolution
        entries.append(entry)
    write_json(args.output, {'reference_fraction': .5, 'folds_sha256': fold_hash,
                             'comparison_sha256': checksum(args.comparison),
                             'reference_selection_sha256': checksum(args.reference_selection),
                             'reference_features_sha256': checksum(args.reference / 'reference-features.npy'),
                             **sampling,
                             'models': entries})
    print('Locked the five measured graders and their geometry for inference.', flush=True)


def native_inference_cache(root: Path, base: SpineCache, output: Path, plan: dict) -> SpineCache:
    size = plan['native_size']
    digest = checksum(Path(spine_native.__file__))
    if digest != plan['preprocess_sha256'] or output.resolve() == base.path.resolve():
        raise ValueError('Native prediction requires the locked preprocessing and a separate cache')
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent) as temporary:
            staging = Path(temporary) / 'native'
            spine_native.prepare(argparse.Namespace(root=root, base=base.path, output=staging,
                                                     size=size, workers=4))
            try:
                staging.rename(output)
            except FileExistsError:
                pass
    config = read_json(output / 'audit.json')['config']
    native = SpineCache.load(output)
    if (config['size'] != size or config['preprocess_sha256'] != digest
            or config['reprojected_from'] != checksum(base.path / 'audit.json')
            or not np.array_equal(native.studies, base.studies) or not np.array_equal(native.valid, base.valid)):
        raise ValueError('The native prediction cache differs from the locked sampling geometry')
    return native


def infer(args):
    torch.set_num_threads(4)
    plan = read_json(args.plan)
    reference_plan = read_json(args.prior.with_suffix('.json'))
    if (reference_plan['selection_sha256'] != plan['reference_selection_sha256']
            or reference_plan['prediction_sha256'] != checksum(args.prior)
            or checksum(args.reference / 'reference-features.npy') != plan['reference_features_sha256']):
        raise ValueError('Reference model output or geometry differs from the deployment lock')
    if [entry['fold'] for entry in plan['models']] != list(range(5)):
        raise ValueError('Expected all five locked MRI models in fold order')
    if bool(args.native_cache) != ('native_size' in plan):
        raise ValueError('Provide a separate --native-cache for a native-resolution plan')
    prepare(args.root, args.cache, 'test', 4)
    cache = SpineCache.load(args.cache)
    grading_cache = native_inference_cache(args.root, cache, args.native_cache, plan) if args.native_cache else cache
    rows = np.arange(len(cache.studies))
    reference = read_submission(args.prior, cache.studies)
    labels = np.full((len(rows), 25), -1, dtype=np.int64)
    inputs, geometry = anchor_inputs(cache, rows)
    folds = []
    first_regions = None
    for entry in plan['models']:
        fold = entry['fold']
        heatmap = args.learned / 'models' / f'exclude-{fold}' / 'final.pt'
        graded = args.runs / f'fold-{fold}' / 'final.pt'
        if (checksum(heatmap) != entry['heatmap_sha256'] or checksum(graded) != entry['grader_sha256']
                or checksum(args.reference / f'localizer-{fold}.npz') != entry['reference_localizer_sha256']):
            raise ValueError('A model artifact differs from the deployment lock')
        state = torch.load(heatmap, map_location='cpu', weights_only=True)
        localizer = DiscHeatmap().cuda().eval()
        localizer.load_state_dict(state['state'])
        points = predict_heatmaps(localizer, inputs, args.batch)
        del localizer
        _, old_metadata = inference_regions(cache, args.reference, fold)
        regions, metadata, _ = make_regions(cache, rows, geometry, points, old_metadata[:, :5, 1],
                                            {'offsets': np.asarray(entry['offsets'], dtype=np.float32)})
        resolution = entry.get('crop_resolution', ROI_SIZE)
        if args.native_cache:
            regions = spine_native.reproject_regions(regions, cache.geometry, grading_cache.geometry)
        state = torch.load(graded, map_location='cpu', weights_only=True)
        if state['config'].get('resolution', ROI_SIZE) != resolution:
            raise ValueError('The grader crop size differs from the deployment lock')
        model = SpineClassifier(pretrained=False, spatial_detail=True).cuda()
        model.load_state_dict(state['model'])
        dataset = RegionDataset(grading_cache, rows, regions, metadata, labels, resolution=resolution)
        probs, _ = predict(model, DataLoader(dataset, batch_size=args.batch, pin_memory=True))
        folds.append(probs)
        if args.trace and first_regions is None:
            first_regions = {'fold': fold, 'coordinates': regions[:, :25].tolist(), 'quality': metadata[:, :25].tolist()}
            if args.native_cache:
                first_regions.update(crop_resolution=resolution, cache_size=plan['native_size'])
        del model
    model_probabilities = np.mean(folds, axis=0, dtype=np.float64)
    fraction = plan['reference_fraction']
    blended = fraction * reference + (1 - fraction) * model_probabilities
    desired = pd.read_csv(args.root / 'sample_submission.csv')
    result = submission(cache.studies, blended)
    if set(result.row_id) != set(desired.row_id) or list(desired.columns) != ['row_id', *SEVERITIES]:
        raise ValueError('Final MRI predictions differ from the requested target rows')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.set_index('row_id').loc[desired.row_id].reset_index().to_csv(args.output, index=False)
    np.testing.assert_allclose(read_submission(args.output, cache.studies), blended, atol=1e-8)
    digest = checksum(args.plan)
    report = {'selection_sha256': digest, 'prediction_sha256': checksum(args.output), 'rows': len(result),
              'studies': len(rows), 'selected': 'equal blend of reference and spatial-detail MRI graders'}
    write_json(args.output.with_suffix('.json'), report)
    if args.trace:
        write_json(args.trace, {'task': 'spine', 'selection_sha256': digest, 'studies': cache.studies.tolist(),
                                'targets': list(TARGETS), 'reference_fraction': fraction, 'new_model_fraction': 1 - fraction,
                                'reference_probabilities': reference.tolist(), 'new_model_probabilities': model_probabilities.tolist(),
                                'folds': list(range(5)), 'fold_probabilities': np.asarray(folds).tolist(),
                                'blended_probabilities': blended.tolist(), 'first_fold_regions': first_regions})
    print(f"Verified {len(result)} prediction rows from the locked MRI ensemble.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    shared = ('learned', 'reference', 'runs', 'output')
    build = sub.add_parser('lock')
    run = sub.add_parser('predict')
    for name in (*shared, 'train-cache', 'folds', 'comparison', 'reference-selection'):
        build.add_argument(f'--{name}', type=Path, required=True)
    build.add_argument('--native-train-cache', type=Path)
    build.add_argument('--native-regions', type=Path)
    for name in (*shared, 'plan', 'root', 'cache', 'prior'):
        run.add_argument(f'--{name}', type=Path, required=True)
    run.add_argument('--batch', type=int, default=8)
    run.add_argument('--native-cache', type=Path)
    run.add_argument('--trace', type=Path)
    args = parser.parse_args()
    if args.phase == 'lock':
        if bool(args.native_train_cache) != bool(args.native_regions):
            parser.error('Provide the native training cache and native regions together')
        lock(args)
    else:
        if args.batch < 1:
            parser.error('Batch size must be positive')
        outcomes = [args.output.resolve(), args.output.with_suffix('.json').resolve()]
        if args.trace and args.trace.resolve() in outcomes:
            parser.error('Use different paths for the submission, summary and trace')
        infer(args)


if __name__ == '__main__':
    main()
