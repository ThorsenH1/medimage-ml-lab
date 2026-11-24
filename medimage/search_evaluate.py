from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from medimage.common import checksum, read_json, write_json
from medimage.radiography_data import CHEST_LABELS
from medimage.radiography_evaluate import all_metrics, fit_calibration, patient_bootstrap, subgroups
from medimage.radiography_models import build_model
from medimage.radiography_train import make_loader, predict_batches, validation_metrics
from medimage.search_adapt import load_adapted, logits_for, pixel_store
from medimage.search_features import load_manifest
from medimage.search_heads import DeviceBank, FeatureClassifier, Recipe, class_weights, head_logits, objective


def _uncorrected_incumbent(task: str, cohort: Path, frame: pd.DataFrame, original: Path, split: str) -> np.ndarray:
    path = original / 'runs' / f'{task}-global' / 'best.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = build_model(task, 'global', pretrained=False).cuda()
    model.load_state_dict(saved['model'])
    loader = make_loader(task, cohort, frame, split, 96 if task == 'chest' else 24, 0)
    z, _ = predict_batches(model, loader)
    return z


def model_artifacts(original: Path, search: Path, profile: str, task: str, plan: dict) -> list[dict]:
    if profile == 'incumbent-global':
        path = original / 'runs' / f'{task}-global' / 'best.pt'
        return [{'base': 'original', 'path': path.relative_to(original).as_posix(), 'sha256': checksum(path)}]
    branch = 'adaptation-reference' if profile == 'adapted-incumbent-224' else 'adaptation'
    paths = ([search / branch / 'best.pt'] if profile.startswith('adapted-')
             else [search / 'final' / profile / str(seed) / 'model.pt' for seed in plan['seeds']])
    return [{'base': 'search', 'path': path.relative_to(search).as_posix(), 'sha256': checksum(path)} for path in paths]


def candidate_plan(values: dict[str, np.ndarray], truth: np.ndarray, task: str) -> list[dict]:
    outcomes = []

    def add(name: str, weights: dict[str, float]) -> None:
        z = sum(values[key] * weight for key, weight in weights.items())
        metrics = validation_metrics(truth, z)
        outcomes.append({'name': name, 'weights': weights, 'metrics': metrics, 'score': objective(task, metrics)})

    for profile in values:
        add(profile, {profile: 1.})
    profiles = sorted((item for item in outcomes if item['name'] != 'incumbent-global'),
                      key=lambda entry: (-entry['score'], entry['name']))
    for profile in profiles:
        name = profile['name']
        for weight in (.25, .50, .75):
            add(f'incumbent+{name}-{weight:.2f}', {'incumbent-global': 1. - weight, name: weight})
    if len(profiles) > 1:
        a, b = [entry['name'] for entry in profiles[:2]]
        add('top-two-new', {a: .5, b: .5})
        add('incumbent+top-two-new', {'incumbent-global': .5, a: .25, b: .25})
    return sorted(outcomes, key=lambda entry: (-entry['score'], len(entry['weights']), entry['name']))


def _feature_logits(profile: str, bank: DeviceBank, task: str, rows: np.ndarray,
                    checkpoints: list[dict], search: Path) -> np.ndarray:
    predictions = []
    for item in checkpoints:
        file = search / item['path']
        if checksum(file) != item['sha256']:
            raise ValueError('Finalist model weights changed after selection')
        checkpoint = torch.load(file, map_location='cpu', weights_only=True)
        config = checkpoint['config']
        if config['features'] != bank.info or config['task'] != task:
            raise ValueError(f'Feature profile {profile} has an incompatible checkpoint')
        model = FeatureClassifier(bank.shape[1], bank.shape[0], 14 if task == 'chest' else 1,
                                  Recipe(**config['recipe'])).cuda()
        model.load_state_dict(checkpoint['model'])
        predictions.append(head_logits(model, bank, rows) - np.log(checkpoint['loss_weights']))
    return np.mean(predictions, axis=0)


def evaluate(task: str, cohort: Path, features: Path, search: Path, original: Path, output: Path,
             pixels_root: Path, weights: Path) -> None:
    torch.set_num_threads(4)
    frame, audit = load_manifest(cohort)
    plan = read_json(search / 'search.json')
    if plan['manifest'] != audit['manifest_sha256'] or plan['task'] != task:
        raise ValueError('Tuning outputs belong to a different experiment')
    labels = CHEST_LABELS if task == 'chest' else ('malignancy_in_lesion_enriched_cohort',)
    truth = frame.loc[:, list(CHEST_LABELS) if task == 'chest' else ['positive']].to_numpy(dtype=np.float32, copy=True)
    val_rows = np.flatnonzero(frame.split.eq('val').to_numpy())
    train_rows = np.flatnonzero(frame.split.eq('train').to_numpy())
    offset = np.log(class_weights(truth[train_rows], .5))
    output.mkdir(parents=True, exist_ok=True)
    incumbent_path = original / 'runs' / f'{task}-global' / 'best.pt'
    incumbent_digest = checksum(incumbent_path)
    cache_path = output / 'incumbent-validation.npz'
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as saved:
            if saved['sha'].item() != incumbent_digest or saved['manifest'].item() != plan['manifest'] or not np.array_equal(saved['rows'], val_rows):
                raise ValueError('Incumbent validation cache is incompatible')
            incumbent = saved['logits']
    else:
        incumbent = _uncorrected_incumbent(task, cohort, frame, original, 'val') - offset
        np.savez_compressed(cache_path, rows=val_rows, logits=incumbent, sha=incumbent_digest, manifest=plan['manifest'])
    val = {'incumbent-global': incumbent}
    finalists = {entry['profile']: entry for entry in plan['finalists']}
    for profile in finalists:
        with np.load(search / 'validation' / f'{profile}.npz', allow_pickle=False) as record:
            if not np.array_equal(record['rows'], val_rows):
                raise ValueError('Candidate validation logits have incorrect row order')
            val[profile] = record['logits']
    adaptations = {}
    for directory in (['adaptation', 'adaptation-reference'] if task == 'chest' else ['adaptation']):
        adapted = read_json(search / directory / 'result.json')
        if adapted['config']['manifest'] != plan['manifest'] or adapted['config']['task'] != task:
            raise ValueError('Completed adaptation uses another cohort')
        adaptations[adapted['name']] = adapted
        with np.load(search / directory / 'validation.npz', allow_pickle=False) as record:
            if not np.array_equal(record['rows'], val_rows):
                raise ValueError('Adapted validation logits have incorrect row order')
            val[adapted['name']] = record['logits']
    decisions = candidate_plan(val, truth[val_rows], task)
    selected = decisions[0]
    chosen_logits = sum(val[name] * weight for name, weight in selected['weights'].items())
    params, thresholds = fit_calibration(truth[val_rows], chosen_logits)
    checkpoint_records = {profile: model_artifacts(original, search, profile, task, finalists.get(profile, {}))
                          for profile in selected['weights']}
    selected_feature_profiles = {profile: (adaptations[profile]['config']['base'] if profile in adaptations
                                           else read_json(features / profile / 'features.json'))
                                 for profile in selected['weights'] if profile != 'incumbent-global'}
    lock = {'task': task, 'manifest': plan['manifest'], 'search_sha256': checksum(search / 'search.json'),
            'criterion': 'equal_weight_macro_auroc_auprc' if task == 'chest' else 'auroc',
            'selected': selected, 'candidates': decisions, 'checkpoints': checkpoint_records,
            'feature_profiles': selected_feature_profiles, 'labels': list(labels), 'slope_intercept': params.tolist(),
            'thresholds': thresholds.tolist(), 'incumbent_logit_offset': offset.tolist(),
            'test_role': 'public benchmark reused from the previous experiment; excluded from current selection'}
    lock_path = output / 'selection.json'
    if lock_path.is_file() and read_json(lock_path) != lock:
        raise ValueError('Model choice was previously locked to another plan')
    write_json(lock_path, lock)

    test_rows = np.flatnonzero(frame.split.eq('test').to_numpy())
    test = frame.iloc[test_rows]
    previous_path = original / 'evaluation' / task / 'test-predictions.npz'
    with np.load(previous_path, allow_pickle=False) as previous:
        if not np.array_equal(previous['rows'], test_rows) or not np.array_equal(previous['labels'], truth[test_rows]):
            raise ValueError('The benchmark differs from the older experiment')
        old_probabilities, old_raw = previous['probabilities'], previous['raw_probabilities']
    previous_thresholds = read_json(previous_path.with_name('calibration.json'))['validation_f1_thresholds']
    logits = np.zeros_like(old_probabilities)
    for profile, weight in selected['weights'].items():
        if profile == 'incumbent-global':
            clip = old_raw.clip(1e-7, 1 - 1e-7)
            preds = np.log(clip) - np.log1p(-clip) - offset
        elif profile in adaptations:
            adapted_path = search / checkpoint_records[profile][0]['path']
            if checksum(adapted_path) != checkpoint_records[profile][0]['sha256']:
                raise ValueError('Fine-tuned checkpoint changed after validation lock')
            model, saved = load_adapted(adapted_path, weights)
            _, _, pixel_array, masks = pixel_store(pixels_root, cohort, task)
            preds = logits_for(model, pixel_array, masks, test_rows, saved['config']['batch']) - np.log(saved['weights'])
            del model
            torch.cuda.empty_cache()
        else:
            bank = DeviceBank(features / profile, plan['manifest'], task, test_rows)
            preds = _feature_logits(profile, bank, task, test_rows, checkpoint_records[profile], search)
            del bank
            torch.cuda.empty_cache()
        logits += preds * weight
    probabilities = expit(logits * params[:, 0] + params[:, 1])
    result = {
        'selection': selected, 'selection_sha256': checksum(lock_path),
        'full_candidate_count': len(decisions), 'initial_search_configurations': len([trial for trial in plan['trials'] if trial['rung'] == 1]),
        'test': all_metrics(truth[test_rows], probabilities, thresholds, labels),
        'previous': all_metrics(truth[test_rows], old_probabilities, np.asarray(previous_thresholds), labels),
        'patient_bootstrap': patient_bootstrap(truth[test_rows], probabilities,
                                              test.get('pixel_cluster', test.patient).to_numpy(), reference=old_probabilities),
        'subgroups': {name: all_metrics(truth[test_rows][mask], probabilities[mask], thresholds, labels)
                      for name, mask in subgroups(task, test).items() if mask.sum() >= 20},
    }
    np.savez_compressed(output / 'test-predictions.npz', rows=test_rows, logits=logits, probabilities=probabilities,
                        labels=truth[test_rows])
    write_json(output / 'metrics.json', result)
    print(f"{task} locked {selected['name']}, test AUROC {result['test']['macro_auroc']:.5f}, AP {result['test']['macro_auprc']:.5f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare final models and validate the frozen selection on the original benchmark')
    parser.add_argument('task', choices=('chest', 'mammo'))
    for name in ('cohort', 'features', 'search', 'original', 'output', 'pixels', 'weights'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.task, args.cohort, args.features, args.search, args.original, args.output, args.pixels, args.weights)


if __name__ == '__main__':
    main()
