from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from medimage.common import checksum, read_json, write_json
from medimage.radiography_data import CHEST_LABELS


def fit_calibration(truth: np.ndarray, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parameters = []
    thresholds = []
    for y, z in zip(truth.T, logits.T, strict=True):
        if np.unique(y).size != 2:
            raise ValueError('No calibration possible for a one-class validation target')
        model = LogisticRegression(C=.7).fit(z[:, None], y)
        slope, intercept = float(model.coef_[0, 0]), float(model.intercept_[0])
        if slope <= 0:
            slope, intercept = .001, float(np.log((y.sum() + .5) / (len(y) - y.sum() + .5)))
        p = expit(slope * z + intercept)
        precision, recall, candidates = precision_recall_curve(y, p)
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-8)
        thresholds.append(float(candidates[np.argmax(f1)]))
        parameters.append((slope, intercept))
    return np.asarray(parameters), np.asarray(thresholds)


def expected_calibration_error(truth: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    bin_index = np.minimum((proba * bins).astype(int), bins - 1)
    residual = np.bincount(bin_index, weights=truth - proba, minlength=bins)
    return float(np.abs(residual).sum() / len(truth))


def all_metrics(truth: np.ndarray, proba: np.ndarray, thresholds: np.ndarray, labels: tuple[str, ...]) -> dict:
    per_label = {}
    for index, name in enumerate(labels):
        y, p = truth[:, index].astype(bool), proba[:, index].astype(np.float64)
        called = p >= thresholds[index]
        positive, negative = y.sum(), (~y).sum()
        evaluable = positive > 0 and negative > 0
        clipped = p.clip(1e-7, 1 - 1e-7)
        per_label[name] = {
            'positives': int(positive), 'negatives': int(negative), 'prevalence': float(y.mean()),
            'auroc': float(roc_auc_score(y, p)) if evaluable else None,
            'auprc': float(average_precision_score(y, p)) if evaluable else None,
            'brier': float(np.mean((p - y) ** 2)),
            'binary_log_loss': float(np.where(y, -np.log(clipped), -np.log1p(-clipped)).mean()),
            'ece_10': expected_calibration_error(y, p),
            'sensitivity': float((y & called).sum() / positive) if positive else None,
            'specificity': float((~y & ~called).sum() / negative) if negative else None,
        }
    aggregate = {}
    for metric in ('auroc', 'auprc', 'brier', 'binary_log_loss', 'ece_10'):
        observed = [item[metric] for item in per_label.values() if item[metric] is not None]
        aggregate[f'macro_{metric}'] = float(np.mean(observed)) if observed else None
    return {'rows': len(truth), **aggregate, 'labels': per_label}


def patient_bootstrap(truth: np.ndarray, proba: np.ndarray, groups: np.ndarray, repeats: int = 400) -> dict:
    groups, mapping = np.unique(groups, return_inverse=True)
    sorted_labels = []
    for y, p in zip(truth.T, proba.T, strict=True):
        order = np.argsort(-p, kind='stable')
        p, y = p[order], y[order]
        cut = np.r_[np.flatnonzero(p[1:] != p[:-1]), len(p) - 1]
        sorted_labels.append((mapping[order], y, cut))
    rng = np.random.default_rng(1801)
    samples = []
    for _ in range(repeats):
        occurrences = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
        estimate = []
        for inverse, y, cut in sorted_labels:
            row_weight = occurrences[inverse]
            tp = np.cumsum(row_weight * y)[cut]
            fp = np.cumsum(row_weight)[cut] - tp
            if tp[-1] == 0 or fp[-1] == 0:
                break
            recall, fpr = np.r_[0, tp / tp[-1]], np.r_[0, fp / fp[-1]]
            precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
            estimate.append([np.trapezoid(recall, fpr), np.sum(np.diff(recall) * precision)])
        if len(estimate) == len(sorted_labels):
            samples.append(np.mean(estimate, axis=0))
    if not samples:
        raise ValueError('No evaluable patient bootstrap replicates')
    intervals = np.percentile(samples, [2.5, 97.5], axis=0).T
    return {'valid_replicates': len(samples), 'patients': len(groups),
            'macro_auroc_ci95': intervals[0].tolist(), 'macro_auprc_ci95': intervals[1].tolist()}


def subgroups(task: str, frame) -> dict[str, np.ndarray]:
    if task == 'chest':
        age = frame.age.to_numpy()
        return {'AP': frame.view.eq('AP').to_numpy(), 'PA': frame.view.eq('PA').to_numpy(),
                'female': frame.sex.eq('F').to_numpy(), 'male': frame.sex.eq('M').to_numpy(),
                '18_to_49': (age >= 18) & (age < 50), '50_to_69': (age >= 50) & (age < 70),
                '70_to_100': (age >= 70) & (age <= 100)}
    both = (frame.CC.ne('') & frame.MLO.ne('')).to_numpy()
    return {'both_views': both, 'single_view': ~both, 'mass_only': frame.type.eq('mass').to_numpy(),
            'calcification_only': frame.type.eq('calc').to_numpy(), 'mixed_lesions': frame.type.eq('calc+mass').to_numpy()}


def verify_development_cache(frame, cache: Path, original_frame, original_cache: Path) -> None:
    current_rows = np.flatnonzero(frame.split.ne('test').to_numpy())
    original_rows = np.flatnonzero(original_frame.split.ne('test').to_numpy())
    if not frame.iloc[current_rows].reset_index(drop=True).equals(original_frame.iloc[original_rows].reset_index(drop=True)):
        raise ValueError('Recovered test cache changed train or validation metadata')
    current = np.load(cache / 'images.npy', mmap_mode='r', allow_pickle=False)
    original = np.load(original_cache / 'images.npy', mmap_mode='r', allow_pickle=False)
    for offset in range(0, len(current_rows), 16):
        section = slice(offset, offset + 16)
        if not np.array_equal(current[current_rows[section]], original[original_rows[section]]):
            raise ValueError('Recovered test cache changed train or validation pixels')


def evaluate(args: argparse.Namespace) -> None:
    import torch

    from medimage.radiography_models import build_model
    from medimage.radiography_train import load_frame, make_loader, predict_batches

    frame, audit = load_frame(args.cache)
    development_manifest = audit['manifest_sha256']
    equivalent_development = False
    if args.development_cache is not None:
        original, original_audit = load_frame(args.development_cache)
        verify_development_cache(frame, args.cache, original, args.development_cache)
        development_manifest = original_audit['manifest_sha256']
        equivalent_development = True
    labels = CHEST_LABELS if args.task == 'chest' else ('malignancy_in_lesion_enriched_cohort',)
    selected = []
    for path in args.runs:
        result = read_json(path / 'result.json')
        config = result['config']
        if config['limit'] is not None or config['task'] != args.task or config['manifest'] != development_manifest:
            raise ValueError('Run uses incompatible task, smoke subset or cohort')
        validation = result['best_validation']
        measure = validation['macro_auroc']
        if args.task == 'chest':
            measure = .5 * (measure + validation['macro_auprc'])
        selected.append((measure, path, result))
    selected.sort(key=lambda item: (-item[0], item[1].name))
    plan = {'selection_criterion': ('validation_macro_auroc' if args.task == 'mammo'
                                   else 'average_of_validation_macro_auroc_and_auprc'),
            'variants': [{'run': path.name, 'measure': measure, 'best_validation': result['best_validation'],
                          'model_parameters': result['model_parameters']}
                         for measure, path, result in selected],
            'winner': selected[0][1].name, 'checkpoint_sha256': checksum(selected[0][1] / 'best.pt'),
            'manifest_sha256': audit['manifest_sha256'], 'development_manifest_sha256': development_manifest,
            'separate_cache_development_pixels_and_rows_verified': equivalent_development}
    lock_path = args.output / 'evaluation-plan.json'
    if lock_path.exists() and read_json(lock_path) != plan:
        raise ValueError('Evaluation directory is locked to a different candidate selection')
    write_json(lock_path, plan)
    path, result = selected[0][1:]
    checkpoint = torch.load(path / 'best.pt', map_location='cpu', weights_only=True)
    if checkpoint['config'] != result['config'] or checkpoint['validation'] != result['best_validation']:
        raise ValueError('Checkpoint and selected experiment metadata disagree')
    model = build_model(args.task, checkpoint['config']['variant'], pretrained=False).cuda()
    model.load_state_dict(checkpoint['model'])
    val = make_loader(args.task, args.cache, frame, 'val', args.batch_size, 0)
    val_logits, val_truth = predict_batches(model, val)
    parameters, thresholds = fit_calibration(val_truth, val_logits)
    calibration = {'labels': list(labels), 'slope_intercept': parameters.tolist(),
                   'validation_f1_thresholds': thresholds.tolist(), 'checkpoint_sha256': plan['checkpoint_sha256']}
    write_json(args.output / 'calibration.json', calibration)
    test = make_loader(args.task, args.cache, frame, 'test', args.batch_size, 0)
    test_frame = frame.iloc[test.dataset.indices]
    logits, truth = predict_batches(model, test)
    raw = expit(logits.astype(np.float64))
    probabilities = expit(logits * parameters[:, 0] + parameters[:, 1])
    metrics = {
        'task': args.task, 'selection': plan, 'calibration': calibration, 'dataset_audit': audit,
        'test_raw': all_metrics(truth, raw, np.full(len(labels), .5), labels),
        'test_calibrated': all_metrics(truth, probabilities, thresholds, labels),
        'patient_bootstrap': patient_bootstrap(truth, probabilities,
                                              test_frame.get('pixel_cluster', test_frame.patient).to_numpy(),
                                              args.bootstrap),
        'test_subgroups': {key: all_metrics(truth[mask], probabilities[mask], thresholds, labels)
                           for key, mask in subgroups(args.task, test_frame).items() if mask.sum() >= 20},
    }
    np.savez_compressed(args.output / 'test-predictions.npz', rows=test.dataset.indices,
                        probabilities=probabilities, raw_probabilities=raw, labels=truth)
    write_json(args.output / 'metrics.json', metrics)
    values = metrics['test_calibrated']
    print(f"{args.task}: {path.name}, test AUROC {values['macro_auroc']:.4f}, AUPRC {values['macro_auprc']:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Lock a validation winner, fit calibration, and evaluate the test cohort')
    parser.add_argument('task', choices=['chest', 'mammo'])
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--development-cache', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--bootstrap', type=int, default=400)
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
