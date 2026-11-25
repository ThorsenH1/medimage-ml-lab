from __future__ import annotations

import argparse
import os
from itertools import combinations
from pathlib import Path
from tempfile import mkstemp

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from medimage.common import checksum, load_manifest, read_json, write_json
from medimage.radiography_data import CHEST_LABELS
from medimage.radiography_evaluate import all_metrics, fit_calibration, patient_bootstrap, validation_metrics


def checked(root: Path, relative: str, digest: str | None = None) -> Path:
    base = root.resolve()
    location = (base / relative).resolve()
    if not location.is_relative_to(base) or (digest is not None and checksum(location) != digest):
        raise ValueError('An experiment artifact is outside its root or differs from its lock')
    return location


def read_scores(path: Path, rows, *, num_labels: int, truth=None, key='logits', dtype=np.float64):
    with np.load(path, allow_pickle=False) as data:
        values = data[key]
        if not np.array_equal(data['rows'], rows) or values.shape != (len(rows), num_labels):
            raise ValueError(f'Predictions are misaligned with the frozen cohort: {path.name}')
        if truth is not None and not np.array_equal(data['labels'], truth):
            raise ValueError('Previous test labels differ from the frozen cohort')
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite predictions')
        return values if dtype is None else values.astype(dtype, copy=False)


def previous_validation(args, plan, rows, count):
    sources = {'incumbent-global': args.previous / 'evaluation' / args.task / 'incumbent-validation.npz',
               'adapted-incumbent-224': args.previous / 'runs' / args.task / 'adaptation-reference' / 'validation.npz',
               'adapted-dino-224': args.previous / 'runs' / args.task / 'adaptation' / 'validation.npz'}
    return sum(weight * read_scores(sources[name], rows, num_labels=count)
               for name, weight in plan['selected']['weights'].items())


def rank_candidate(y, scores, weights, parameters, method):
    result = merge(scores, weights, parameters if method == 'calibrated-logit' else {})
    return {'weights': weights, 'method': method, 'metrics': validation_metrics(y, result)}, result


def weighted_components(scores, weights, parameters):
    for key, fraction in weights.items():
        scaling = np.asarray(parameters.get(key, [[1., 0.] for _ in range(scores[key].shape[1])]))
        yield key, np.asarray(fraction) * (scores[key] * scaling[:, 0] + scaling[:, 1])


def merge(scores, weights, parameters):
    return np.sum([value for _, value in weighted_components(scores, weights, parameters)], axis=0)


def cross_fit_calibration(truth, scores, folds):
    calibrated = {key: np.empty_like(value) for key, value in scores.items()}
    for fit, check in folds:
        for key, values in scores.items():
            parameters, _ = fit_calibration(truth[fit], values[fit])
            calibrated[key][check] = values[check] * parameters[:, 0] + parameters[:, 1]
    return calibrated


def label_weights(truth, scores, default, strength):
    quality = np.stack([roc_auc_score(truth, scores[key], average=None) for key in default])
    minority = np.minimum(truth.sum(axis=0), (1 - truth).sum(axis=0))
    reliability = minority / (minority + 60)
    adjusted = np.exp((quality - quality.max(axis=0)) * strength * reliability)
    adjusted = adjusted * np.asarray(list(default.values()))[:, None]
    adjusted /= adjusted.sum(axis=0)
    return {key: value.tolist() for key, value in zip(default, adjusted, strict=True)}


def label_adaptive_candidate(truth, scores, calibrated, weights, folds, strength):
    out = np.empty_like(next(iter(scores.values())))
    for fit, check in folds:
        fractions = label_weights(truth[fit], {key: scores[key][fit] for key in weights}, weights, strength)
        out[check] = merge({key: calibrated[key][check] for key in weights}, fractions, {})
    return {'weights': weights, 'method': 'calibrated-label-logit', 'weight_strength': strength,
            'label_weights': label_weights(truth, scores, weights, strength),
            'metrics': validation_metrics(truth, out)}


def fit_pairwise_weights(truth, matrix, strength, center=None, max_pairs=None, seed=8119):
    positive = truth[:, 0].astype(bool)
    cases, controls = matrix[positive], matrix[~positive]
    if max_pairs is not None and len(cases) * len(controls) > max_pairs:
        random = np.random.default_rng(seed)
        difference = (cases[random.integers(len(cases), size=max_pairs)]
                      - controls[random.integers(len(controls), size=max_pairs)])
    else:
        difference = (cases[:, None] - controls[None]).reshape(-1, matrix.shape[1])
    if not len(difference):
        raise ValueError('Pairwise fusion requires both classes')
    anchor = np.full(matrix.shape[1], 1 / matrix.shape[1]) if center is None else center

    def loss(weights):
        margin = difference @ weights
        change = weights - anchor
        value = np.logaddexp(0., -margin).mean() + strength * np.square(change).sum()
        gradient = -(difference.T @ expit(-margin)) / len(margin) + 2 * strength * change
        return value, gradient

    result = minimize(loss, anchor, method='SLSQP', jac=True, bounds=[(0., 1.)] * len(anchor),
                      constraints={'type': 'eq', 'fun': lambda value: value.sum() - 1},
                      options={'ftol': 1e-9, 'maxiter': 100})
    if not result.success:
        raise ValueError(f'Pairwise fusion optimization failed: {result.message}')
    weights = np.maximum(result.x, 0)
    weights[weights < 1e-5] = 0
    return weights / weights.sum()


def pairwise_candidate(truth, scores, folds, strength):
    def fit_blend(fit):
        ranking = sorted(scores, key=lambda name: roc_auc_score(truth[fit, 0], scores[name][fit, 0]), reverse=True)
        chosen = ranking[:5]
        calibration = [fit_calibration(truth[fit], scores[key][fit])[0][0] for key in chosen]
        calibrated = np.column_stack([scores[key][:, 0] * slope + intercept
                                      for key, (slope, intercept) in zip(chosen, calibration, strict=True)])
        fractions = fit_pairwise_weights(truth[fit], calibrated[fit], strength)
        return chosen, calibrated, fractions

    out = np.empty_like(truth, dtype=np.float64)
    for fit, check in folds:
        _, calibrated, fractions = fit_blend(fit)
        out[check, 0] = calibrated[check] @ fractions
    chosen, _, fractions = fit_blend(np.arange(len(truth)))
    selected = {key: float(fraction) for key, fraction in zip(chosen, fractions, strict=True) if fraction > 0}
    return {'weights': selected, 'method': 'calibrated-pairwise-logit', 'weight_strength': strength,
            'metrics': validation_metrics(truth, out)}


def stable_candidate(truth, scores, groups, families):
    if len(set(families.values())) < 2:
        raise ValueError('Stable blending requires at least two model families')
    predicted = np.zeros_like(truth, dtype=np.float64)
    collected = {key: 0. for key in scores}
    seeds = (421, 510, 719)
    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fit, check in splitter.split(truth, truth[:, 0], groups):
            ranked = sorted(scores, key=lambda key: roc_auc_score(truth[fit, 0], scores[key][fit, 0]),
                            reverse=True)
            unique = {}
            for key in ranked:
                unique.setdefault(families[key], key)
            names = list(unique.values())[:6]
            projections = {}
            for key in names:
                calibration, _ = fit_calibration(truth[fit], scores[key][fit])
                slope, intercept = calibration[0]
                projections[key] = scores[key] * slope + intercept
            fits = []
            for size in (2, 3):
                for subset in combinations(names, size):
                    logits = np.mean([projections[key][fit] for key in subset], axis=0)
                    fits.append((roc_auc_score(truth[fit, 0], logits[:, 0]), subset))
            fits.sort(key=lambda item: item[0], reverse=True)
            retained = fits[:4]
            values = np.asarray([score for score, _ in retained])
            strength = np.exp((values - values[0]) / .01)
            strength /= strength.sum()
            for fraction, (_, subset) in zip(strength, retained, strict=True):
                contribution = float(fraction) / (len(seeds) * len(subset))
                for key in subset:
                    predicted[check] += contribution * projections[key][check]
                    collected[key] += contribution / splitter.n_splits
    return {'weights': {key: fraction for key, fraction in collected.items() if fraction > 0},
            'method': 'calibrated-validation-bag', 'selection_folds': len(seeds) * 5,
            'metrics': validation_metrics(truth, predicted)}


def validation_lock(args):
    frame, audit = load_manifest(args.cohort)
    rows = np.flatnonzero(frame.split.eq('val').to_numpy())
    labels = CHEST_LABELS if args.task == 'chest' else ('malignancy_in_lesion_enriched_cohort',)
    truth = frame.loc[:, list(CHEST_LABELS) if args.task == 'chest' else ['positive']].to_numpy(np.int8)[rows]
    old_path = args.previous / 'evaluation' / args.task / 'selection.json'
    old_plan = read_json(old_path)
    if old_plan['manifest'] != audit['manifest_sha256']:
        raise ValueError('The previous lock covers a different cohort')
    metadata = {'previous-locked': {'prior_selection_sha256': checksum(old_path)}}
    scores = {'previous-locked': previous_validation(args, old_plan, rows, len(labels))}
    for spec in args.candidate:
        name, separator, relative = spec.partition('=')
        run, at, pixel_key = relative.partition('@')
        if not separator or not at or name in metadata or not name or not pixel_key:
            raise ValueError('Use a unique name=run-directory@pixel-cache candidate')
        path = checked(args.work, run)
        record = read_json(path / 'result.json')
        config = record['config']
        if config['manifest'] != audit['manifest_sha256'] or config['task'] != args.task:
            raise ValueError('Candidate training cohort differs from validation cohort')
        model_path, predictions = path / 'best.pt', path / 'validation.npz'
        if not model_path.is_file() or checksum(predictions) != record['validation_sha256']:
            raise ValueError('Candidate model or validation score changed after training')
        cache = pixels_location(args, pixel_key) / ('pixels.json' if args.task == 'chest' else 'audit.json')
        if checksum(cache) != config['pixel_sha256']:
            raise ValueError('Candidate input specification differs from training')
        signature = read_json(cache)
        image_shape = signature['shape'][1:] if args.task == 'chest' else signature['config']['shape'][2:]
        metadata[name] = {'run': run, 'pixels': pixel_key, 'checkpoint_sha256': record['best_sha256'],
                          'validation_sha256': record['validation_sha256'], 'config': config,
                          'image_shape': image_shape}
        scores[name] = read_scores(predictions, rows, num_labels=len(labels))
    parameters = {key: fit_calibration(truth, values)[0].tolist() for key, values in scores.items()}
    identities = frame.iloc[rows]['pixel_cluster' if args.task == 'chest' else 'patient'].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(truth, groups=identities))
    calibrated_scores = cross_fit_calibration(truth, scores, folds)
    recipes = [{key: 1.} for key in scores]
    for left, right in combinations(scores, 2):
        recipes.extend({left: fraction, right: 1 - fraction} for fraction in (.25, .5, .75))
    recipes.extend({key: 1 / 3 for key in names} for names in combinations(scores, 3))
    candidates = []
    for weights in recipes:
        methods = ('raw-logit', 'calibrated-logit') if len(weights) > 1 else ('raw-logit',)
        for method in methods:
            values = calibrated_scores if method == 'calibrated-logit' else scores
            candidates.append(rank_candidate(truth, values, weights, {}, method)[0])
        if args.task == 'chest' and 'previous-locked' in weights and len(weights) > 1:
            candidates.extend(label_adaptive_candidate(truth, scores, calibrated_scores, weights, folds, strength)
                              for strength in (8, 24))
    if args.pairwise:
        if args.task != 'mammo':
            raise ValueError('Pairwise ranking blend is implemented for the breast-level endpoint')
        candidates.extend(pairwise_candidate(truth, scores, folds, strength) for strength in (.03, .12, .4))
    if args.stable_blend:
        if args.task != 'mammo':
            raise ValueError('Validation bagging is implemented for the breast-level endpoint')
        families = {key: entry.get('config', {}).get('original_sha256', entry.get('checkpoint_sha256', key))
                    for key, entry in metadata.items()}
        candidates.append(stable_candidate(truth, scores, identities, families))
    candidates.sort(key=lambda item: (item['metrics']['macro_auroc'], item['metrics']['macro_auprc']),
                    reverse=True)
    eligible = ([item for item in candidates
                 if item['method'] in ('calibrated-pairwise-logit', 'calibrated-validation-bag')]
                if args.crossfit_only else candidates)
    if not eligible:
        raise ValueError('Cross-fit-only selection requires a nested blend candidate')
    selected = eligible[0]
    selected_logits = selected_scores(lock_selected=selected, scores=scores, calibration=parameters)
    calibration, thresholds = fit_calibration(truth, selected_logits)
    criterion = ('nested group validation macro AUROC' if args.crossfit_only else 'validation macro AUROC')
    lock = {'task': args.task, 'manifest': audit['manifest_sha256'], 'criterion': f'{criterion}, AP tie-breaker',
            'blend_validation_calibration': 'Five group-disjoint folds inside validation',
            'validation_rows': len(rows), 'labels': list(labels), 'candidates': metadata,
            'validation_ranking': candidates, 'selected': selected, 'component_calibration': parameters,
            'ensemble_calibration': calibration.tolist(), 'thresholds': thresholds.tolist(),
            'test_role': 'Previously reported official benchmark. Current candidates and calibration locked before scoring.'}
    path = args.output / 'selection.json'
    if path.exists() and read_json(path) != lock:
        raise ValueError('Use a separate output directory for a new benchmark decision')
    write_json(path, lock)
    print({'selected': selected, 'evaluated_validation_recipes': len(candidates)}, flush=True)


def selected_scores(lock_selected, scores, calibration):
    parameters = calibration if lock_selected['method'].startswith('calibrated-') else {}
    return merge(scores, lock_selected.get('label_weights', lock_selected['weights']), parameters)


def pixels_location(args, name):
    if args.task == 'chest' and name == 'previous-chest224':
        return args.previous / 'pixels' / 'chest'
    return checked(args.work / 'pixels', name)


def load_component(work: Path, candidate, weights: Path, clip: Path | None):
    import torch

    from medimage.auc_patches import PatchNet
    from medimage.auc_train import new_model

    path = checked(work, candidate['run'])
    checkpoint = checked(path, 'best.pt', candidate['checkpoint_sha256'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = saved['config']
    if config != candidate['config']:
        raise ValueError('Checkpoint recipe differs from the frozen selection')
    if config['backbone'] == 'chest-fusion':
        from medimage.auc_fusion import FusionHead, FusionRecipe

        model = FusionHead(config['dimensions'], FusionRecipe(**config['recipe']))
    elif config['backbone'] == 'patch-mil':
        from medimage.auc_mammo_mil import BreastMIL

        model = BreastMIL(**saved['head'])
    elif config['backbone'] == 'mammo-texture':
        from medimage.auc_mammo_texture import TextureClassifier

        model = TextureClassifier(**saved['head'])
    elif config['backbone'] == 'patch-r18':
        model = PatchNet(weights)
    else:
        model = new_model(config, clip if config['backbone'] == 'mammo-clip' else weights)
    if config['task'] == 'mammo' and hasattr(model, 'weight_sha256') and model.weight_sha256 != config['weight_sha256']:
        raise ValueError('Mammography foundation differs from the selected recipe')
    model.load_state_dict(saved['model'])
    model.cuda().to(memory_format=torch.channels_last).eval()
    return model, saved


def component_logits(model, saved, arrays, rows, weights=None):
    from medimage.auc_patches import scan
    from medimage.auc_train import predict

    config = saved['config']
    if config['backbone'] == 'patch-mil':
        from medimage.auc_mammo_mil import predict as predict_mil

        logits = predict_mil(model, saved, arrays, rows, weights)
    elif config['backbone'] == 'mammo-texture':
        from medimage.auc_mammo_texture import breast_features, predict as predict_texture

        logits = predict_texture(model, breast_features(arrays, rows))
    elif config['backbone'] == 'patch-r18':
        logits = scan(model, arrays, rows)[saved['pooling']]
    else:
        logits = predict(model, config['task'], arrays, rows, config['batch'], config['tta'],
                         np.asarray(saved['prior_correction'], dtype=np.float32))
    return logits.astype(np.float64)


def test_component(args, candidate, rows):
    config = candidate['config']
    if config['backbone'] == 'chest-fusion':
        from medimage.auc_fusion import bank_for, cached_prediction

        model, saved = load_component(args.work, candidate, args.weights, args.clip)
        banks = bank_for(args.previous / 'features' / 'chest', config['profiles'], config['manifest'], rows)
        return cached_prediction(model, banks, rows, np.asarray(saved['prior_correction'], dtype=np.float32))

    from medimage.auc_train import store

    _, _, arrays, signature = store(args.task, args.cohort, pixels_location(args, candidate['pixels']))
    if signature != candidate['config']['pixel_sha256']:
        raise ValueError('Test pixel representation differs from training')
    model, saved = load_component(args.work, candidate, args.weights, args.clip)
    return component_logits(model, saved, arrays, rows, args.weights)


def prediction_examples(task, frame, rows, truth, probabilities, thresholds, labels):
    rng = np.random.default_rng(80015)
    order = rng.permutation(len(rows))
    examples = []
    for matching_truth, matching_call in ((1, 1), (0, 0), (0, 1), (1, 0)):
        target = (truth[order] == matching_truth) & ((probabilities[order] >= thresholds) == bool(matching_call))
        if not target.any():
            continue
        row, column = np.argwhere(target)[0]
        actual = order[row]
        sample = frame.iloc[rows[actual]]
        projection = sample['view'] if task == 'chest' else '+'.join(view for view in ('CC', 'MLO') if sample[view])
        examples.append({'example': len(examples) + 1, 'target': labels[column], 'projection': projection,
                         'reference': matching_truth, 'decision': matching_call,
                         'probability': float(probabilities[actual, column]), 'threshold': float(thresholds[column])})
    return examples


def save_scores(destination: Path, **arrays) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = mkstemp(dir=destination.parent, prefix='.scoring-', suffix='.npz')
    staging = Path(filename)
    try:
        with os.fdopen(descriptor, 'wb') as output:
            np.savez_compressed(output, **arrays)
        staging.replace(destination)
    finally:
        staging.unlink(missing_ok=True)


def locked_component_scores(args, candidate, path, key, digest, rows, truth):
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if existing['selection_sha256'].item() != digest or existing['component'].item() != key:
                raise ValueError('Cached component scores belong to another locked ensemble')
        values = read_scores(path, rows, num_labels=truth.shape[1], truth=truth, dtype=None)
        print(f'Reusing completed component {key} on {len(rows)} official test examples', flush=True)
        return values
    print(f'Scoring locked component {key} on {len(rows)} official test examples', flush=True)
    values = test_component(args, candidate, rows)
    if values.shape != truth.shape or not np.isfinite(values).all():
        raise ValueError('Scored component does not match the locked test labels')
    save_scores(path, rows=rows, labels=truth, logits=values, selection_sha256=digest, component=key)
    return values


def heldout_probabilities(args, lock, digest, rows, labels, truth, reference):
    import torch

    destination = args.output / 'test-predictions.npz'
    if destination.exists():
        with np.load(destination, allow_pickle=False) as existing:
            if 'selection_sha256' in existing:
                if existing['selection_sha256'].item() != digest:
                    raise ValueError('Cached test scores belong to a different locked ensemble')
                return read_scores(destination, rows, num_labels=len(labels), truth=truth, key='probabilities')
    selected = lock['selected']
    scores = {}
    for index, key in enumerate(selected['weights']):
        if key == 'previous-locked':
            scores[key] = read_scores(reference / 'test-predictions.npz', rows, num_labels=len(labels), truth=truth)
        else:
            checkpoint = args.output / 'component-predictions' / f'{index:03d}.npz'
            scores[key] = locked_component_scores(args, lock['candidates'][key], checkpoint,
                                                  key, digest, rows, truth)
        torch.cuda.empty_cache()
    logits = selected_scores(selected, scores, lock['component_calibration'])
    parameters = np.asarray(lock['ensemble_calibration'])
    probabilities = expit(logits * parameters[:, 0] + parameters[:, 1])
    save_scores(destination, rows=rows, labels=truth, probabilities=probabilities,
                logits=logits, selection_sha256=digest)
    return probabilities


def evaluate_lock(args):
    import torch

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    frame, audit = load_manifest(args.cohort)
    path = args.output / 'selection.json'
    lock = read_json(path)
    if lock['task'] != args.task or lock['manifest'] != audit['manifest_sha256']:
        raise ValueError('Locked decision covers another evaluation cohort')
    rows = np.flatnonzero(frame.split.eq('test').to_numpy())
    labels = tuple(lock['labels'])
    truth = frame.loc[:, list(CHEST_LABELS) if args.task == 'chest' else ['positive']].to_numpy(np.int8)[rows]
    reference = args.previous / 'evaluation' / args.task
    if checksum(reference / 'selection.json') != lock['candidates']['previous-locked']['prior_selection_sha256']:
        raise ValueError('Previous benchmark lock changed')
    before = read_scores(reference / 'test-predictions.npz', rows, num_labels=len(labels), truth=truth,
                         key='probabilities')
    digest = checksum(path)
    probabilities = heldout_probabilities(args, lock, digest, rows, labels, truth, reference)
    thresholds = np.asarray(lock['thresholds'])
    benchmark = all_metrics(truth, probabilities, thresholds, labels)
    reference_metrics = all_metrics(truth, before, np.asarray(read_json(reference / 'selection.json')['thresholds']), labels)
    bootstrap = patient_bootstrap(truth, probabilities, frame.iloc[rows].patient.to_numpy(),
                                  repeats=args.bootstrap, reference=before)
    report = {'task': args.task, 'selection_sha256': digest, 'selected': lock['selected'], 'test': benchmark,
              'reference_test': reference_metrics, 'test_uncertainty': bootstrap,
              'auroc_delta': benchmark['macro_auroc'] - reference_metrics['macro_auroc'],
              'examples': prediction_examples(args.task, frame, rows, truth, probabilities, thresholds, labels)}
    if args.task == 'mammo':
        strata = frame.iloc[rows].type.to_numpy()
    else:
        strata = frame.iloc[rows].view.to_numpy()
    report['test_subsets'] = {str(key): all_metrics(truth[strata == key], probabilities[strata == key],
                                                  thresholds, labels) for key in np.unique(strata)}
    write_json(args.output / 'metrics.json', report)
    print({key: benchmark[key] for key in ('rows', 'macro_auroc', 'macro_auprc')}, flush=True)


def main():
    parser = argparse.ArgumentParser(description='Lock validation winners separately from official benchmark evaluation')
    parser.add_argument('phase', choices=('lock', 'test'))
    parser.add_argument('task', choices=('chest', 'mammo'))
    for name in ('work', 'previous', 'cohort', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--candidate', action='append', default=[])
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--clip', type=Path)
    parser.add_argument('--pairwise', action='store_true', help='Compare convex pairwise-ranking blends on validation')
    parser.add_argument('--stable-blend', action='store_true', help='Compare group-fold aggregation of small blends')
    parser.add_argument('--crossfit-only', action='store_true', help='Select among nested group cross-fit blend candidates')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.phase == 'lock':
        with threadpool_limits(limits=2):
            validation_lock(args)
    else:
        evaluate_lock(args)


if __name__ == '__main__':
    main()
