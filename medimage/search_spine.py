from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from medimage.common import checksum, read_json, write_json
from medimage.radiography_train import save_checkpoint
from medimage.search_spine_features import DEEP_DIM, TEXTURE_DIM
from medimage.spine_localize import SpineCache, read_folds
from medimage.spine_metrics import (GROUPS, apply_temperature, bootstrap_delta, fit_temperature, read_submission,
                                    read_targets, score, submission)


@dataclass(frozen=True)
class SpineRecipe:
    kind: str
    descriptor: str
    width: int
    lr: float
    decay: float
    dropout: float
    ordinal: float
    severe: float
    class_power: float = 1.


RECIPES = (
    SpineRecipe('local', 'deep', 96, .001, .004, .18, 0., .6),
    SpineRecipe('local', 'texture', 96, .0015, .01, .25, .12, .7),
    SpineRecipe('local', 'hybrid', 128, .0006, .003, .3, .17, 1.),
    SpineRecipe('conv', 'deep', 128, .0011, .009, .3, .05, .55),
    SpineRecipe('conv', 'texture', 96, .0015, .015, .33, .15, .85),
    SpineRecipe('conv', 'hybrid', 128, .0007, .006, .33, .14, .65),
    SpineRecipe('conv', 'hybrid', 192, .00055, .02, .48, .25, 1.1, 1.15),
    SpineRecipe('attention', 'deep', 128, .00065, .005, .28, .12, .85),
    SpineRecipe('attention', 'texture', 96, .0009, .01, .35, .15, .7),
    SpineRecipe('attention', 'hybrid', 128, .0006, .008, .35, .2, .8),
    SpineRecipe('attention', 'hybrid', 192, .0004, .015, .5, .3, 1.2, .85),
    SpineRecipe('attention', 'hybrid', 96, .0011, .012, .4, .12, .4),
)
BUDGETS = (5, 12, 24)


class SpineFeatureHead(nn.Module):
    def __init__(self, recipe: SpineRecipe) -> None:
        super().__init__()
        self.recipe = recipe
        width = recipe.width
        dimension = {'deep': DEEP_DIM, 'texture': TEXTURE_DIM, 'hybrid': DEEP_DIM + TEXTURE_DIM}[recipe.descriptor]
        self.deep_norm = nn.LayerNorm(DEEP_DIM) if recipe.descriptor != 'texture' else nn.Identity()
        self.texture_norm = nn.LayerNorm(TEXTURE_DIM) if recipe.descriptor != 'deep' else nn.Identity()
        self.token = nn.Sequential(nn.Linear(dimension, width), nn.LayerNorm(width), nn.GELU())
        self.meta = nn.Linear(5, width)
        self.level = nn.Parameter(torch.randn(5, width) * .025)
        self.condition = nn.Parameter(torch.randn(6, width) * .025)
        if recipe.kind == 'attention':
            self.context = nn.MultiheadAttention(width, 4, dropout=.12, batch_first=True)
        if recipe.kind == 'conv':
            self.sequence = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.auxiliary_view = nn.Linear(width, width)
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.Dropout(recipe.dropout), nn.Linear(width, 3))

    def forward(self, inputs, metadata):
        deep, texture = inputs.float().split((DEEP_DIM, TEXTURE_DIM), dim=-1)
        if self.recipe.descriptor == 'hybrid':
            x = torch.cat([self.deep_norm(deep), self.texture_norm(texture)], dim=-1)
        elif self.recipe.descriptor == 'deep':
            x = self.deep_norm(deep)
        else:
            x = self.texture_norm(texture)
        width = self.recipe.width
        x = (self.token(x) * metadata[..., :1] + self.meta(metadata)).reshape(len(x), 6, 5, width)
        x = x + self.condition[:, None] + self.level[None, :]
        x = x + self.auxiliary_view(x[:, 5:6])
        if self.recipe.kind == 'conv':
            sequence = x.flatten(0, 1).transpose(1, 2)
            x = x + self.sequence(sequence).transpose(1, 2).reshape_as(x)
        x = x.flatten(1, 2)
        if self.recipe.kind == 'attention':
            context, _ = self.context(x, x, x, need_weights=False)
            x = x + context
        return self.readout(x[:, :25])


def loss_for(logits: torch.Tensor, targets: torch.Tensor, recipe: SpineRecipe) -> torch.Tensor:
    labels = targets.clamp_min(0)
    known = targets >= 0
    class_scale = logits.new_tensor([1., 2., 4.], dtype=torch.float32).pow(recipe.class_power)
    weights = class_scale[labels] * known
    p = logits.float().softmax(-1)
    nll = F.cross_entropy(logits.float().transpose(1, 2), labels, reduction='none')
    ordinal = (p.cumsum(-1)[..., :2] - F.one_hot(labels, 3).cumsum(-1)[..., :2]).square().mean(-1)
    weighted = weights * (nll + recipe.ordinal * ordinal)
    local = [weighted[:, group].sum() / weights[:, group].sum().clamp_min(1.) for group in GROUPS]
    any_prob = (p[:, :5, 2] * known[:, :5]).max(1).values.clamp(1e-6, 1 - 1e-6)
    is_severe = (targets[:, :5] == 2).any(1).float()
    severe_weights = weights[:, :5].max(1).values
    severe = F.binary_cross_entropy(any_prob, is_severe, weight=severe_weights, reduction='sum') / severe_weights.sum().clamp_min(1.)
    return (sum(local) + recipe.severe * severe) / (3 + recipe.severe)


@torch.inference_mode()
def probabilities(model, values, metadata, rows, batch=96) -> np.ndarray:
    model.eval()
    chunks = []
    for idx in torch.as_tensor(rows, device='cuda').split(batch):
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(values[idx], metadata[idx])
        chunks.append(logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(chunks)


def fit(recipe: SpineRecipe, values, metadata, labels, tr, va, directory, epochs, seed, signature):
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = SpineFeatureHead(recipe).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.lr, weight_decay=recipe.decay, fused=True)
    grad = torch.amp.GradScaler('cuda', init_scale=4096.)
    storage = directory / 'state.pt'
    config = {**signature, 'seed': seed, 'recipe': asdict(recipe)}
    records = []
    tune_epoch = signature['phase'] == 'inner'
    best = float('inf')
    best_predictions = None
    if storage.exists():
        saved = torch.load(storage, map_location='cpu', weights_only=True)
        if saved['config'] != config:
            raise ValueError('MRI run cannot resume from another configuration')
        records = saved['history']
        if len(records) > epochs:
            return min((entry for entry in records[:epochs] if 'score' in entry), key=lambda entry: entry['score'])
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        grad.load_state_dict(saved['scaler'])
        if saved['best_predictions'] is not None:
            best_predictions = saved['best_predictions'].numpy()
            best = min(entry['score'] for entry in records if 'score' in entry)
    y = torch.from_numpy(labels).cuda()
    selected = torch.as_tensor(tr, device='cuda')
    start = time.monotonic()
    for epoch in range(len(records), epochs):
        start_epoch = time.monotonic()
        torch.manual_seed(seed + epoch * 2311)
        model.train()
        for rows in selected[torch.randperm(len(selected), device='cuda')].split(48):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                output = model(values[rows], metadata[rows])
            loss = loss_for(output, y[rows], recipe)
            grad.scale(loss).backward()
            grad.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.)
            grad.step(optimizer)
            grad.update()
        records.append({'epochs': epoch + 1})
        if tune_epoch:
            predictions = probabilities(model, values, metadata, va)
            result = score(labels[va], predictions)
            records[-1].update(**result, seconds=round(time.monotonic() - start_epoch, 2))
            if result['score'] < best:
                best = result['score']
                best_predictions = predictions
    if not tune_epoch and 'score' not in records[-1]:
        predictions = probabilities(model, values, metadata, va)
        records[-1].update(**score(labels[va], predictions), seconds=round(time.monotonic() - start, 2))
        best_predictions = predictions
    if best_predictions is None:
        raise ValueError('MRI trial has no valid validation evidence')
    save_checkpoint(storage, {'config': config, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                              'scaler': grad.state_dict(), 'history': records,
                              'best_predictions': torch.from_numpy(best_predictions)})
    np.save(directory / 'validation.npy', best_predictions, allow_pickle=False)
    return min(records, key=lambda record: record['score']) if tune_epoch else records[-1]


def search(root: Path, cache_path: Path, folds_path: Path, features: Path, previous: Path, output: Path) -> None:
    torch.set_num_threads(4)
    cache = SpineCache.load(cache_path)
    assignments = read_folds(folds_path, cache.studies)
    truth = read_targets(root, cache.studies)
    audit_sha = checksum(folds_path)
    results = np.empty((len(truth), 25, 3), np.float64)
    calibrated = np.empty_like(results)
    audit = []
    for fold in range(5):
        directory = features / f'fold-{fold}'
        store = output / f'fold-{fold}'
        signature = read_json(directory / 'features.json')
        inner = (fold + 1) % 5
        if signature['manifest'] != audit_sha or signature['fold'] != fold or signature['inner_validation_fold'] != inner:
            raise ValueError('MRI feature fold provenance does not match the target split')
        with np.load(directory / 'geometry.npz', allow_pickle=False) as item:
            metadata = dict(item)
        inner_tr = np.flatnonzero(~np.isin(assignments, [fold, inner]))
        inner_va = np.flatnonzero(assignments == inner)
        outer_tr, outer_val = np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)
        expected = (~np.isin(np.arange(5), [fold, inner])[None]) & (np.arange(5)[None] != assignments[inner_tr, None])
        expected_outer = (np.arange(5)[None] != assignments[:, None]) & (np.arange(5)[None] != fold)
        if (not np.array_equal(metadata['studies'], cache.studies) or not np.array_equal(inner_tr, metadata['inner_rows'])
                or not np.array_equal(expected, metadata['inner_provenance'])
                or not np.array_equal(expected_outer, metadata['outer_provenance'])):
            raise ValueError('Prepared MRI records violate nested split provenance')
        original = np.load(directory / 'outer.npy', mmap_mode='r', allow_pickle=False)
        independent = np.load(directory / 'inner.npy', mmap_mode='r', allow_pickle=False)
        stage_data = torch.from_numpy(np.concatenate((independent, original[inner_va]), axis=0)).cuda()
        stage_meta = torch.from_numpy(np.concatenate((metadata['inner_meta'], metadata['outer_meta'][inner_va]), axis=0)).cuda()
        stage_labels = truth[np.r_[inner_tr, inner_va]]
        inner_fit = np.arange(len(inner_tr))
        inner_holdout = np.arange(len(inner_tr), len(stage_labels))
        active = list(range(len(RECIPES)))
        records = []
        for rung, epochs in enumerate(BUDGETS):
            rank = []
            for trial in active:
                result = fit(RECIPES[trial], stage_data, stage_meta, stage_labels, inner_fit, inner_holdout,
                             store / 'screen' / f'{trial:02d}', epochs, 1701 + fold, {**signature, 'phase': 'inner'})
                rank.append((result['score'], trial))
                records.append({'trial': trial, 'rung': rung + 1, 'budget': epochs, 'recipe': asdict(RECIPES[trial]), **result})
            rank.sort()
            active = [index for _, index in rank[:4 if rung == 0 else 2]]
            print(f'MRI fold {fold}, rung {rung + 1}: {len(rank)} trained, best {rank[0][1]} = {rank[0][0]:.5f}', flush=True)
        chosen = active[0]
        refit_epochs = next(record['epochs'] for record in reversed(records) if record['trial'] == chosen)
        selected_recipe = RECIPES[chosen]
        inner_predictions = np.load(store / 'screen' / f'{chosen:02d}' / 'validation.npy', allow_pickle=False)
        temperature = fit_temperature(truth[inner_va], inner_predictions)
        del stage_data, stage_meta
        outer_data = torch.from_numpy(np.array(original, copy=True)).cuda()
        outer_meta = torch.from_numpy(metadata['outer_meta']).cuda()
        checkpoints, outputs = [], []
        for seed in (2161 + fold, 3511 + fold):
            run = store / 'refit' / str(seed)
            fit(selected_recipe, outer_data, outer_meta, truth, outer_tr, outer_val, run, refit_epochs, seed,
                {**signature, 'phase': 'outer'})
            saved = torch.load(run / 'state.pt', map_location='cpu', weights_only=True)
            model_path = run / 'model.pt'
            save_checkpoint(model_path, {'model': saved['model'], 'config': saved['config']})
            checkpoints.append({'path': model_path.relative_to(output).as_posix(), 'sha256': checksum(model_path)})
            outputs.append(np.load(run / 'validation.npy', allow_pickle=False))
        results[outer_val] = np.mean(outputs, axis=0, dtype=np.float64)
        calibrated[outer_val] = apply_temperature(results[outer_val], temperature)
        audit.append({'fold': fold, 'inner_val': inner, 'trial': chosen, 'training_studies': len(outer_tr),
                      'validation_studies': len(outer_val), 'inner_training_count': len(inner_tr),
                      'inner_validation_count': len(inner_va), 'temperature': temperature.tolist(),
                      'recipe': asdict(selected_recipe), 'fit_epochs': refit_epochs, 'outer': score(truth[outer_val], results[outer_val]),
                      'outer_calibrated': score(truth[outer_val], calibrated[outer_val]),
                      'checkpoints': checkpoints, 'trials': records})
        print(f"MRI outer fold {fold} calibrated log loss {audit[-1]['outer_calibrated']['score']:.5f}", flush=True)
        del outer_data, outer_meta
        torch.cuda.empty_cache()
    previous_evaluation = previous / 'evaluation' / 'spine'
    previous_name = read_json(previous_evaluation / 'metrics.json')['selected_for_local_inference']
    prior = read_submission(previous_evaluation / f'{previous_name}-oof.csv', cache.studies)
    candidates = {'incumbent': (prior, 0., False), 'feature_uncalibrated': (results, 1., False),
                  'feature_calibrated': (calibrated, 1., True)}
    for fraction in (.15, .30, .50, .70):
        for name, values in (('raw', results), ('cal', calibrated)):
            candidates[f'incumbent+{name}-{fraction:.2f}'] = ((1 - fraction) * prior + fraction * values, fraction, name == 'cal')
    previous_folds = [score(truth[assignments == fold], prior[assignments == fold])['score'] for fold in range(5)]
    evaluation = []
    for name, (pred, fraction, calibration) in candidates.items():
        per_fold = [score(truth[assignments == fold], pred[assignments == fold])['score'] for fold in range(5)]
        evaluation.append({'name': name, 'fraction': fraction, 'calibrated': calibration,
                           'overall': score(truth, pred), 'per_fold': per_fold,
                           'better_folds': sum(candidate < reference for candidate, reference in zip(per_fold, previous_folds, strict=True)),
                           'delta_ci95': bootstrap_delta(truth, pred, prior)})
    evaluation.sort(key=lambda result: (result['overall']['score'], result['fraction']))
    selected = next(record for record in evaluation if record['name'] == 'incumbent' or
                    (record['better_folds'] >= 3 and record['delta_ci95'][1] < 0))
    final = candidates[selected['name']][0]
    for name, values in (('selected', final), ('feature_raw', results), ('feature_calibrated', calibrated)):
        submission(cache.studies, values).to_csv(output / f'{name}-oof.csv', index=False)
    write_json(output / 'selection.json', {'selected': selected, 'candidates': evaluation, 'folds': audit,
                                           'initial_trials': 5 * len(RECIPES), 'budgets': list(BUDGETS),
                                           'fold_sha256': audit_sha, 'pretrained_sha256': signature['pretrained_sha256'],
                                           'test_role': 'sample for schema and execution only; no public grades in local test'})
    print(f"MRI selected {selected['name']} with {selected['overall']['score']:.5f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Nested five-fold MRI model selection and comparison to previous incumbent')
    for name in ('root', 'cache', 'folds', 'features', 'previous', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    search(args.root, args.cache, args.folds, args.features, args.previous, args.output)


if __name__ == '__main__':
    main()
