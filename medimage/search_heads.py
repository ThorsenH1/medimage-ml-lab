from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from medimage.common import assert_disjoint, checksum, read_json, write_json
from medimage.radiography_data import CHEST_LABELS
from medimage.radiography_train import save_checkpoint, validation_metrics
from medimage.search_features import load_manifest

PROFILES = ('effb0-160', 'r50-224', 'effb0-224', 'dino-224')
DEFAULT_SEED = 3803


@dataclass(frozen=True)
class Recipe:
    head: str
    width: int
    learning_rate: float
    decay: float
    dropout: float
    class_power: float
    tile_dropout: float = 0.


RECIPES = (
    Recipe('linear', 0, .0012, .001, .0, 0.),
    Recipe('linear', 0, .003, .015, .12, .5),
    Recipe('mlp', 96, .0005, .003, .18, .0),
    Recipe('mlp', 160, .0009, .002, .25, .5),
    Recipe('mlp', 256, .0015, .008, .35, .5, .08),
    Recipe('mlp', 128, .002, .015, .50, .0, .12),
    Recipe('mlp', 256, .00035, .0008, .33, .25),
    Recipe('attention', 96, .00065, .002, .18, .0),
    Recipe('attention', 128, .001, .007, .3, .5, .12),
    Recipe('attention', 160, .00048, .003, .35, .25),
    Recipe('attention', 224, .0014, .012, .45, .5, .16),
    Recipe('attention', 128, .00065, .02, .5, .75, .22),
)


class FeatureClassifier(nn.Module):
    def __init__(self, dim: int, positions: int, outputs: int, recipe: Recipe) -> None:
        super().__init__()
        self.recipe = recipe
        self.normalize = nn.LayerNorm(dim)
        self.register_buffer('protected', torch.asarray([position % 7 == 0 for position in range(positions)]))
        if recipe.head == 'linear':
            self.readout = nn.Sequential(nn.Dropout(recipe.dropout), nn.Linear(dim, outputs))
        else:
            width = recipe.width
            self.project = nn.Sequential(nn.Linear(dim, width), nn.LayerNorm(width), nn.GELU())
            self.position = nn.Parameter(torch.randn(positions, width) * .02)
            self.readout = nn.Sequential(nn.LayerNorm(width * 2), nn.Dropout(recipe.dropout),
                                         nn.Linear(width * 2, outputs))
            if recipe.head == 'attention':
                self.gate = nn.Linear(width, outputs)
                self.gate_context = nn.Linear(width, width)
                self.regional_readout = nn.Parameter(torch.randn(outputs, width) * .02)

    def forward(self, features, available):
        if self.training and self.recipe.tile_dropout:
            available = available & ((torch.rand(available.shape, device=available.device) >= self.recipe.tile_dropout)
                                     | self.protected)
        weights = available.unsqueeze(-1)
        features = self.normalize(features.float())
        if self.recipe.head == 'linear':
            return self.readout((features * weights).sum(1) / weights.sum(1).clamp_min(1))
        spatial = self.project(features) + self.position
        pooled = (spatial * weights).sum(1) / weights.sum(1).clamp_min(1)
        global_weights = weights & self.protected[None, :, None]
        global_mean = (spatial * global_weights).sum(1) / global_weights.sum(1).clamp_min(1)
        logits = self.readout(torch.cat([pooled, global_mean], dim=1))
        if self.recipe.head == 'attention':
            attention = self.gate(spatial * self.gate_context(spatial).sigmoid()).float()
            attention = attention.masked_fill(~weights, -1e4).softmax(1)
            evidence = torch.einsum('bpc, bpd -> bcd', attention, spatial.float())
            logits = logits + (evidence * self.regional_readout).sum(-1)
        return logits


class DeviceBank:
    def __init__(self, directory: Path, manifest: str, task: str, restrict: np.ndarray | None = None) -> None:
        self.info = read_json(directory / 'features.json')
        if self.info['manifest'] != manifest or self.info['task'] != task:
            raise ValueError('Features and cohort do not agree')
        array = np.load(directory / 'features.npy', mmap_mode='r', allow_pickle=False)
        if list(array.shape) != self.info['shape']:
            raise ValueError('Feature shape differs from its manifest')
        masks = np.load(directory / 'mask.npy', allow_pickle=False)
        self.data = torch.empty(array.shape, dtype=torch.float16, device='cuda')
        usable = np.arange(len(array)) if restrict is None else restrict
        for offset in range(0, len(usable), 1024):
            indices = usable[offset:offset + 1024]
            self.data[torch.as_tensor(indices, device='cuda')] = torch.from_numpy(array[indices]).cuda()
        self.masks = torch.from_numpy(masks).cuda()

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.data.shape[1:])


def inner_split(frame: pd.DataFrame, task: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    eligible = frame.loc[frame.split.eq('train')]
    grouping = 'pixel_cluster' if task == 'chest' else 'patient'
    if task == 'chest':
        clusters = eligible.groupby(grouping, sort=True)[list(CHEST_LABELS)].max()
        split = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.15, random_state=seed)
        _, val = next(split.split(np.zeros(len(clusters)), clusters))
        validation = eligible[grouping].isin(clusters.index[val]).to_numpy()
    else:
        split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        _, val = next(split.split(eligible, eligible.positive, eligible.patient))
        validation = np.isin(np.arange(len(eligible)), val)
    assert_disjoint(eligible[grouping].to_numpy(), validation)
    return eligible.index.to_numpy()[~validation], eligible.index.to_numpy()[validation]


def objective(task: str, score: dict) -> float:
    return (score['macro_auroc'] + score['macro_auprc']) / 2 if task == 'chest' else score['macro_auroc']


@torch.inference_mode()
def head_logits(model: FeatureClassifier, bank: DeviceBank, indices: np.ndarray, batch: int = 1024) -> np.ndarray:
    model.eval()
    outputs = []
    for rows in np.array_split(indices, max(1, int(np.ceil(len(indices) / batch)))):
        idx = torch.as_tensor(rows, device='cuda')
        with torch.autocast('cuda', dtype=torch.float16):
            z = model(bank.data[idx], bank.masks[idx])
        outputs.append(z.float().cpu().numpy())
    result = np.concatenate(outputs)
    if not np.isfinite(result).all():
        raise ValueError('Feature model returned nonfinite scores')
    return result


def class_weights(labels: np.ndarray, power: float) -> np.ndarray:
    positives = labels.sum(0)
    ratio = (len(labels) - positives + 1.) / (positives + 1.)
    return np.clip(ratio ** power, 1., 6.).astype(np.float32)


def fit(task: str, bank: DeviceBank, labels: np.ndarray, train: np.ndarray, val: np.ndarray,
        recipe: Recipe, directory: Path, epochs: int, seed: int, batch: int, phase: str) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    config = {'task': task, 'recipe': asdict(recipe), 'features': bank.info, 'seed': seed, 'batch': batch,
              'phase': phase, 'split_hash': hashlib.sha256(train.tobytes() + val.tobytes()).hexdigest()}
    torch.manual_seed(seed)
    positions, dim = bank.shape
    model = FeatureClassifier(dim, positions, labels.shape[1], recipe).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate, weight_decay=recipe.decay, fused=True)
    grad = torch.amp.GradScaler('cuda', init_scale=4096.)
    optim_path = directory / 'state.pt'
    history = []
    tune_epoch = task == 'chest' and phase == 'inner'
    if optim_path.is_file():
        saved = torch.load(optim_path, map_location='cpu', weights_only=True)
        if saved['config'] != config:
            raise ValueError('Cannot continue a trial with a changed signature')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        grad.load_state_dict(saved['scaler'])
        history = saved['history']
        if len(history) > epochs:
            prior = (max(history[:epochs], key=lambda record: record['score']) if tune_epoch
                     else next((entry for entry in history if entry['epochs'] == epochs), None))
            if prior is None:
                raise ValueError('Requested earlier resource rung has no recorded score')
            return prior
    index = torch.as_tensor(train, device='cuda')
    y = torch.from_numpy(labels).cuda()
    weights = class_weights(labels[train], recipe.class_power)
    loss_weight = torch.from_numpy(weights).cuda()
    begin = time.monotonic()
    skipped = 0
    for epoch in range(len(history), epochs):
        start_epoch = time.monotonic()
        model.train()
        torch.manual_seed(seed + epoch * 983)
        order = index[torch.randperm(len(index), device='cuda')]
        for rows in order.split(batch):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                z = model(bank.data[rows], bank.masks[rows])
            loss = F.binary_cross_entropy_with_logits(z.float(), y[rows], pos_weight=loss_weight)
            grad.scale(loss).backward()
            grad.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scale = grad.get_scale()
            grad.step(optimizer)
            grad.update()
            skipped += grad.get_scale() < scale
        history.append({'epochs': epoch + 1})
        if tune_epoch:
            metrics = validation_metrics(labels[val], head_logits(model, bank, val))
            history[-1].update(score=objective(task, metrics), metrics=metrics,
                               seconds=round(time.monotonic() - start_epoch, 2), skipped_fp16_steps=skipped)
    if 'score' not in history[-1]:
        logits = head_logits(model, bank, val)
        metrics = validation_metrics(labels[val], logits)
        history[-1].update(score=objective(task, metrics), metrics=metrics,
                           seconds=round(time.monotonic() - begin, 2), skipped_fp16_steps=skipped)
    save_checkpoint(optim_path, {'config': config, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                 'scaler': grad.state_dict(), 'history': history, 'loss_weights': weights.tolist()})
    write_json(directory / 'history.json', {'config': config, 'rungs': [r for r in history if 'score' in r]})
    return max(history, key=lambda record: record['score']) if tune_epoch else history[-1]


def search(task: str, cohort: Path, features: Path, output: Path, seed: int) -> None:
    torch.set_num_threads(4)
    frame, audit = load_manifest(cohort)
    marker = output / 'search.json'
    if marker.is_file():
        saved = read_json(marker)
        if saved['manifest'] != audit['manifest_sha256'] or saved['task'] != task or saved['seed'] != seed:
            raise ValueError('Completed search belongs to a different task, cohort or seed')
        for finalist in saved['finalists']:
            profile = finalist['profile']
            bank_info = read_json(features / profile / 'features.json')
            for fresh_seed in finalist['seeds']:
                path = output / 'final' / profile / str(fresh_seed) / 'model.pt'
                model = torch.load(path, map_location='cpu', weights_only=True)
                if model['config']['features'] != bank_info or model['config']['recipe'] != finalist['recipe']:
                    raise ValueError('Completed finalist differs from current feature profile')
            if not (output / 'validation' / f'{profile}.npz').is_file():
                raise ValueError('Completed search is missing finalist validation predictions')
        print(f'Validated completed {task} hyperparameter search', flush=True)
        return
    inner_train, inner_val = inner_split(frame, task, seed)
    full_train = np.flatnonzero(frame.split.eq('train').to_numpy())
    full_val = np.flatnonzero(frame.split.eq('val').to_numpy())
    labels = frame.loc[:, list(CHEST_LABELS) if task == 'chest' else ['positive']].to_numpy(dtype=np.float32, copy=True)
    budgets = (3, 8, 15) if task == 'chest' else (8, 20, 45)
    batch = 512 if task == 'chest' else 48
    all_results, finalists = [], []
    for profile in PROFILES:
        bank = DeviceBank(features / profile, audit['manifest_sha256'], task, np.r_[full_train, full_val])
        candidates = list(range(len(RECIPES)))
        for rung, epochs in enumerate(budgets):
            scored = []
            for index in candidates:
                directory = output / 'screen' / profile / f'trial-{index:02d}'
                result = fit(task, bank, labels, inner_train, inner_val, RECIPES[index], directory, epochs, seed, batch, 'inner')
                scored.append((result['score'], index))
                all_results.append({'profile': profile, 'trial': index, 'rung': rung + 1, 'budget': epochs,
                                    'recipe': asdict(RECIPES[index]), **result})
            scored.sort(key=lambda item: (-item[0], item[1]))
            print(f'{task}/{profile} rung {rung + 1}: {len(scored)} trained, best trial {scored[0][1]} = {scored[0][0]:.5f}', flush=True)
            candidates = [index for _, index in scored[:(4 if rung == 0 else 2)]]
        index = candidates[0]
        fit_epochs = next(record['epochs'] for record in reversed(all_results) if record['trial'] == index)
        final = []
        for fresh_seed in (seed + 91, seed + 501):
            directory = output / 'final' / profile / str(fresh_seed)
            fit(task, bank, labels, full_train, full_val, RECIPES[index], directory, fit_epochs, fresh_seed, batch, 'refit')
            saved = torch.load(directory / 'state.pt', map_location='cpu', weights_only=True)
            head = FeatureClassifier(bank.shape[1], bank.shape[0], labels.shape[1], RECIPES[index]).cuda()
            head.load_state_dict(saved['model'])
            logs = head_logits(head, bank, full_val) - np.log(saved['loss_weights'])
            final.append(logs)
            save_checkpoint(directory / 'model.pt', {'config': saved['config'], 'model': saved['model'],
                                                      'loss_weights': saved['loss_weights']})
        combined = np.mean(final, axis=0)
        output.joinpath('validation').mkdir(exist_ok=True)
        np.savez_compressed(output / 'validation' / f'{profile}.npz', rows=full_val, logits=combined)
        summary = validation_metrics(labels[full_val], combined)
        finalists.append({'profile': profile, 'trial': index, 'recipe': asdict(RECIPES[index]),
                          'seeds': [seed + 91, seed + 501], 'fit_epochs': fit_epochs, 'validation': summary,
                          'score': objective(task, summary)})
        print(f'{task}/{profile} refit = {objective(task, summary):.5f}', flush=True)
        del bank
        torch.cuda.empty_cache()
    write_json(output / 'search.json', {'task': task, 'manifest': audit['manifest_sha256'], 'seed': seed,
                                       'inner_train_count': len(inner_train), 'inner_val_count': len(inner_val),
                                       'refit_train_count': len(full_train), 'final_val_count': len(full_val),
                                       'budgets': list(budgets), 'trials': all_results, 'finalists': finalists})


def main() -> None:
    parser = argparse.ArgumentParser(description='Successive-halving search over pretrained multi-instance models')
    parser.add_argument('task', choices=('chest', 'mammo'))
    for name in ('cohort', 'features', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    search(args.task, args.cohort, args.features, args.output, args.seed)


if __name__ == '__main__':
    main()
