from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def checksum(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def assert_disjoint(groups: np.ndarray, split: np.ndarray) -> None:
    memberships = {}
    for group, partition in zip(groups, split, strict=True):
        prior = memberships.setdefault(group, partition)
        if prior != partition:
            raise ValueError(f'A patient/study occurs in both {prior} and {partition}')


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = True
