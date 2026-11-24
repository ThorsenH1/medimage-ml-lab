from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from medimage.common import assert_disjoint, checksum, read_json, write_json

CHEST_LABELS = (
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
    'Fibrosis', 'Pleural_Thickening', 'Hernia',
)
SEED = 271828
CHEST_SIZE = 160
MAMMO_SHAPE = (512, 320)


def chest_manifest(root: Path) -> tuple[pd.DataFrame, dict]:
    annotations = root / 'Data_Entry_2017.csv'
    development, official_test = root / 'train_val_list.txt', root / 'test_list.txt'
    frame = pd.read_csv(annotations).rename(columns={
        'Image Index': 'image', 'Patient ID': 'patient', 'Finding Labels': 'findings',
        'View Position': 'view', 'Patient Age': 'age', 'Patient Gender': 'sex',
    })[['image', 'patient', 'findings', 'view', 'age', 'sex']]
    if frame.image.duplicated().any():
        raise ValueError('Duplicate ChestX-ray14 annotation image names')
    expected = {'train': set(development.read_text().split()), 'test': set(official_test.read_text().split())}
    if expected['train'] & expected['test'] or set(frame.image) != expected['train'] | expected['test']:
        raise ValueError('Official ChestX-ray14 lists do not partition the annotation table')
    frame['split'] = np.where(frame.image.isin(expected['test']), 'test', 'train')
    assert_disjoint(frame.patient.to_numpy(), frame.split.to_numpy())
    paths = {}
    for path in sorted(root.glob('images_*/images/*.png')):
        if path.name in paths:
            raise ValueError('Duplicate local ChestX-ray14 image names')
        paths[path.name] = path.relative_to(root).as_posix()
    audit = {
        'upstream_sha256': {path.name: checksum(path) for path in (annotations, development, official_test)},
        'upstream_images': {key: len(value) for key, value in expected.items()},
        'missing_images_by_official_split': {key: len(value - paths.keys()) for key, value in expected.items()},
        'local_images_without_annotations': len(set(paths) - set(frame.image)),
        'resolution': CHEST_SIZE,
    }
    frame['path'] = frame.image.map(paths)
    frame = frame.dropna(subset=['path']).copy()
    findings = frame.findings.str.split('|').map(set)
    if set().union(*findings) - set(CHEST_LABELS) - {'No Finding'}:
        raise ValueError('Unknown ChestX-ray14 finding')
    for label in CHEST_LABELS:
        frame[label] = findings.map(lambda row: int(label in row))
    development_rows = frame.loc[frame.split == 'train']
    patients = development_rows.groupby('patient', sort=True)[list(CHEST_LABELS)].max()
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.12, random_state=SEED)
    _, validation = next(splitter.split(np.zeros(len(patients)), patients))
    frame.loc[frame.patient.isin(patients.index[validation]), 'split'] = 'val'
    assert_disjoint(frame.patient.to_numpy(), frame.split.to_numpy())
    counts = frame.groupby('split')
    audit['splits'] = {
        key: {'images': len(part), 'patients': part.patient.nunique(),
              'positives': part[list(CHEST_LABELS)].sum().to_dict()}
        for key, part in counts
    }
    return frame.sort_values('path').reset_index(drop=True), audit


def _series_uid(value: str) -> str:
    return PurePosixPath(value.strip()).parent.name


def mammography_manifest(root: Path) -> tuple[pd.DataFrame, dict]:
    csv_root = root / 'csv'
    mapping_file = csv_root / 'dicom_info.csv'
    mapping = pd.read_csv(mapping_file, usecols=['image_path', 'SeriesDescription', 'SeriesInstanceUID',
                                                 'StudyInstanceUID', 'Rows', 'Columns'], keep_default_na=False)
    known = mapping.SeriesDescription.str.lower().eq('full mammogram images')
    unlabeled = mapping.SeriesDescription.str.strip().eq('') & mapping.Rows.ge(1800) & mapping.Columns.ge(1800)
    mapping = mapping[known | unlabeled]
    available = defaultdict(set)
    recovered = defaultdict(set)
    for row in mapping.itertuples(index=False):
        parts = PurePosixPath(row.image_path.strip()).parts
        if len(parts) != 4 or parts[1] != 'jpeg' or parts[2] != row.SeriesInstanceUID or '..' in parts:
            raise ValueError('Invalid mammography JPEG mapping')
        relative = Path(*parts[1:])
        if (root / relative).is_file():
            if row.SeriesDescription.strip():
                available[row.SeriesInstanceUID].add(relative.as_posix())
            else:
                recovered[(row.SeriesInstanceUID, row.StudyInstanceUID)].add(relative.as_posix())
    files = [csv_root / f'{kind}_case_description_{split}_set.csv'
             for kind in ('mass', 'calc') for split in ('train', 'test')]
    detail_frames = []
    lesion_audit = Counter()
    for path in files:
        table = pd.read_csv(path).rename(columns={
            'patient_id': 'patient', 'left or right breast': 'side', 'image view': 'view',
            'image file path': 'source',
        })[['patient', 'side', 'view', 'pathology', 'source']]
        if set(table.pathology) - {'MALIGNANT', 'BENIGN', 'BENIGN_WITHOUT_CALLBACK'}:
            raise ValueError('Unknown CBIS-DDSM pathology category')
        table['positive'] = table.pathology.eq('MALIGNANT').astype(int)
        table['official_split'] = 'train' if '_train_' in path.name else 'test'
        table['lesion_type'] = path.name.split('_')[0]
        table['series'] = table.source.map(_series_uid)
        paths = []
        metadata_recovered = 0
        for series, source in zip(table.series, table.source, strict=True):
            documented = available[series]
            # Some official Calc-Test full-image rows omit SeriesDescription.
            # Require the listed full-image series UID, study UID and full-resolution dimensions.
            inferred = recovered[(series, PurePosixPath(source.strip()).parts[1])]
            candidates = documented or inferred
            paths.append(next(iter(candidates)) if len(candidates) == 1 else '')
            metadata_recovered += int(not documented and len(inferred) == 1)
        table['path'] = paths
        lesion_audit[f'{table.official_split.iloc[0]}_rows'] += len(table)
        lesion_audit[f'{table.official_split.iloc[0]}_unmapped_rows'] += int(table.path.eq('').sum())
        lesion_audit[f'{table.official_split.iloc[0]}_recovered_missing_series_description'] += metadata_recovered
        detail_frames.append(table)
    lesions = pd.concat(detail_frames, ignore_index=True)
    if set(lesions.side) - {'LEFT', 'RIGHT'} or set(lesions.view) - {'CC', 'MLO'}:
        raise ValueError('Unexpected CBIS-DDSM side or projection')
    test_patients = set(lesions.loc[lesions.official_split == 'test', 'patient'])
    train_patients = set(lesions.loc[lesions.official_split == 'train', 'patient'])
    overlap = test_patients & train_patients
    # Discard all rows for patients assigned inconsistently by the two lesion subsets.
    lesions = lesions.loc[~lesions.patient.isin(overlap)].copy()
    records = []
    ambiguous = 0
    for (patient, side), group in lesions.groupby(['patient', 'side'], sort=True):
        views = {view: sorted(set(group.loc[group.view.eq(view), 'path']) - {''}) for view in ('CC', 'MLO')}
        if any(len(paths) > 1 for paths in views.values()):
            ambiguous += 1
            continue
        if not any(views.values()):
            continue
        records.append({'patient': patient, 'side': side, 'split': group.official_split.iloc[0],
                        'positive': group.positive.max(), 'type': '+'.join(sorted(set(group.lesion_type))),
                        **{view: paths[0] if paths else '' for view, paths in views.items()}})
    frame = pd.DataFrame.from_records(records)
    development = frame.loc[frame.split.eq('train')]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    _, validation = next(splitter.split(development, development.positive, development.patient))
    frame.loc[development.index[validation], 'split'] = 'val'
    assert_disjoint(frame.patient.to_numpy(), frame.split.to_numpy())
    for view in ('CC', 'MLO'):
        if frame.loc[frame[view].ne(''), view].duplicated().any():
            raise ValueError('Full mammogram reused by multiple breasts')
    audit = {
        'upstream_sha256': {path.name: checksum(path) for path in [mapping_file, *files]},
        'lesion_rows': dict(lesion_audit), 'excluded_cross_split_patients': len(overlap),
        'excluded_ambiguous_breasts': ambiguous,
        'breasts_with_no_mapped_view': lesions.groupby(['patient', 'side']).path.apply(lambda col: col.eq('').all()).sum().item(),
        'resolution': list(MAMMO_SHAPE),
        'splits': {key: {'breasts': len(part), 'patients': part.patient.nunique(),
                         'positive': int(part.positive.sum()),
                         'both_views': int((part.CC.ne('') & part.MLO.ne('')).sum())}
                   for key, part in frame.groupby('split')},
    }
    return frame, audit


def read_image(path: Path, flags: int) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None or image.ndim != 2:
        raise ValueError('An indexed image is unreadable or has unexpected channels')
    return image


def resize_chest(path: Path) -> np.ndarray:
    image = read_image(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(image, (CHEST_SIZE, CHEST_SIZE), interpolation=cv2.INTER_AREA)


def prepare_breast(path: Path, output_shape: tuple[int, int] = MAMMO_SHAPE) -> np.ndarray:
    image = read_image(path, cv2.IMREAD_REDUCED_GRAYSCALE_2)
    preview = cv2.resize(image, (160, 256), interpolation=cv2.INTER_AREA)
    threshold = max(8, float(np.percentile(preview, 90)) * .12)
    mask = (preview > threshold).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        raise ValueError('No breast foreground was found')
    component = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    x, y, width, height, area = stats[component]
    if area < .07 * preview.size:
        raise ValueError('Breast foreground occupies less than seven percent of the image')
    top = max(0, int((y - 2) * image.shape[0] / 256))
    bottom = min(image.shape[0], int(np.ceil((y + height + 2) * image.shape[0] / 256)))
    left = max(0, int((x - 2) * image.shape[1] / 160))
    right = min(image.shape[1], int(np.ceil((x + width + 2) * image.shape[1] / 160)))
    tissue = image[top:bottom, left:right].copy()
    foreground = cv2.resize((components == component).astype(np.uint8),
                            (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)[top:bottom, left:right]
    tissue[foreground == 0] = 0
    wing = max(1, tissue.shape[1] // 6)
    if foreground[:, -wing:].sum() > foreground[:, :wing].sum():
        tissue = tissue[:, ::-1]
    scaled_height, scaled_width = output_shape
    return cv2.resize(tissue, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)


def prepare(task: str, root: Path, output: Path, workers: int) -> None:
    cv2.setNumThreads(1)
    output.mkdir(parents=True, exist_ok=True)
    version = 1 if task == 'chest' else 4
    if (output / 'audit.json').exists():
        audit = read_json(output / 'audit.json')
        if audit['pipeline_version'] == version and all((output / name).is_file() for name in ('images.npy', 'manifest.csv')):
            for filename, digest in audit['upstream_sha256'].items():
                source = (root if task == 'chest' else root / 'csv') / filename
                if checksum(source) != digest:
                    raise ValueError('Source CSV or partition changed; prepare into a new cache directory')
            print(f'Found validated {task} cache', flush=True)
            return
        raise ValueError('Cache artifact is incomplete or has an incompatible version')
    if task == 'chest':
        frame, audit = chest_manifest(root)
        shape = (len(frame), CHEST_SIZE, CHEST_SIZE)

        def load(row):
            return resize_chest(root / row.path)
    else:
        frame, audit = mammography_manifest(root)
        shape = (len(frame), 2, *MAMMO_SHAPE)

        def load(row):
            return np.stack([prepare_breast(root / value) if value else np.zeros(MAMMO_SHAPE, np.uint8)
                             for value in (row.CC, row.MLO)])

    print({'task': task, 'splits': audit['splits'], 'shape': shape}, flush=True)
    start, batch_size = 0, 256
    state_path = output / 'cache-progress.json'
    partial = output / 'images.partial.npy'
    complete = output / 'images.npy'
    manifest = output / 'manifest.csv'
    frame.to_csv(output / 'manifest.partial', index=False)
    manifest_hash = checksum(output / 'manifest.partial')
    if state_path.exists():
        state = read_json(state_path)
        if state['manifest'] != manifest_hash:
            raise ValueError('Source inventory changed during cache creation')
        start = state['written']
    if start == len(frame) and complete.exists():
        data = None
    else:
        data = np.lib.format.open_memmap(partial, mode='r+' if start else 'w+', dtype=np.uint8, shape=shape)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in tqdm(range(start, len(frame), batch_size), desc=f'Prepare {task}', mininterval=8):
                end = min(offset + batch_size, len(frame))
                rows = frame.iloc[offset:end].itertuples(index=False)
                for index, image in enumerate(pool.map(load, rows), offset):
                    data[index] = image
                data.flush()
                write_json(state_path, {'manifest': manifest_hash, 'written': end})
        del data
        os.replace(partial, complete)
    os.replace(output / 'manifest.partial', manifest)
    audit.update(pipeline_version=version, manifest_sha256=manifest_hash, array_shape=list(shape))
    write_json(output / 'audit.json', audit)
    state_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit cohort and make a resumable pixel cache')
    parser.add_argument('task', choices=['chest', 'mammo'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    prepare(args.task, args.root, args.output, args.workers)


if __name__ == '__main__':
    main()
