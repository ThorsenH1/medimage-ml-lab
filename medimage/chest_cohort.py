from __future__ import annotations

import argparse
import hashlib
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from medimage.common import assert_disjoint, checksum, read_json, write_json
from medimage.radiography_data import CHEST_LABELS, SEED


def exact_pixel_groups(images: np.ndarray) -> list[list[int]]:
    first = {}
    repeated = defaultdict(list)
    for index, image in enumerate(images):
        digest = hashlib.blake2b(image, digest_size=16).digest()
        anchor = first.setdefault(digest, index)
        if anchor != index:
            if not np.array_equal(image, images[anchor]):
                raise ValueError('Nonidentical pixels share an image fingerprint')
            repeated[anchor].append(index)
    return [[anchor, *duplicates] for anchor, duplicates in repeated.items()]


def grouped_partitions(frame: pd.DataFrame, duplicates: list[list[int]]) -> tuple[pd.DataFrame, dict]:
    frame = frame.copy()
    patients = frame.patient.to_numpy()
    parent = {patient: patient for patient in set(patients)}

    def root(patient):
        while parent[patient] != patient:
            parent[patient] = parent[parent[patient]]
            patient = parent[patient]
        return patient

    for indices in duplicates:
        identifiers = sorted({root(patients[index]) for index in indices})
        for patient in identifiers[1:]:
            parent[patient] = identifiers[0]
    frame['pixel_cluster'] = [root(patient) for patient in patients]
    test = frame.split.eq('test')
    test_clusters = set(frame.loc[test, 'pixel_cluster'])
    exclude = ~test & frame.pixel_cluster.isin(test_clusters)
    development = frame.loc[~test & ~exclude]
    cluster_labels = development.groupby('pixel_cluster', sort=True)[list(CHEST_LABELS)].max()
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.12, random_state=SEED)
    _, validation = next(splitter.split(np.zeros(len(cluster_labels)), cluster_labels))
    validation_clusters = cluster_labels.index[validation]
    frame.loc[~test, 'split'] = 'train'
    frame.loc[~test & frame.pixel_cluster.isin(validation_clusters), 'split'] = 'val'
    frame.loc[exclude, 'split'] = 'excluded'
    retained = frame.loc[~exclude]
    for group in ('patient', 'pixel_cluster'):
        assert_disjoint(retained[group].to_numpy(), retained.split.to_numpy())
    if not frame.split.eq('test').equals(test):
        raise AssertionError('Official test images changed during alias grouping')
    for indices in duplicates:
        if frame.iloc[indices].loc[lambda rows: rows.split.ne('excluded'), 'split'].nunique() > 1:
            raise AssertionError('Identical pixels occur across fit, validation or test')
    audit = {'duplicates_groups': len(duplicates), 'duplicate_rows_after_first': sum(len(group) - 1 for group in duplicates),
             'development_patients_excluded_for_test_alias': frame.loc[exclude, 'patient'].nunique(),
             'development_images_excluded_for_test_alias': int(exclude.sum()),
             'splits': {key: {'images': len(part), 'patients': part.patient.nunique(),
                              'pixel_clusters': part.pixel_cluster.nunique(),
                              'positives': part[list(CHEST_LABELS)].sum().to_dict()}
                        for key, part in retained.groupby('split')}}
    return frame, audit


def prepare(cache: Path) -> None:
    audit_path = cache / 'audit.json'
    path = cache / 'manifest.csv'
    original_audit_path = cache / 'original-audit.json'
    original_manifest = cache / 'original-manifest.csv'
    initial = read_json(audit_path)
    audit = read_json(original_audit_path) if original_audit_path.is_file() else initial
    signature = {'code_sha256': checksum(Path(__file__)), 'source_manifest_sha256': audit['manifest_sha256']}
    if 'duplicate_policy' in initial:
        if initial['duplicate_policy'] == signature and initial['manifest_sha256'] == checksum(path):
            print('Found verified ChestX-ray14 cohort with grouped duplicate images', flush=True)
            return
        raise ValueError('Grouped chest cache provenance does not match the input cache')
    if not original_manifest.is_file():
        if checksum(path) != audit['manifest_sha256']:
            raise ValueError('Chest source manifest hash mismatch')
        temporary_original = original_manifest.with_suffix('.partial')
        temporary_original.write_bytes(path.read_bytes())
        os.replace(temporary_original, original_manifest)
    if checksum(original_manifest) != audit['manifest_sha256']:
        raise ValueError('Chest source manifest hash mismatch')
    write_json(original_audit_path, audit)
    images = np.load(cache / 'images.npy', mmap_mode='r', allow_pickle=False)
    duplicates = exact_pixel_groups(images)
    frame = pd.read_csv(original_manifest, keep_default_na=False)
    if len(images) != len(frame):
        raise ValueError('Chest cache pixels and metadata differ in length')
    old_cross = sum(frame.iloc[group].split.nunique() > 1 for group in duplicates)
    cohort, details = grouped_partitions(frame, duplicates)
    temporary = path.with_suffix('.partial')
    cohort.to_csv(temporary, index=False)
    os.replace(temporary, path)
    audit['original_split_summary'] = audit['splits']
    audit.update(details)
    audit.update(duplicate_policy=signature, original_cross_partition_duplicate_groups=old_cross,
                 manifest_sha256=checksum(path))
    write_json(audit_path, audit)
    print({'original_cross_partition_groups': old_cross, **details}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Group pixel-identical NIH images before fitting or model selection')
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.cache)


if __name__ == '__main__':
    main()
