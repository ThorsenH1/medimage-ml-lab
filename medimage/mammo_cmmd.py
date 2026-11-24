"""Prepare a patient-disjoint breast-level cohort from the public TCIA CMMD release."""
from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
import pydicom

from medimage.auc_mammo_data import view_array
from medimage.common import checksum, write_json

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
VIEWS = {'399162004': 'CC', '399368009': 'MLO'}


def grade_table(path: Path) -> dict:
    with ZipFile(path) as archive:
        shared = [''.join(item.itertext()) for item in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        sheet = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
    records = []
    for row in sheet.findall('.//m:row', NS):
        record = {}
        for cell in row:
            value = cell.find('m:v', NS)
            address = re.match(r'[A-Z]+', cell.attrib['r'])
            if value is not None and value.text is not None:
                record[address[0]] = shared[int(value.text)] if cell.attrib.get('t') == 's' else value.text
        records.append(record)
    headings = {value: key for key, value in records[0].items()}
    required = {'ID1', 'LeftRight', 'classification', 'abnormality'}
    if not required.issubset(headings):
        raise ValueError('The worksheet differs from the documented TCIA data release')
    output = defaultdict(set)
    for row in records[1:]:
        primary = row[headings['ID1']].strip()
        patient = row.get(headings.get('ID2'), primary).strip() or primary
        side = row[headings['LeftRight']].strip().upper()
        status = row[headings['classification']].strip().lower()
        if side not in ('L', 'R') or status not in ('benign', 'malignant'):
            continue
        lesion = row[headings['abnormality']].strip().lower()
        kind = 'both' if lesion == 'both' or ('calc' in lesion and 'mass' in lesion) else (
            'calc' if 'calc' in lesion else 'mass' if 'mass' in lesion else 'other')
        output[patient, side].add((int(status == 'malignant'), kind))
    return {key: (next(iter({label for label, _ in values})),
                  next(iter({kind for _, kind in values})) if len({kind for _, kind in values}) == 1 else 'both')
            for key, values in output.items() if len({label for label, _ in values}) == 1}


def image_info(path: Path):
    keys = ('PatientID', 'ImageLaterality', 'ViewCodeSequence', 'Rows', 'Columns', 'SOPInstanceUID')
    dataset = pydicom.dcmread(path, specific_tags=keys, stop_before_pixels=True)
    coded = dataset.get('ViewCodeSequence', [])
    view = VIEWS.get(str(coded[0].CodeValue)) if coded else None
    laterality = str(dataset.get('ImageLaterality', '')).strip().upper()
    if view is None or laterality not in ('L', 'R'):
        return None
    variants = coded[0].get('ViewModifierCodeSequence', [])
    if any('magnif' in str(item.get('CodeMeaning', '')).lower() for item in variants):
        return None
    return ((str(dataset.PatientID), laterality), view, (int(dataset.Rows) * int(dataset.Columns),
            str(dataset.SOPInstanceUID)), path)


def group_partition(patient: str) -> str:
    value = int(hashlib.sha256(f'cmmd-breast-v1:{patient}'.encode()).hexdigest()[:8], 16) / 2**32
    return 'train' if value < .65 else 'val' if value < .80 else 'test'


def normalized_view(path: Path, canvas: tuple[int, int]) -> np.ndarray:
    dataset = pydicom.dcmread(path)
    raw = dataset.pixel_array
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError('Expected a two-dimensional mammogram')
    if raw.dtype == np.uint8 and dataset.PhotometricInterpretation == 'MONOCHROME2':
        image = raw
    else:
        pixels = raw.astype(np.float32)
        if dataset.PhotometricInterpretation == 'MONOCHROME1':
            pixels *= -1
        low, high = np.percentile(pixels, [.5, 99.9])
        if not high > low:
            raise ValueError('Mammogram has no dynamic range')
        image = np.clip((pixels - low) * 255 / (high - low), 0, 255).astype(np.uint8)
    return view_array(image, canvas)[0]


def prepare(args: argparse.Namespace) -> None:
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    labels = grade_table(args.labels)
    files = sorted(args.dicom.rglob('*.dcm'))
    if args.expected and len(files) != args.expected:
        raise ValueError(f'Expected {args.expected} complete DICOMs, found {len(files)}')
    if not files:
        raise ValueError('The CMMD data directory contains no DICOM files')
    options = defaultdict(dict)
    with ThreadPoolExecutor(args.workers) as pool:
        for info in pool.map(image_info, files):
            if info is None:
                continue
            key, name, rank, path = info
            if key in labels and (name not in options[key] or rank > options[key][name][0]):
                options[key][name] = (rank, path)
    rows = []
    for (patient, side), scans in sorted(options.items()):
        status, lesion = labels[patient, side]
        rows.append({'patient': patient, 'side': side, 'split': group_partition(patient),
                     'positive': status, 'type': lesion,
                     **{view: scans[view][1].relative_to(args.dicom).as_posix() if view in scans else ''
                        for view in VIEWS.values()}})
    if not rows:
        raise ValueError('No DICOM projection could be matched to the labeled sides')
    args.output.mkdir(parents=True, exist_ok=False)
    shape = (args.height, args.height * 5 // 8)
    values = np.lib.format.open_memmap(args.output / 'pixels.npy', mode='w+', dtype=np.uint8,
                                       shape=(len(rows), 2, *shape))
    owners = {}
    overlap = set()
    skipped = Counter()
    for start in range(0, len(rows), 32):
        stop = min(start + 32, len(rows))
        values[start:stop] = 0
        positions = [(index, column, args.dicom / rows[index][name])
                     for index in range(start, stop)
                     for column, name in enumerate(('CC', 'MLO')) if rows[index][name]]
        with ThreadPoolExecutor(args.workers) as pool:
            for (index, column, _), pixels in zip(
                    positions, pool.map(lambda item: normalized_view(item[2], shape), positions), strict=True):
                values[index, column] = pixels
                digest = hashlib.sha256(pixels).digest()
                patient = rows[index]['patient']
                original = owners.setdefault(digest, patient)
                if original != patient:
                    overlap.update((original, patient))
        if start % 256 == 0:
            print(f'Prepared {min(start + 32, len(rows))} of {len(rows)} labeled breasts', flush=True)
    values.flush()
    del values
    for row in rows:
        if row['patient'] in overlap:
            row['split'] = 'excluded'
            skipped['duplicate_linked_breasts'] += 1
    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.output / 'manifest.csv', index=False)
    np.save(args.output / 'views.npy', manifest[['CC', 'MLO']].ne('').to_numpy(), allow_pickle=False)
    np.save(args.output / 'heatmaps.npy',
            np.zeros((len(rows), 2, 4, shape[0] // 32, shape[1] // 32), dtype=np.uint8), allow_pickle=False)
    np.save(args.output / 'annotation_available.npy', np.zeros((len(rows), 2), bool), allow_pickle=False)
    groups = {name: {'breasts': int(len(part)), 'patients': int(part.patient.nunique()),
                     'positives': int(part.positive.sum())} for name, part in manifest.groupby('split')}
    if not {'train', 'val', 'test'}.issubset(groups):
        raise ValueError('The public cohort did not produce three patient-grouped partitions')
    digest = checksum(args.output / 'manifest.csv')
    record = {'source': 'TCIA CMMD', 'doi': '10.7937/tcia.eqde-4b16',
              'manifest_sha256': digest, 'workbook_sha256': checksum(args.labels), 'dicoms': len(files),
              'splits': groups, 'excluded': skipped, 'config': {'cohort': digest,
              'shape': [len(rows), 2, *shape], 'split_protocol': 'patient hash, 65/15/20; exact view repeats removed'}}
    write_json(args.output / 'audit.json', record)
    print(groups, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dicom', 'labels', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--height', type=int, choices=(1024, 1536), default=1536)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--expected', type=int, default=5202)
    args = parser.parse_args()
    if args.workers < 1 or args.expected < 0:
        parser.error('Workers must be positive and the expected DICOM count cannot be negative')
    prepare(args)


if __name__ == '__main__':
    main()
