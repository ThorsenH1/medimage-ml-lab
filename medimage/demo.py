from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from medimage.spine_data import SEVERITIES, SERIES, TARGETS

PHANTOM_STUDY = 999000


def save_png(path: Path, pixels: np.ndarray) -> None:
    image = np.clip(pixels, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError('PNG encoding failed')
    path.write_bytes(encoded.tobytes())


def xray_phantom(output: Path) -> None:
    y, x = np.mgrid[-1:1:512j, -1:1:512j]
    profile = 12 + 130 * np.exp(-((x / .76) ** 2 + (y / .85) ** 2))
    chambers = np.exp(-(((np.abs(x) - .33) / .19) ** 2 + (y / .65) ** 2))
    bands = 15 * np.cos((y + .25 * x ** 2) * 36) ** 8 * np.exp(-((x / .83) ** 2 + y ** 2))
    column = 75 * np.exp(-((x / .08) ** 2 + (y / .82) ** 2))
    save_png(output / 'phantom.png', profile - 47 * chambers + bands + column)


def mammography_phantom(output: Path) -> None:
    y, x = np.mgrid[-1:1:1000j, 0:1:740j]
    for view, displacement in (('cc', 0.), ('mlo', -.14)):
        support = (x / .86) ** 2 + ((y - displacement + .15 * x) / .87) ** 2
        gradient = np.where(support < 1., 19 + 140 * (1 - support), 0)
        tissue = 21 * np.cos(40 * x + 27 * y) ** 2 * np.exp(-((x - .38) ** 2 + y ** 2) * 11)
        shape = 38 * np.exp(-((x - .48) ** 2 * 54 + (y - displacement - .07) ** 2 * 49))
        save_png(output / f'phantom-{view}.png', np.where(support < 1., gradient + tissue + shape, 0))


def dicom_slice(path: Path, image: np.ndarray, series_uid: str, study_uid: str, number: int,
                origin: list[float], orientation: list[float]) -> None:
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    item = FileDataset(path, {}, file_meta=meta, preamble=bytes(128))
    item.SOPClassUID = MRImageStorage
    item.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    item.StudyInstanceUID = study_uid
    item.SeriesInstanceUID = series_uid
    item.PatientID = 'SYNTHETIC-PHANTOM'
    item.Modality = 'MR'
    item.InstanceNumber = number
    item.ImagePositionPatient = origin
    item.ImageOrientationPatient = orientation
    item.PixelSpacing = [1., 1.]
    item.Rows, item.Columns = image.shape
    item.SamplesPerPixel = 1
    item.PhotometricInterpretation = 'MONOCHROME2'
    item.PixelRepresentation = 0
    item.BitsAllocated = 16
    item.BitsStored = 12
    item.HighBit = 11
    item.PixelData = image.astype('<u2').tobytes()
    item.save_as(path, enforce_file_format=True)


def mri_phantom(output: Path) -> None:
    y, x = np.mgrid[-1:1:160j, -1:1:160j]
    rows, previews = [], []
    study_uid = generate_uid()
    for view, description in enumerate(SERIES):
        series_id = 9901 + view
        rows.append({'study_id': PHANTOM_STUDY, 'series_id': series_id, 'series_description': description})
        series_uid = generate_uid()
        target = output / 'test_images' / str(PHANTOM_STUDY) / str(series_id)
        target.mkdir(parents=True, exist_ok=True)
        depth = 19 if view == 2 else 9
        for index in range(depth):
            t = (index - depth // 2) / (depth // 2)
            if view == 2:
                image = 230 + 950 * np.exp(-((x / .58) ** 2 + (y / .67) ** 2))
                image += 570 * np.exp(-((x - .12 * t) ** 2 + (y + .20) ** 2) * 19)
                vectors, origin = [1., 0., 0., 0., 1., 0.], [-80., -80., 4.5 * (depth // 2 - index)]
            else:
                stripe = np.exp(-((x - .13 * y ** 2 + .07 * t) / .33) ** 2 - (y / .82) ** 2)
                image = 205 + 900 * stripe * (1.1 + .32 * np.cos(19 * y))
                image += 175 * np.exp(-(((x + .32) / .09) ** 2 + (y / .83) ** 2))
                vectors, origin = [0., 1., 0., 0., 0., -1.], [3. * (index - depth // 2), -80., 80.]
            image += 17. * np.sin(18 * x + 21 * y + t)
            dicom_slice(target / f'{index + 1}.dcm', image, series_uid, study_uid, index + 1, origin, vectors)
            if index == depth // 2:
                previews.append(255 * (image - image.min()) / (image.max() - image.min()))
    pd.DataFrame(rows).to_csv(output / 'test_series_descriptions.csv', index=False)
    template = pd.DataFrame(np.full((25, 3), 1. / 3), columns=SEVERITIES)
    template.insert(0, 'row_id', [f'{PHANTOM_STUDY}_{target}' for target in TARGETS])
    template.to_csv(output / 'sample_submission.csv', index=False)
    save_png(output / 'phantom.png', np.concatenate(previews, axis=1))


def main() -> None:
    parser = argparse.ArgumentParser(description='Build non-patient synthetic phantoms for inference demonstrations')
    parser.add_argument('task', choices=('chest', 'mammo', 'spine'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {'chest': xray_phantom, 'mammo': mammography_phantom, 'spine': mri_phantom}[args.task](args.output)


if __name__ == '__main__':
    main()
