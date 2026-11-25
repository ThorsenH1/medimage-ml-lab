"""Resolve the project's CBIS-DDSM projections to their original public DICOM series."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from medimage.common import load_manifest, read_json, write_json


def build(args: argparse.Namespace, index: pd.DataFrame) -> None:
    frame, audit = load_manifest(args.cohort)
    headers = ['image_path', 'SeriesInstanceUID', 'SeriesDescription', 'Rows', 'Columns']
    table = pd.read_csv(args.project_root / 'csv' / 'dicom_info.csv', usecols=headers, keep_default_na=False)
    table['path'] = table.image_path.str.removeprefix('CBIS-DDSM/')
    projections = frame.melt(id_vars=['split', 'type'], value_vars=['CC', 'MLO'], value_name='path')
    projections = projections.loc[projections.path.ne('')]
    matches = projections.merge(table, how='left', on='path', validate='one_to_one')
    available = index.loc[index.collection_id.eq('cbis_ddsm'), ['SeriesInstanceUID', 'series_size_MB']]
    matches = matches.merge(available, how='left', on='SeriesInstanceUID', validate='one_to_one')
    if matches.series_size_MB.isna().any() or matches.Rows.min() < 1000 or matches.Columns.min() < 500:
        raise ValueError('Native registry contains a missing or undersized original projection')
    fields = ['split', 'type', 'path', 'SeriesInstanceUID', 'series_size_MB']
    plan = {'cohort_sha256': audit['manifest_sha256'], 'views': matches[fields].to_dict('records')}
    if args.plan.exists() and read_json(args.plan) != plan:
        raise ValueError('Use a separate file for a different frozen acquisition registry')
    write_json(args.plan, plan)
    print(matches.groupby('split').series_size_MB.agg(views='size', source_MB='sum').to_string(), flush=True)


def complete(root: Path, row: dict) -> bool:
    files = list((root / row['SeriesInstanceUID']).glob('*.dcm'))
    return len(files) == 1 and files[0].stat().st_size == round(row['series_size_MB'] * 1_000_000)


def download(args: argparse.Namespace, client) -> None:
    plan = read_json(args.plan)
    for split in ('train', 'val', 'test'):
        rows = [row for row in plan['views'] if row['split'] == split]
        expected = {row['SeriesInstanceUID'] for row in rows}
        if not expected or len(rows) != len(expected):
            raise ValueError('A split contains an empty or nonunique native DICOM registry')
        root = args.destination / split
        remaining = [row for row in rows if not complete(root, row)]
        print(f"{split}: {len(expected)} projection series, {len(remaining)} remaining, "
              f"{sum(row['series_size_MB'] for row in remaining) / 1000:.2f} GB", flush=True)
        if remaining:
            client.download_dicom_series([row['SeriesInstanceUID'] for row in remaining],
                                         downloadDir=str(root), quiet=True,
                                         show_progress_bar=False, dirTemplate='%SeriesInstanceUID')
        verified = sum(complete(root, row) for row in rows)
        if verified != len(expected):
            raise ValueError(f'{split} has {verified} of {len(expected)} complete original series')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='phase', required=True)
    create = commands.add_parser('plan')
    fetch = commands.add_parser('download')
    for command, names in ((create, ('cohort', 'project-root', 'plan')), (fetch, ('plan', 'destination'))):
        for name in names:
            command.add_argument(f'--{name}', required=True, type=Path)
    args = parser.parse_args()
    from idc_index import IDCClient

    client = IDCClient()
    if args.phase == 'plan':
        build(args, client.index)
    else:
        download(args, client)


if __name__ == '__main__':
    main()
