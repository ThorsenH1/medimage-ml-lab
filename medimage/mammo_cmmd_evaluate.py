"""Compare fixed CBIS-only and CMMD-adapted checkpoints on held-out CMMD patients."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from medimage.auc_selection import checked, load_component
from medimage.auc_train import predict, store
from medimage.common import assert_disjoint, checksum, read_json, write_json


def evaluate(args: argparse.Namespace) -> None:
    torch.set_num_threads(3)
    frame, audit, arrays, pixel_digest = store('mammo', args.cohort, args.cohort)
    if audit.get('source') != 'TCIA CMMD':
        raise ValueError('Expected the prepared TCIA CMMD cohort')
    assert_disjoint(frame.patient.to_numpy(), frame.split.to_numpy())
    cohort_digest = audit['manifest_sha256']
    rows = np.flatnonzero(frame.split.eq('test'))
    test = frame.iloc[rows]
    truth = test.positive.to_numpy()
    groups = test.patient.to_numpy()
    patient_rows = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    if len(np.unique(truth)) != 2 or len(patient_rows) < 2:
        raise ValueError('Both labels and multiple patients are required in the CMMD test split')

    candidates = {}
    for title, run in (('CBIS-only', args.source_run), ('CBIS-to-CMMD', args.adapted_run)):
        result = read_json(checked(args.work, run) / 'result.json')
        config = result['config']
        candidates[title] = {'run': run, 'config': config, 'checkpoint_sha256': result['best_sha256'],
                             'evaluation': 'external dataset' if title == 'CBIS-only'
                             else 'held-out patients at training site'}
    source, adapted = (candidate['config'] for candidate in candidates.values())
    if (source['task'] != 'mammo' or adapted['task'] != 'mammo' or source['manifest'] == cohort_digest
            or source.get('parent_manifest') == cohort_digest or adapted['manifest'] != cohort_digest
            or adapted['pixel_sha256'] != pixel_digest or adapted.get('parent_manifest') != source['manifest']
            or adapted['parent_sha256'] != candidates['CBIS-only']['checkpoint_sha256']):
        raise ValueError('The adapted checkpoint must descend from the CBIS-only checkpoint and use this CMMD cohort')

    plan = {'cohort_sha256': cohort_digest, 'pixel_audit_sha256': pixel_digest, 'partition': 'test',
            'breasts': len(rows), 'patients': len(patient_rows), 'positives': int(truth.sum()),
            'metrics': ['AUROC', 'average precision'], 'candidates': candidates}
    selection = args.output / 'selection.json'
    if selection.exists() and read_json(selection) != plan:
        raise ValueError('Use a new output directory for a different test plan')
    write_json(selection, plan)

    random = np.random.default_rng(args.seed)
    draws = random.integers(0, len(patient_rows), size=(args.repetitions, len(patient_rows)))
    resamples = [np.concatenate([patient_rows[index] for index in draw]) for draw in draws]
    resamples = [indices for indices in resamples if np.unique(truth[indices]).size == 2]
    if len(resamples) < 2:
        raise ValueError('Too few patient resamples contain both classes')
    results, distributions = {}, []
    for title, component in candidates.items():
        model, saved = load_component(args.work, component, args.weights, None)
        scores = predict(model, 'mammo', arrays, rows, args.batch, saved['config']['tta'],
                         np.asarray(saved['prior_correction'], dtype=np.float32))[:, 0]
        del model
        torch.cuda.empty_cache()
        distribution = np.array([roc_auc_score(truth[indices], scores[indices]) for indices in resamples])
        distributions.append(distribution)
        results[title] = {'auroc': roc_auc_score(truth, scores), 'auprc': average_precision_score(truth, scores),
                          'auroc_95_patient_bootstrap': np.quantile(distribution, [.025, .975]).tolist(),
                          'best_epoch': saved['epoch'], 'validation': saved['validation']}
        print(title, results[title], flush=True)

    write_json(args.output / 'metrics.json', {
        'selection_sha256': checksum(selection), 'test_breasts': len(rows), 'test_patients': len(patient_rows),
        'results': results, 'resamples': len(resamples), 'seed': args.seed,
        'paired_delta_auroc_95': np.quantile(distributions[1] - distributions[0], [.025, .975]).tolist(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort', 'work', 'weights', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--source-run', default='runs/mammo-sharp1536')
    parser.add_argument('--adapted-run', default='runs/cmmd-effb0')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--repetitions', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=90244)
    args = parser.parse_args()
    if args.batch < 1 or args.repetitions < 2:
        parser.error('Use a positive batch size and at least two patient resamples')
    evaluate(args)


if __name__ == '__main__':
    main()
