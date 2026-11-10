"""Signer-grouped CV confined to official ASL Citizen training clips."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from sign_language.dataset import write_json
from sign_language.extraction import fingerprint


def metadata(cache):
    cache = Path(cache)
    manifest = json.loads((cache/'manifest.json').read_text())
    path = cache/'train-index.json'
    rows = json.loads(path.read_text())
    if any(r['split'] != 'train' for r in rows):
        raise ValueError('CV may use only official training clips')
    targets = np.asarray([r['class_id'] for r in rows], dtype=np.int64)
    groups = np.asarray([r['participant_id'] for r in rows])
    if not np.array_equal(targets, np.load(cache/'train-targets.npy', mmap_mode='r')):
        raise ValueError('Training index and cached labels differ')
    return manifest, rows, targets, groups, hashlib.sha256(path.read_bytes()).hexdigest()


def make_plan(cache, destination, folds=3):
    if folds not in (3, 5):
        raise ValueError('Use 3 or 5 grouped folds')
    manifest, rows, targets, groups, digest = metadata(cache)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    partitions = []
    for i, (train, validation) in enumerate(splitter.split(np.zeros(len(rows)), targets, groups)):
        partitions.append({'fold': i, 'train': train.tolist(), 'validation': validation.tolist(),
                           'train_clips': len(train), 'validation_clips': len(validation),
                           'train_signers': len(set(groups[train])), 'validation_signers': len(set(groups[validation])),
                           'train_classes': len(set(targets[train])), 'validation_classes': len(set(targets[validation]))})
    plan = {'version': 'asl-train-signer-cv-v1', 'scope': 'official_train_only',
            'cache_id': manifest['cache_id'], 'train_index_sha256': digest,
            'sklearn': importlib.metadata.version('scikit-learn'), 'seed': 42,
            'n_splits': folds, 'partitions': partitions}
    plan['plan_id'] = fingerprint(plan)
    validate_plan(plan, manifest, targets, groups, digest)
    destination = Path(destination)
    if destination.exists() and json.loads(destination.read_text()) != plan:
        raise ValueError('Existing fold plan differs; choose another output directory')
    write_json(destination, plan)
    return plan


def validate_plan(plan, manifest, targets, groups, digest):
    content = {k: v for k, v in plan.items() if k != 'plan_id'}
    if (plan.get('plan_id') != fingerprint(content) or plan.get('scope') != 'official_train_only'
            or plan.get('cache_id') != manifest['cache_id'] or plan.get('train_index_sha256') != digest):
        raise ValueError('Fold plan identity differs from training cache')
    if len(plan['partitions']) != plan['n_splits']:
        raise ValueError('Incomplete fold plan')
    seen = np.zeros(len(targets), dtype=np.int64)
    vocabulary = set(range(len(manifest['labels'])))
    for i, item in enumerate(plan['partitions']):
        train = np.asarray(item['train'], dtype=np.int64)
        val = np.asarray(item['validation'], dtype=np.int64)
        combined = np.concatenate([train, val])
        if (item['fold'] != i or not len(train) or not len(val)
                or not np.array_equal(np.sort(combined), np.arange(len(targets)))):
            raise ValueError('Each fold must partition all training rows exactly once')
        if set(groups[train]) & set(groups[val]):
            raise ValueError('Signer leakage in fold')
        if set(targets[train]) != vocabulary:
            raise ValueError('Fold training data does not cover all labels')
        seen[val] += 1
    if not (seen == 1).all():
        raise ValueError('Each clip must be held out in exactly one fold')


def load_fold(cache, fold_file, fold):
    manifest, rows, targets, groups, digest = metadata(cache)
    plan = json.loads(Path(fold_file).read_text())
    validate_plan(plan, manifest, targets, groups, digest)
    if fold is None or not 0 <= fold < plan['n_splits']:
        raise ValueError('Choose an existing zero-based fold index')
    part = plan['partitions'][fold]
    return part['train'], part['validation'], {
        'mode': 'grouped_cv_official_train_only', 'plan_id': plan['plan_id'], 'fold': fold,
        'train_clips': len(part['train']), 'validation_clips': len(part['validation']),
        'train_signers': len(set(groups[part['train']])),
        'validation_signers': len(set(groups[part['validation']])),
        'validation_classes': len(set(targets[part['validation']])),
        'official_validation_used': False, 'test_evaluated': False,
    }


def summarize(root):
    root = Path(root); plan = json.loads((root/'folds.json').read_text())
    selected = []
    for fold in range(plan['n_splits']):
        path = root/f'fold-{fold+1}'
        if not (path/'status.json').exists():
            continue
        status = json.loads((path/'status.json').read_text())
        if status['status'] != 'completed':
            continue
        config = status['config']
        if config['partition'].get('plan_id') != plan['plan_id'] or config['partition']['fold'] != fold:
            raise ValueError('Run belongs to a different CV plan')
        records = [json.loads(p.read_text()) for p in path.glob('epoch-*.json')]
        best = min(records, key=lambda x: x['val']['loss'])
        selected.append({'fold': fold, 'epoch': best['epoch'], 'epochs_completed': status['epoch'],
                         'train': best['train'], 'validation': best['val'], 'partition': config['partition']})
    result = {'status': 'complete' if len(selected) == plan['n_splits'] else 'partial',
              'plan_id': plan['plan_id'], 'folds_completed': len(selected), 'n_splits': plan['n_splits'],
              'scope': 'development CV; fold holdouts also select checkpoints; not a final test estimate',
              'official_validation_used': False, 'test_evaluated': False, 'folds': selected}
    if selected:
        result['metrics'] = {key: {'mean': float(np.mean([r['validation'][key] for r in selected])),
                                  'sample_std': float(np.std([r['validation'][key] for r in selected], ddof=1)) if len(selected) > 1 else None}
                             for key in ('loss', 'top1', 'top5', 'macro_f1', 'weighted_f1')}
    official = root/'official-refit'
    if (official/'status.json').exists():
        s = json.loads((official/'status.json').read_text())
        if s['status'] == 'completed':
            rows = [json.loads(p.read_text()) for p in official.glob('epoch-*.json')]
            r = min(rows, key=lambda x: x['val']['loss'])
            result['official_refit'] = {'epoch': r['epoch'], 'train': r['train'], 'validation': r['val']}
            result['official_validation_used_for_refit'] = True
    write_json(root/'summary.json', result)
    return result
