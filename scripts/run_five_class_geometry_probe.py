"""Fit one fixed geometry-only baseline; no test reads, new captures or main-model changes."""
import json
from pathlib import Path
import pickle
import platform
import shutil
import sys
import time
import warnings

import torch  # Initialize Windows native libraries first; the probe fits on CPU.
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sign_language.five_sign_pilot import sha256, write_json
from sign_language.geometry_classifier import FEATURES, LABELS, geometry_vector, evaluate, grouped_oof, make_probe
from sign_language.hand_articulation import summarize_articulation
from sign_language.preprocessing import CONFIG


def read(path):
    return json.loads(path.read_text())


def decisions(rows, probabilities):
    return [{**{key: row.get(key) for key in ('video_file', 'participant_id', 'capture_id')},
             'expected': row.get('gloss', row.get('expected')),
             'predicted': LABELS[int(p.argmax())], 'scores': p.tolist(),
             'correct': LABELS[int(p.argmax())] == row.get('gloss', row.get('expected'))}
            for row, p in zip(rows, probabilities, strict=True)]


def main():
    output = ROOT / 'models/asl-five-class-geometry-only-v1'
    if output.exists():
        raise FileExistsError('Preserve previous runs; use a new version')
    cache = ROOT / 'data/training/asl-five-signs-128-v1'
    manifest = read(cache / 'manifest.json')
    if manifest['labels'] != {name: i for i, name in enumerate(LABELS)} or manifest['identity']['preprocessing'] != CONFIG:
        raise ValueError('Unexpected cache schema')
    if sha256(ROOT / 'src/sign_language/preprocessing.py') != manifest['identity']['preprocessing_sha256']:
        raise ValueError('Preprocessing source changed')
    sources = {f'{s}-{kind}.{ext}': sha256(cache / f'{s}-{kind}.{ext}') for s in ('train', 'val')
               for kind, ext in [('features', 'npy'), ('lengths', 'npy'), ('targets', 'npy'), ('index', 'json')]}
    if any(h != manifest['files_sha256'][name] for name, h in sources.items()):
        raise ValueError('Source cache checksum differs')
    protected = {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / 'models').rglob('*')
                 if p.is_file() and (p.name == 'best.pt' or p.suffix == '.pkl')}
    data = {}
    for split in ('train', 'val'):
        rows = read(cache / f'{split}-index.json')
        f = np.load(cache / f'{split}-features.npy', mmap_mode='r')
        lengths = np.load(cache / f'{split}-lengths.npy')
        y = np.load(cache / f'{split}-targets.npy')
        if y.tolist() != [manifest['labels'][row['gloss']] for row in rows]:
            raise ValueError('Index/target mismatch')
        summaries = [summarize_articulation(f[i, :int(lengths[i])]) for i in range(len(rows))]
        data[split] = {'rows': rows, 'articulation': summaries,
                       'x': np.stack([geometry_vector(s) for s in summaries]), 'y': y,
                       'groups': np.asarray([r['participant_id'] for r in rows])}
    if set(data['train']['groups']) & set(data['val']['groups']):
        raise ValueError('Signer leakage')
    output.mkdir()
    configuration = {'experiment': 'fixed_five_class_geometry_only_v1', 'diagnostic_only': True,
        'labels': list(LABELS), 'features': list(FEATURES), 'cache_id': manifest['cache_id'],
        'model': 'Median imputation + optional missingness indicators + StandardScaler + L2 multinomial logistic regression',
        'parameters': {'C': 1., 'solver': 'lbfgs', 'max_iter': 2000, 'random_state': 42},
        'fit_split': 'official training only, all 75 rows', 'augmentation': None,
        'selection': 'No search; fixed 3-fold training-only signer-grouped OOF, then fit all training',
        'test_read': False, 'webcam_opened': False, 'main_defaults_changed': False,
        'live_scope': 'Retrospective, design-informed NO/HELLO scalar summaries, never fitting/tuning data',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'source_sha256': {n: sha256(ROOT / 'src/sign_language' / n) for n in
                          ('geometry_classifier.py', 'yes_no_geometry.py', 'hand_articulation.py')},
        'train_val_cache_sha256': sources}
    write_json(output / 'config.json', configuration)
    train, validation = data['train'], data['val']
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        oof, folds = grouped_oof(train['x'], train['y'], train['groups'])
        model = make_probe().fit(train['x'], train['y'])
    if not np.array_equal(model.classes_, np.arange(len(LABELS))):
        raise AssertionError('Label order changed')
    with (output / 'model.pkl').open('wb') as stream:
        pickle.dump(model, stream)
    report = {'status': 'complete', 'config': configuration,
        'counts': {s: {g: int((d['y'] == i).sum()) for i, g in enumerate(LABELS)} for s, d in data.items()},
        'unique_signers': {s: len(set(d['groups'])) for s, d in data.items()},
        'missing_values': {s: int(np.isnan(d['x']).sum()) for s, d in data.items()},
        'insufficient_clips': {s: sum(a['insufficient'] for a in d['articulation']) for s, d in data.items()},
        'train_fit': evaluate(train['y'], model.predict_proba(train['x'])),
        'train_grouped_oof': evaluate(train['y'], oof), 'folds': folds,
        'validation': evaluate(validation['y'], model.predict_proba(validation['x'])),
        'predictions': {'train_fit': decisions(train['rows'], model.predict_proba(train['x'])),
                        'train_grouped_oof': decisions(train['rows'], oof),
                        'validation': decisions(validation['rows'], model.predict_proba(validation['x']))},
        'fit_seconds': time.perf_counter() - started, 'model_sha256': sha256(output / 'model.pkl'),
        'coefficient_count_including_bias': int(model.named_steps['classifier'].coef_.size + model.named_steps['classifier'].intercept_.size),
        'solver_iterations': model.named_steps['classifier'].n_iter_.tolist()}
    # Freeze model and configuration before loading the already inspected live summaries.
    live_path = ROOT / 'docs/assets/gru-geometry-guided-2026-11-12.json'
    live = read(live_path)['live']
    if live['status'] != 'complete' or live.get('user_completion_confirmed') is not True:
        raise ValueError('Live prompts not confirmed')
    trials = live['trials']
    if [r['expected'] for r in trials] != ['NO', 'HELLO'] * 3:
        raise ValueError('Unexpected retrospective trial set')
    x = np.stack([geometry_vector(r['articulation']) for r in trials])
    y = np.asarray([LABELS.index(r['expected']) for r in trials])
    probability = model.predict_proba(x)
    report['retrospective_live'] = {'metrics': evaluate(y, probability), 'predictions': decisions(trials, probability),
        'source_sha256': sha256(live_path), 'used_for_fitting_or_parameter_selection': False,
        'limits': 'Previously inspected 3 NO and 3 HELLO attempts from one user; design-informed, not a fresh five-class webcam benchmark.',
        'matched_sequence_model_results': {key: {'correct': sum(t['variants'][key]['correct'] for t in trials),
            'per_class_correct': {name: sum(t['variants'][key]['correct'] for t in trials if t['expected'] == name) for name in ('NO', 'HELLO')}}
            for key in ('original', 'mean_pool', 'geometry_off')}}
    prior = read(ROOT / 'docs/assets/gru-geometry-trial-2026-11-12.json')
    report['same_validation_sequence_model_reference'] = {n: {k: prior[n]['validation'][k]
        for k in ('samples', 'correct', 'top1', 'macro_f1', 'per_class')} for n in ('control', 'geometry')}
    if prior['specification']['cache_sha256'] != sources:
        raise ValueError('Historical comparison cache differs')
    report['protected_weights_and_probes_unchanged'] = all(sha256(ROOT / name) == h for name, h in protected.items())
    report['train_val_cache_unchanged'] = all(sha256(cache / name) == h for name, h in sources.items())
    if not report['protected_weights_and_probes_unchanged'] or not report['train_val_cache_unchanged']:
        raise AssertionError('Protected artifact changed')
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    for name in configuration['source_sha256']:
        shutil.copy2(ROOT / 'src/sign_language' / name, snapshot / name)
    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
    write_json(output / 'report.json', report)
    write_json(ROOT / 'docs/assets/five-class-geometry-only-2026-11-12.json', report)
    print(json.dumps({k: report[k] for k in ('train_fit', 'train_grouped_oof', 'validation')}, indent=2))
    print('RETROSPECTIVE_LIVE', json.dumps(report['retrospective_live']))


if __name__ == '__main__':
    main()
