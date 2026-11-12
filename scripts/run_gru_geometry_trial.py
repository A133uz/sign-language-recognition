"""Fixed paired five-class trial; official test data and live data stay unused."""
import json
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sign_language.five_sign_pilot import POLICY, early_stop_reason, select_epoch, sha256, write_json
from sign_language.gru_mean_pool_model import SignMeanPoolGRU
from sign_language.gru_geometry_model import (FEATURES, SignGeometryMeanPoolGRU,
                                            geometry_batch, fit_geometry_statistics)
from sign_language.training import LandmarkDataset, atomic_checkpoint, cuda_check, epoch_pass
from sign_language.regularization import PROFILES
from sign_language.preprocessing import CONFIG

OUTPUT = ROOT / 'models/asl-gru-five-signs-geometry-v1'
CACHE = ROOT / 'data/training/asl-five-signs-128-v1'


def read(path):
    return json.loads(path.read_text())


def collect_geometry(split):
    dataset = LandmarkDataset(CACHE, split)
    return np.concatenate([geometry_batch(frames, lengths)
                           for frames, lengths, _ in DataLoader(dataset, batch_size=16)])


def evaluate(model, split, names, *, zero_geometry=False):
    """Recorded per-row five-class decisions; clean inputs, no augmentation."""
    index = read(CACHE / f'{split}-index.json')
    matrix = np.zeros((len(names), len(names)), np.int64)
    predictions = []
    loss = 0.
    model.eval()
    with torch.inference_mode():
        for frames, lengths, targets in DataLoader(LandmarkDataset(CACHE, split), batch_size=16):
            frames = frames.cuda()
            with torch.autocast('cuda', dtype=torch.float16):
                logits = (model(frames, lengths, zero_geometry=True) if zero_geometry
                          else model(frames, lengths))
            loss += float(torch.nn.functional.cross_entropy(logits.float(), targets.cuda(), reduction='sum'))
            for target, probabilities in zip(targets.tolist(), logits.float().softmax(1).cpu().tolist(), strict=True):
                predicted = int(np.argmax(probabilities))
                row = index[len(predictions)]
                matrix[target, predicted] += 1
                predictions.append({**row, 'expected': names[target], 'prediction': names[predicted],
                                    'scores': probabilities, 'correct': target == predicted})
    tp = matrix.diagonal().astype(float)
    support = matrix.sum(1)
    precision = np.divide(tp, matrix.sum(0), out=np.zeros(len(names)), where=matrix.sum(0) > 0)
    recall = np.divide(tp, support, out=np.zeros(len(names)), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(names)), where=precision + recall > 0)
    return {'samples': len(predictions), 'correct': int(tp.sum()), 'top1': float(tp.sum() / len(predictions)),
            'loss': loss / len(predictions), 'macro_f1': float(f1.mean()), 'labels': names,
            'confusion': matrix.tolist(), 'predictions': predictions,
            'per_class': {name: {'correct': int(tp[i]), 'support': int(support[i]),
                                 'recall': float(recall[i]), 'f1': float(f1[i])} for i, name in enumerate(names)}}


def train(name, manifest, statistics):
    output = OUTPUT / name
    output.mkdir()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model_class = SignMeanPoolGRU if name == 'control' else SignGeometryMeanPoolGRU
    args = {'input_size': 254, 'hidden_size': 256, 'layers': 2, 'classes': 5, 'dropout': .5}
    model = model_class(**args)
    if name == 'geometry':
        model.set_geometry_statistics(statistics)
    model = model.cuda()
    config = {'architecture': 'gru-mean-pool-v1' if name == 'control' else 'gru-mean-pool-geometry-v1',
              'model': args, 'preprocessing': CONFIG, 'cache_id': manifest['cache_id'],
              'seed': 42, 'epochs': 28, 'batch_size': 4, 'loader_workers': 2,
              'optimizer': 'AdamW', 'learning_rate': .001, 'weight_decay': .001,
              'regularization': PROFILES['regularized-v1'], 'cudnn_enabled': False, 'amp': True,
              'selection': POLICY, 'official_test_read': False,
              'geometry_features': list(FEATURES) if name == 'geometry' else [],
              'geometry_statistics': statistics if name == 'geometry' else None}
    source = ROOT / 'src/sign_language'
    config['source_sha256'] = {n: sha256(source / n) for n in
        ('gru_model.py', 'gru_mean_pool_model.py', 'gru_geometry_model.py', 'hand_articulation.py',
         'five_sign_pilot.py', 'training.py', 'regularization.py', 'preprocessing.py')}
    write_json(output / 'config.json', config)
    write_json(output / 'labels.json', manifest['labels'])
    generator = torch.Generator().manual_seed(42)
    loaders = {s: DataLoader(LandmarkDataset(CACHE, s), batch_size=4 if s == 'train' else 16,
                            shuffle=s == 'train', num_workers=2, persistent_workers=True, pin_memory=True,
                            generator=generator if s == 'train' else None) for s in ('train', 'val')}
    clean_loader = DataLoader(LandmarkDataset(CACHE, 'train'), batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    scaler = torch.amp.GradScaler('cuda')
    criterion = torch.nn.CrossEntropyLoss()
    records = []
    best_loss, best_accuracy = float('inf'), -1.
    loss_stale = accuracy_stale = 0
    stop = 'epoch_cap'
    started = time.monotonic()
    try:
        for epoch in range(1, 29):
            write_json(output / 'status.json', {'status': 'training', 'epoch': epoch})
            training = epoch_pass(model, loaders['train'], criterion, 'cuda', optimizer, scaler, augment=True)
            clean = epoch_pass(model, clean_loader, criterion, 'cuda', scaler=scaler)
            validation = epoch_pass(model, loaders['val'], criterion, 'cuda', scaler=scaler)
            for result in (training, clean, validation):
                result.pop('top5', None)
            record = {'epoch': epoch, 'train': training, 'train_clean': clean, 'val': validation,
                      'accuracy_gap': clean['top1'] - validation['top1'], 'loss_gap': validation['loss'] - clean['loss']}
            records.append(record)
            atomic_checkpoint(output / f'epoch-{epoch:03d}.pt',
                              {'config': config, 'model': model.state_dict(), 'labels': manifest['labels'],
                               'epoch': epoch, 'metrics': record})
            write_json(output / f'epoch-{epoch:03d}.json', record)
            if validation['loss'] < best_loss - POLICY['loss_min_delta']:
                best_loss = validation['loss']
                loss_stale = 0
            else:
                loss_stale += 1
            accuracy_stale = 0 if validation['top1'] > best_accuracy + 1.e-12 else accuracy_stale + 1
            best_accuracy = max(best_accuracy, validation['top1'])
            print(f'{name} {epoch}: train {clean["top1"]:.1%}, val {validation["top1"]:.1%}, CE {validation["loss"]:.4f}', flush=True)
            reason = early_stop_reason(records, loss_stale, accuracy_stale)
            if reason:
                stop = reason
                break
        chosen, selection = select_epoch(records)
        shutil.copy2(output / f'epoch-{chosen["epoch"]:03d}.pt', output / 'best.pt')
        state = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        names = [g for g, _ in sorted(manifest['labels'].items(), key=lambda item: item[1])]
        report = {'selection': selection, 'epochs_completed': len(records), 'stop_reason': stop,
                  'parameters': sum(p.numel() for p in model.parameters()), 'seconds': time.monotonic() - started,
                  'checkpoint_sha256': sha256(output / 'best.pt'), 'epoch_metrics': records,
                  'train': evaluate(model, 'train', names), 'validation': evaluate(model, 'val', names)}
        if name == 'geometry':
            report['validation_zero_geometry_ablation'] = evaluate(model, 'val', names, zero_geometry=True)
            report['geometry_head_weights'] = model.classifier.weight[:, 256:].detach().cpu().tolist()
        write_json(output / 'report.json', report)
        write_json(output / 'status.json', {'status': 'complete', 'selected_epoch': chosen['epoch'], 'test_evaluated': False})
        return report
    finally:
        for loader in loaders.values():
            iterator = getattr(loader, '_iterator', None)
            if iterator is not None:
                iterator._shutdown_workers()
        del model
        torch.cuda.empty_cache()


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve prior runs; use a new version')
    torch.set_num_threads(2)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    gpu = cuda_check()
    manifest = read(CACHE / 'manifest.json')
    sources = {f'{s}-{k}.{ext}': sha256(CACHE / f'{s}-{k}.{ext}') for s in ('train', 'val')
               for k, ext in [('features', 'npy'), ('lengths', 'npy'), ('targets', 'npy'), ('index', 'json')]}
    if any(v != manifest['files_sha256'][k] for k, v in sources.items()):
        raise ValueError('Cache checksum differs')
    if manifest['identity']['preprocessing'] != CONFIG:
        raise ValueError('Preprocessing differs')
    protected = {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / 'models').rglob('best.pt')}
    rows = {s: read(CACHE / f'{s}-index.json') for s in ('train', 'val')}
    if {r['participant_id'] for r in rows['train']} & {r['participant_id'] for r in rows['val']}:
        raise ValueError('Signer leakage')
    for split in rows:
        targets = np.load(CACHE / f'{split}-targets.npy')
        if targets.tolist() != [manifest['labels'][r['gloss']] for r in rows[split]]:
            raise ValueError('Index and targets disagree')
    raw = {s: collect_geometry(s) for s in ('train', 'val')}
    statistics = fit_geometry_statistics(raw['train'])
    OUTPUT.mkdir()
    specification = {'status': 'prespecified', 'gpu': gpu, 'features': list(FEATURES),
                     'geometry_statistics': statistics, 'fit_on': 'official training only; all 75 rows',
                     'extra_trainable_parameters': 85, 'test_read': False, 'webcam_opened': False,
                     'protocol': 'One fixed seed; same settings and validation-only policy as prior mean-pool GRU; no search',
                     'measurement_policy': 'Recompute summaries from the actual augmented frames in every training forward; missing values use train medians',
                     'branch': 'Concatenate 17 standardized scalars with 256 dropped-out pooled outputs; one jointly trained linear head',
                     'initialization': 'Same GRU and original head weights, 85 extra weights zero; RNG position preserved',
                     'missing_geometry_values': {s: int(np.isnan(v).sum()) for s, v in raw.items()},
                     'cache_sha256': sources, 'protected_weights_sha256': protected,
                     'defaults_changed': False, 'fresh_independent_benchmark': False}
    write_json(OUTPUT / 'experiment.json', specification)
    snapshot = OUTPUT / 'source_snapshot'
    snapshot.mkdir()
    for name in ('gru_geometry_model.py', 'gru_model.py', 'gru_mean_pool_model.py', 'hand_articulation.py',
                 'training.py', 'regularization.py', 'preprocessing.py', 'five_sign_pilot.py'):
        shutil.copy2(ROOT / 'src/sign_language' / name, snapshot / name)
    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
    report = {'specification': specification}
    try:
        for name in ('control', 'geometry'):
            report[name] = train(name, manifest, statistics)
        report['main_weights_unchanged'] = all(sha256(ROOT / n) == h for n, h in protected.items())
        report['train_val_cache_unchanged'] = all(sha256(CACHE / n) == h for n, h in sources.items())
        if not report['main_weights_unchanged'] or not report['train_val_cache_unchanged']:
            raise AssertionError('Protected artifacts changed')
        before = report['control']['validation']['predictions']
        after = report['geometry']['validation']['predictions']
        if [r['video_file'] for r in before] != [r['video_file'] for r in after]:
            raise AssertionError('Paired rows differ')
        report['paired_validation_changes'] = {
            'improved': sum(not a['correct'] and b['correct'] for a, b in zip(before, after, strict=True)),
            'regressed': sum(a['correct'] and not b['correct'] for a, b in zip(before, after, strict=True))}
        historical = ROOT / 'models/asl-gru-five-signs-mean-pool-v1/best.pt'
        old = torch.load(historical, map_location='cpu', weights_only=True)
        repeated = torch.load(OUTPUT / 'control/best.pt', map_location='cpu', weights_only=True)
        report['control_matches_historical'] = {
            'epoch_equal': old['epoch'] == repeated['epoch'],
            'all_model_tensors_equal': all(torch.equal(v, repeated['model'][k]) for k, v in old['model'].items())}
        report['status'] = 'complete'
        write_json(OUTPUT / 'report.json', report)
        write_json(ROOT / 'docs/assets/gru-geometry-trial-2026-11-12.json', report)
        print('GEOMETRY_TRIAL_COMPLETE', json.dumps({n: {'epoch': report[n]['selection']['selected_epoch'],
              'validation': report[n]['validation']['correct'], 'samples': 19} for n in ('control', 'geometry')}), flush=True)
    except BaseException as exc:
        write_json(OUTPUT / 'failure.json', {'error': str(exc), 'type': type(exc).__name__})
        raise


if __name__ == '__main__':
    main()
