"""Paired 15-class geometry GRUs: temporal augmentation versus matched control."""
import json
import hashlib
from pathlib import Path
import random
import shutil
import sys
import time

import torch
import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sign_language.expanded_pilot import SIGNS, prepare_subset, verify_parent_chunks
from sign_language.five_sign_pilot import POLICY, early_stop_reason, select_epoch, sha256, write_json
from sign_language.gru_geometry_model import FEATURES, SignGeometryMeanPoolGRU, geometry_batch, fit_geometry_statistics
from sign_language.preprocessing import CONFIG
from sign_language.regularization import PROFILES
from sign_language.temporal_augmentation import CONFIG as TEMPORAL_CONFIG, TemporalCollator
from sign_language.training import LandmarkDataset, atomic_checkpoint, cuda_check, epoch_pass

PARENT = ROOT / 'data/training/asl-gru-128-v1'
CACHE = ROOT / 'data/training/asl-fifteen-signs-128-v1'
OUTPUT = ROOT / 'models/asl-gru-fifteen-signs-temporal-v1'


def read(path):
    return json.loads(path.read_text())


def evaluate(model, split, names):
    rows = read(CACHE / f'{split}-index.json')
    matrix = np.zeros((len(names), len(names)), np.int64)
    decisions = []
    loss = 0.
    top5 = 0
    model.eval()
    with torch.inference_mode():
        for frames, lengths, targets in DataLoader(LandmarkDataset(CACHE, split), batch_size=16):
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(frames.cuda(), lengths).float()
            loss += float(torch.nn.functional.cross_entropy(logits, targets.cuda(), reduction='sum'))
            probabilities = logits.softmax(1).cpu().numpy()
            for target, p in zip(targets.tolist(), probabilities, strict=True):
                predicted = int(p.argmax())
                matrix[target, predicted] += 1
                order = np.argsort(-p)
                top5 += int(target in order[:5])
                decisions.append({**rows[len(decisions)], 'expected': names[target], 'prediction': names[predicted],
                                  'scores': p.tolist(), 'correct': target == predicted})
    tp = matrix.diagonal().astype(float)
    support = matrix.sum(1)
    precision = np.divide(tp, matrix.sum(0), out=np.zeros(len(names)), where=matrix.sum(0) > 0)
    recall = np.divide(tp, support, out=np.zeros(len(names)), where=support > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(len(names)), where=precision+recall > 0)
    return {'samples': len(rows), 'correct': int(tp.sum()), 'top1': float(tp.sum()/len(rows)),
            'top5': top5/len(rows), 'loss': loss/len(rows), 'macro_f1': float(f1.mean()),
            'labels': names, 'confusion': matrix.tolist(), 'predictions': decisions,
            'per_class': {g: {'support': int(support[i]), 'correct': int(tp[i]),
                              'recall': float(recall[i]), 'f1': float(f1[i])} for i, g in enumerate(names)}}


def train_variant(name, manifest, statistics, *, output_root=None,
                  model_class=SignGeometryMeanPoolGRU, geometry_features=FEATURES,
                  architecture='gru-mean-pool-geometry-v1', additional_config=None,
                  extra_sources=(), capture_validation=False):
    output = (OUTPUT if output_root is None else output_root) / name
    output.mkdir()
    temporal = name == 'temporal'
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model_args = {'input_size': 254, 'hidden_size': 256, 'layers': 2, 'classes': len(SIGNS), 'dropout': .5}
    model = model_class(**model_args)
    model.set_geometry_statistics(statistics)
    initial_hash = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.state_dict().values())).hexdigest()
    model = model.cuda()
    config = {'architecture': architecture, 'model': model_args, 'preprocessing': CONFIG,
              'cache_id': manifest['cache_id'], 'seed': 42, 'batch_size': 4, 'loader_workers': 2,
              'epochs': 28, 'optimizer': 'AdamW', 'learning_rate': .001, 'weight_decay': .001,
              'regularization': PROFILES['regularized-v1'], 'temporal_augmentation': TEMPORAL_CONFIG if temporal else None,
              'geometry_features': list(geometry_features), 'geometry_statistics': statistics,
              'amp': True, 'cudnn_enabled': False, 'selection': POLICY, 'official_test_read': False,
              'augmentation_order': 'train-only temporal collate -> existing spatial perturbation -> GRU and recomputed 17 geometry summaries',
              'source_sha256': {n: sha256(ROOT / 'src/sign_language' / n) for n in
                  ('gru_model.py', 'gru_mean_pool_model.py', 'gru_geometry_model.py', 'hand_articulation.py',
                   'preprocessing.py', 'training.py', 'regularization.py', 'temporal_augmentation.py', 'five_sign_pilot.py')}}
    config.update(additional_config or {})
    config['source_sha256'].update({n:sha256(ROOT/'src/sign_language'/n) for n in extra_sources})
    write_json(output / 'config.json', config)
    write_json(output / 'labels.json', manifest['labels'])
    generator = torch.Generator().manual_seed(42)
    loaders = {s: DataLoader(LandmarkDataset(CACHE, s), batch_size=4 if s == 'train' else 16,
                            shuffle=s == 'train', num_workers=2, persistent_workers=True, pin_memory=True,
                            generator=generator if s == 'train' else None,
                            collate_fn=TemporalCollator(enabled=temporal and s == 'train')) for s in ('train', 'val')}
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
            write_json(output / 'status.json', {'status': 'training', 'epoch': epoch, 'epoch_cap': 28})
            training = epoch_pass(model, loaders['train'], criterion, 'cuda', optimizer, scaler, augment=True)
            clean = epoch_pass(model, clean_loader, criterion, 'cuda', scaler=scaler)
            validation = epoch_pass(model, loaders['val'], criterion, 'cuda', scaler=scaler)
            record = {'epoch': epoch, 'train': training, 'train_clean': clean, 'val': validation,
                      'accuracy_gap': clean['top1']-validation['top1'], 'loss_gap': validation['loss']-clean['loss']}
            if capture_validation:
                record['validation_details'] = evaluate(model,'val',list(SIGNS))
            records.append(record)
            atomic_checkpoint(output / f'epoch-{epoch:03d}.pt', {'config': config, 'model': model.state_dict(),
                              'labels': manifest['labels'], 'epoch': epoch, 'metrics': record})
            write_json(output / f'epoch-{epoch:03d}.json', record)
            if validation['loss'] < best_loss - POLICY['loss_min_delta']:
                best_loss = validation['loss']
                loss_stale = 0
            else:
                loss_stale += 1
            accuracy_stale = 0 if validation['top1'] > best_accuracy + 1.e-12 else accuracy_stale + 1
            best_accuracy = max(best_accuracy, validation['top1'])
            target_note = ''
            if capture_validation:
                per_class = record['validation_details']['per_class']
                target_note = f', PLEASE {per_class["PLEASE"]["correct"]}/4, MY {per_class["MY"]["correct"]}/4'
            print(f'{name} {epoch:02d}: clean train {clean["top1"]:.1%}, val {validation["top1"]:.1%}, CE {validation["loss"]:.4f}{target_note}', flush=True)
            reason = early_stop_reason(records, loss_stale, accuracy_stale)
            if reason:
                stop = reason
                break
        chosen, selection = select_epoch(records)
        shutil.copy2(output / f'epoch-{chosen["epoch"]:03d}.pt', output / 'best.pt')
        state = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        result = {'selection': selection, 'epochs_completed': len(records), 'stop_reason': stop,
                  'parameters': sum(p.numel() for p in model.parameters()), 'seconds': time.monotonic()-started,
                  'initial_model_sha256': initial_hash, 'checkpoint_sha256': sha256(output / 'best.pt'),
                  'epoch_metrics': records, 'train': evaluate(model, 'train', list(SIGNS)),
                  'validation': evaluate(model, 'val', list(SIGNS)), 'official_test_read': False}
        write_json(output / 'report.json', result)
        write_json(output / 'status.json', {'status': 'complete', 'selected_epoch': chosen['epoch'], 'test_evaluated': False})
        return result
    finally:
        for loader in loaders.values():
            iterator = getattr(loader, '_iterator', None)
            if iterator is not None:
                iterator._shutdown_workers()
        del model
        torch.cuda.empty_cache()


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve previous trial; use a new version')
    torch.set_num_threads(2)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    gpu = cuda_check()
    protected = {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / 'models').rglob('*')
                 if p.is_file() and (p.name == 'best.pt' or p.suffix == '.pkl')}
    subset = prepare_subset(PARENT, CACHE) if not CACHE.exists() else read(CACHE / 'manifest.json')
    if list(subset['labels']) != list(SIGNS) or subset['identity']['parent_cache_id'] != read(PARENT / 'manifest.json')['cache_id']:
        raise ValueError('Expanded subset differs')
    for name, digest in subset['files_sha256'].items():
        if sha256(CACHE / name) != digest:
            raise ValueError('Expanded cache checksum differs')
    verify_parent_chunks(PARENT, subset)
    training_geometry = np.concatenate([geometry_batch(frames, lengths)
        for frames, lengths, _ in DataLoader(LandmarkDataset(CACHE, 'train'), batch_size=16)])
    statistics = fit_geometry_statistics(training_geometry)
    OUTPUT.mkdir()
    specification = {'status': 'prespecified', 'gpu': gpu, 'classes': list(SIGNS), 'counts': subset['counts'],
                     'participants': subset['participants'], 'cache_id': subset['cache_id'],
                     'geometry_statistics_fit_on': 'Original official training clips only',
                     'missing_training_geometry_values': int(np.isnan(training_geometry).sum()),
                     'comparison': 'Both 15-class mean-pool geometry GRUs, from scratch; temporal sampling is the sole configured difference',
                     'cap': 28, 'selection': POLICY, 'single_seed': 42, 'search': False,
                     'source_landmarks_reextracted': False, 'test_read': False, 'webcam_opened': False,
                     'defaults_changed': False, 'five_class_models_preserved': True}
    write_json(OUTPUT / 'experiment.json', specification)
    snapshot = OUTPUT / 'source_snapshot'
    snapshot.mkdir()
    for name in ('expanded_pilot.py', 'temporal_augmentation.py', 'gru_model.py', 'gru_mean_pool_model.py',
                 'gru_geometry_model.py', 'hand_articulation.py', 'preprocessing.py', 'training.py', 'regularization.py', 'five_sign_pilot.py'):
        shutil.copy2(ROOT / 'src/sign_language' / name, snapshot / name)
    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
    report = {'specification': specification}
    try:
        for name in ('control', 'temporal'):
            report[name] = train_variant(name, subset, statistics)
        if report['control']['initial_model_sha256'] != report['temporal']['initial_model_sha256']:
            raise AssertionError('Paired initial model states differ')
        report['previous_weights_and_probes_unchanged'] = all(sha256(ROOT / name) == digest for name, digest in protected.items())
        report['subset_unchanged'] = all(sha256(CACHE / n) == h for n, h in subset['files_sha256'].items())
        report['parent_selected_chunks_unchanged'] = verify_parent_chunks(PARENT, subset)
        if not report['previous_weights_and_probes_unchanged'] or not report['subset_unchanged']:
            raise AssertionError('Protected artifacts changed')
        control = report['control']['validation']['predictions']
        temporal = report['temporal']['validation']['predictions']
        if [r['video_file'] for r in control] != [r['video_file'] for r in temporal]:
            raise AssertionError('Validation rows differ')
        report['paired_validation_changes'] = {
            'improved': sum(not a['correct'] and b['correct'] for a, b in zip(control, temporal, strict=True)),
            'regressed': sum(a['correct'] and not b['correct'] for a, b in zip(control, temporal, strict=True))}
        report['status'] = 'complete'
        write_json(OUTPUT / 'report.json', report)
        write_json(ROOT / 'docs/assets/fifteen-sign-temporal-2026-11-12.json', report)
        print('FIFTEEN_SIGN_COMPLETE', json.dumps({name: {'epoch': report[name]['selection']['selected_epoch'],
              'correct': report[name]['validation']['correct'], 'samples': report[name]['validation']['samples']}
              for name in ('control', 'temporal')}), flush=True)
    except BaseException as exc:
        write_json(OUTPUT / 'failure.json', {'type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
