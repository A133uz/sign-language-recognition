"""Frozen-checkpoint evaluation; no optimizer, augmentation or model selection."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from sign_language.dataset import EXPECTED_COUNTS, write_json
from sign_language.extraction import load_rows, output_lock, shard_plan
from sign_language.gru_model import SignGRU
from sign_language.preprocessing import CONFIG
from sign_language.training import LandmarkDataset


def classification_metrics(confusion):
    cm = np.asarray(confusion)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or (cm < 0).any() or not cm.sum():
        raise ValueError('Expected a nonempty square confusion matrix of counts')
    tp = np.diag(cm).astype(float); support = cm.sum(1); predicted = cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(tp), where=(precision+recall) > 0)
    return {'top1': float(tp.sum()/cm.sum()), 'macro_precision': float(precision.mean()),
            'macro_recall': float(recall.mean()), 'macro_f1': float(f1.mean()),
            'weighted_f1': float((f1*support).sum()/support.sum())}, precision, recall, f1


def validate_cache(cache, dataset, checkpoint, split):
    manifest = json.loads((cache/'manifest.json').read_text())
    status = json.loads((cache/'status.json').read_text())
    config = checkpoint['config']
    if status['status'] != 'complete' or manifest['cache_id'] != config['cache_id'] or status['cache_id'] != manifest['cache_id']:
        raise ValueError('Checkpoint and complete feature-cache identity must match')
    if checkpoint['labels'] != manifest['labels'] or config['preprocessing'] != manifest['identity']['preprocessing']:
        raise ValueError('Checkpoint label mapping or preprocessing differs')
    if config['preprocessing'] != CONFIG:
        raise ValueError('Current preprocessor configuration differs from checkpoint')
    source = hashlib.sha256(Path(__file__).with_name('preprocessing.py').read_bytes()).hexdigest()
    if source != manifest['identity']['preprocessing_sha256'] or source != config['source_sha256']['preprocessing.py']:
        raise ValueError('Preprocessing source hash differs')
    model_source = hashlib.sha256(Path(__file__).with_name('gru_model.py').read_bytes()).hexdigest()
    if model_source != config['source_sha256']['gru_model.py']:
        raise ValueError('Model implementation differs from selected checkpoint')
    rows, labels, _ = load_rows(dataset)
    expected = [row for _, group in shard_plan(rows, (split,), 100, None) for row in group]
    index = json.loads((cache/f'{split}-index.json').read_text())
    if labels != checkpoint['labels'] or index != expected or len(index) != EXPECTED_COUNTS[split]:
        raise ValueError('Cached index does not match the official split records')
    arrays = [np.load(cache/f'{split}-{name}.npy', mmap_mode='r') for name in ('features', 'lengths', 'targets')]
    features, lengths, targets = arrays
    if features.shape != (len(index), 128, 254) or len(lengths) != len(index) or len(targets) != len(index):
        raise ValueError('Prepared array dimensions differ')
    if not np.array_equal(targets, [r['class_id'] for r in index]) or (lengths < 1).any() or (lengths > 128).any():
        raise ValueError('Invalid targets or sequence lengths')
    offset = 0
    for name, group in shard_plan(rows, (split,), 100, None):
        marker = json.loads((cache/f'{name}.prepared.json').read_text())
        end = offset+len(group)
        digest = hashlib.sha256(features[offset:end].tobytes()+lengths[offset:end].tobytes()+targets[offset:end].tobytes()).hexdigest()
        if (marker['cache_id'] != manifest['cache_id'] or marker['start'] != offset
                or marker['count'] != len(group) or marker['sha256'] != digest):
            raise ValueError(f'Prepared feature integrity failure: {name}')
        offset = end
        if offset % 5000 == 0:
            print(f'Integrity verified: {offset}/{len(index)} {split} clips', flush=True)
    return index


def evaluate(checkpoint_path, cache, dataset, output, *, split='test', batch_size=64, workers=2):
    checkpoint_path, cache, dataset, output = map(Path, (checkpoint_path, cache, dataset, output))
    if split not in ('val', 'test') or batch_size < 1 or workers not in (0, 1, 2, 3, 4):
        raise ValueError('Invalid evaluation split, batch size or workers')
    with output_lock(output.parent/'locks', output):
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError('Evaluation output is nonempty; choose a new output directory')
        began = time.monotonic(); checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        state = {'status': 'verifying', 'split': split, 'checkpoint': str(checkpoint_path.resolve()),
                 'checkpoint_sha256': checkpoint_hash, 'started_utc': datetime.now(timezone.utc).isoformat(),
                 'training_performed': False, 'augmentation': False}
        write_json(output/'status.json', state)
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if checkpoint['config'] != json.loads((checkpoint_path.parent/'config.json').read_text()):
                raise ValueError('Checkpoint configuration differs from its run record')
            if not all(torch.isfinite(t).all() for t in checkpoint['model'].values()):
                raise ValueError('Checkpoint has nonfinite weights')
            index = validate_cache(cache, dataset, checkpoint, split)
            verification_seconds = time.monotonic()-began
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; refusing silent CPU fallback')
            torch.backends.cudnn.enabled = checkpoint['config']['cudnn_enabled']
            torch.set_num_threads(2)
            model = SignGRU(**checkpoint['config']['model']).cuda().eval()
            model.load_state_dict(checkpoint['model'])
            labels = checkpoint['labels']; names = [None]*len(labels)
            for gloss, class_id in labels.items():
                names[class_id] = gloss
            loader = DataLoader(LandmarkDataset(cache, split), batch_size=batch_size, shuffle=False,
                                num_workers=workers, pin_memory=True, persistent_workers=False,
                                **({'prefetch_factor': 2} if workers else {}))
            confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
            person_counts = {}; processed = correct5 = 0; loss_sum = 0.
            criterion = torch.nn.CrossEntropyLoss(reduction='sum')
            state.update(status='evaluating', checkpoint_epoch=checkpoint['epoch'], processed=0, total=len(index))
            write_json(output/'status.json', state)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); inference_began = time.monotonic()
            with torch.inference_mode(), (output/'predictions.jsonl').open('w', encoding='utf-8') as stream:
                for step, (frames, lengths, targets) in enumerate(loader, 1):
                    frames, gpu_targets = frames.cuda(non_blocking=True), targets.cuda(non_blocking=True)
                    with torch.autocast('cuda', dtype=torch.float16, enabled=checkpoint['config']['amp']):
                        logits = model(frames, lengths)
                        loss = criterion(logits, gpu_targets)
                    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite evaluation output')
                    predicted = logits.argmax(1)
                    top_ids = logits.topk(5, dim=1).indices
                    top_scores = logits.float().softmax(1).gather(1, top_ids).cpu().numpy()
                    predicted, top_ids, actual = predicted.cpu().numpy(), top_ids.cpu().numpy(), targets.numpy()
                    np.add.at(confusion, (actual, predicted), 1)
                    loss_sum += loss.item()
                    for i, target in enumerate(actual):
                        row = index[processed+i]; pred = int(predicted[i]); target = int(target)
                        hit5 = target in top_ids[i]; correct5 += int(hit5)
                        stats = person_counts.setdefault(row['participant_id'], {'samples': 0, 'top1_correct': 0, 'top5_correct': 0})
                        stats['samples'] += 1; stats['top1_correct'] += int(pred == target); stats['top5_correct'] += int(hit5)
                        stream.write(json.dumps({**row, 'predicted_id': pred, 'predicted_gloss': names[pred],
                            'top5_ids': top_ids[i].tolist(), 'top5_glosses': [names[j] for j in top_ids[i]],
                            'top5_softmax_scores': [float(x) for x in top_scores[i]], 'top1_correct': pred == target,
                            'top5_correct': bool(hit5)}, ensure_ascii=False)+'\n')
                    processed += len(actual)
                    if step % 100 == 0 or processed == len(index):
                        state['processed'] = processed; write_json(output/'status.json', state)
                        print(f'Evaluated {processed}/{len(index)} clips; loss={loss_sum/processed:.4f}', flush=True)
            torch.cuda.synchronize(); inference_seconds = time.monotonic()-inference_began
            if processed != len(index) or confusion.sum() != len(index):
                raise ValueError('Evaluation did not cover the full split')
            metrics, precision, recall, f1 = classification_metrics(confusion)
            metrics.update(top5=correct5/processed, loss=loss_sum/processed, samples=processed)
            np.savez_compressed(output/'confusion.npz', counts=confusion, labels=np.asarray(names))
            with (output/'per-class.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream); writer.writerow(['class_id', 'gloss', 'support', 'precision', 'recall', 'f1'])
                for i, name in enumerate(names):
                    writer.writerow([i, name, int(confusion[i].sum()), precision[i], recall[i], f1[i]])
            for stats in person_counts.values():
                stats.update(top1=stats['top1_correct']/stats['samples'], top5=stats['top5_correct']/stats['samples'])
            write_json(output/'per-signer.json', person_counts)
            after_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            if after_hash != checkpoint_hash:
                raise RuntimeError('Checkpoint file changed during evaluation')
            state.update(status='complete', metrics=metrics, cache_id=checkpoint['config']['cache_id'],
                         checkpoint_epoch=checkpoint['epoch'], checkpoint_unchanged=True,
                         selection=checkpoint['config']['selection'],
                         validation_at_selection=checkpoint['metrics']['val'],
                         classes=len(labels), signers=len(person_counts),
                         verification_seconds=verification_seconds, inference_seconds=inference_seconds,
                         elapsed_seconds=time.monotonic()-began, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                         gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__),
                         cudnn_enabled=torch.backends.cudnn.enabled, amp=checkpoint['config']['amp'],
                         finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(output/'report.json', state); write_json(output/'status.json', state)
            return state
        except BaseException as error:
            state.update(status='failed', error=f'{type(error).__name__}: {error}')
            write_json(output/'status.json', state)
            raise
