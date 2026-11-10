"""Read-only checkpoint/data audit and validation-only feature perturbations."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import cv2
import numpy as np
import torch

from sign_language.dataset import write_json
from sign_language.preprocessing import normalize_sequence
from sign_language.recognition import SignRecognizer

TARGETS = ('HELLO', 'THANKYOU', 'NO')


def reflect_features(sequence):
    """Feature-space horizontal reflection plus anatomical left/right exchange."""
    out = sequence.copy()
    swap = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]
    pose = out[:, :26].reshape(-1, 13, 2)
    pose[:] = pose[:, swap]
    pose[..., 0] *= -1
    out[:, 26:39] = out[:, 26:39][:, swap]
    for start, end, dims in ((39, 123, 2), (123, 249, 3)):
        points = out[:, start:end].reshape(-1, 2, 21, dims)
        points[:] = points[:, ::-1].copy()
        points[..., 0] *= -1
    out[:, 249:251] = out[:, 249:251][:, ::-1]
    out[:, 251:253] = out[:, 251:253][:, ::-1]
    return out


def bounded(sequence):
    if len(sequence) > 128:
        sequence = sequence[np.linspace(0, len(sequence)-1, 128).round().astype(int)]
    return sequence


def variant(sequence, mode):
    n = len(sequence)
    if mode == 'original':
        return sequence.copy()
    if mode == 'hold_end_25pct':
        return bounded(np.concatenate([sequence, np.repeat(sequence[-1:], max(1, n//4), axis=0)]))
    if mode == 'hold_start_25pct':
        return bounded(np.concatenate([np.repeat(sequence[:1], max(1, n//4), axis=0), sequence]))
    if mode == 'double_time_steps':
        return bounded(np.repeat(sequence, 2, axis=0))
    if mode == 'half_time_steps':
        return sequence[::2].copy()
    if mode == 'middle_80pct':
        trim = max(1, n//10) if n > 2 else 0
        return sequence[trim:n-trim].copy() if trim else sequence.copy()
    if mode == 'mirror':
        return reflect_features(sequence)
    if mode in ('fill_tail_hand_context', 'fill_head_hand_context', 'fill_both_hand_context'):
        out = sequence.copy()
        for slot in (0, 1):
            valid = np.flatnonzero(sequence[:, 249+slot] > .5)
            if not len(valid):
                continue
            columns = list(range(39+42*slot, 39+42*(slot+1))) + list(range(123+63*slot, 123+63*(slot+1))) + [249+slot, 251+slot]
            if mode != 'fill_tail_hand_context':
                out[:valid[0], columns] = sequence[valid[0], columns]
            if mode != 'fill_head_hand_context':
                out[valid[-1]+1:, columns] = sequence[valid[-1], columns]
        return out
    raise ValueError(mode)


def infer(recognizer, sequences, amp=True, batch_size=64):
    predictions, scores, logits_all = [], [], []
    for start in range(0, len(sequences), batch_size):
        group = sequences[start:start+batch_size]
        data = np.zeros((len(group), 128, 254), np.float32)
        lengths = []
        for i, sequence in enumerate(group):
            data[i, :len(sequence)] = sequence
            lengths.append(len(sequence))
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            logits = recognizer.model(torch.from_numpy(data).cuda(), torch.tensor(lengths)).float()
        values, ids = logits.softmax(-1).topk(5, dim=-1)
        predictions.extend(ids.cpu().tolist())
        scores.extend(values.cpu().tolist())
        logits_all.append(logits.cpu().numpy())
    return np.asarray(predictions), np.asarray(scores), np.concatenate(logits_all)


def metrics(ids, targets):
    return {'samples': len(targets), 'top1': float((ids[:, 0] == targets).mean()),
            'top5': float((ids == targets[:, None]).any(1).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation-sample', type=int, default=512)
    parser.add_argument('--fresh-videos', action='store_true')
    parser.add_argument('--context-variants', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    args.output.mkdir(parents=True)
    checkpoint = ROOT/'models/asl-gru-128-cv-regularized-v1/official-refit/best.pt'
    cache = ROOT/'data/training/asl-gru-128-v1'
    raw = ROOT/'data/landmarks/asl-citizen-raw-v1-cpu-hands-pose'
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'training_performed': False, 'test_used_for_perturbations': False,
              'command': sys.argv, 'results': {}}

    def save(key, result):
        report['results'][key] = result
        write_json(args.output/'audit.json', report)
        print(key, json.dumps(result)[:1800], flush=True)

    torch.set_num_threads(2)
    recognizer = SignRecognizer(checkpoint, ROOT/'models')
    before = recognizer.checkpoint_sha256
    manifest = json.loads((cache/'manifest.json').read_text())
    source = json.loads((raw/'run-manifest.json').read_text())
    assets = {}
    for name, path in (('hands', recognizer.hand_model), ('pose', recognizer.pose_model)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assets[name] = {'sha256': digest, 'matches_extraction': digest == source['identity']['models'][name]['sha256']}
        assert assets[name]['matches_extraction']
    assert recognizer.labels == manifest['labels'] == json.loads((raw/'labels.json').read_text())
    assert recognizer.config['cache_id'] == manifest['cache_id']
    libraries = {name: importlib.metadata.version(name) for name in source['identity']['libraries']}
    save('identity', {'checkpoint_sha256': before, 'epoch': recognizer.epoch, 'assets': assets,
                     'libraries': libraries, 'extraction_libraries': source['identity']['libraries'],
                     'label_and_cache_match': True, 'autocast_dtype': str(torch.get_autocast_dtype('cuda'))})

    indices = {s: json.loads((cache/f'{s}-index.json').read_text()) for s in ('train', 'val')}
    arrays = {s: [np.load(cache/f'{s}-{name}.npy', mmap_mode='r') for name in ('features', 'lengths', 'targets')]
              for s in indices}
    target_results = {}
    for split in indices:
        selected = [i for i, row in enumerate(indices[split]) if row['gloss'] in TARGETS]
        a, lengths, targets = arrays[split]
        sequences = [np.array(a[i, :int(lengths[i])], dtype=np.float32) for i in selected]
        ids, _, _ = infer(recognizer, sequences)
        target_results[split] = {}
        for name in TARGETS:
            mask = np.array([indices[split][i]['gloss'] == name for i in selected])
            target_results[split][name] = metrics(ids[mask], np.asarray(targets[selected])[mask])
    save('target_train_validation', target_results)

    rng = np.random.default_rng(42)
    a, lengths, targets = arrays['val']
    selected = sorted(rng.choice(len(a), min(args.validation_sample, len(a)), replace=False).tolist())
    sequences = [np.array(a[i, :int(lengths[i])], dtype=np.float32) for i in selected]
    labels = np.asarray(targets[selected])
    write_json(args.output/'validation-sample.json', [indices['val'][i] for i in selected])
    assert np.array_equal(reflect_features(reflect_features(sequences[0])), sequences[0])
    ids, _, base_logits = infer(recognizer, sequences)
    original_ids = ids.copy()
    experiments = {'original': metrics(ids, labels)}
    modes = ['hold_end_25pct', 'hold_start_25pct', 'double_time_steps', 'half_time_steps', 'middle_80pct', 'mirror']
    if args.context_variants:
        modes += ['fill_tail_hand_context', 'fill_head_hand_context', 'fill_both_hand_context']
    for mode in modes:
        changed = [variant(x, mode) for x in sequences]
        ids, _, _ = infer(recognizer, changed)
        experiments[mode] = {**metrics(ids, labels),
                             'top1_changed_fraction': float((ids[:, 0] != original_ids[:, 0]).mean())}
        save('validation_feature_perturbations', experiments)
    ids32, _, logits32 = infer(recognizer, sequences, amp=False)
    singleton_ids, _, singleton_logits = infer(recognizer, sequences[:32], batch_size=1)
    save('numeric_and_batch_checks', {
        'fp32': metrics(ids32, labels), 'amp_fp32_top1_agreement': float((ids32[:, 0] == original_ids[:, 0]).mean()),
        'amp_fp32_mean_abs_logit_difference': float(np.abs(base_logits-logits32).mean()),
        'singleton_batched_top1_agreement': float((singleton_ids[:, 0] == original_ids[:32, 0]).mean()),
        'singleton_batched_max_abs_logit_difference': float(np.abs(singleton_logits-base_logits[:32]).max())})

    test_dir = ROOT/'data/evaluations/asl-gru-regularized-test-v1'
    signer = json.loads((test_dir/'per-signer.json').read_text())
    save('existing_test_signer_results', signer)
    save('existing_test_target_results', {row['gloss']: row for row in
         __import__('csv').DictReader((test_dir/'per-class.csv').open(encoding='utf-8')) if row['gloss'] in TARGETS})

    fresh_results = []
    for split in ('train', 'val'):
        for name in TARGETS:
            i = next(i for i, row in enumerate(indices[split]) if row['gloss'] == name)
            row = indices[split][i]
            shard = raw/f'{split}-{i//100:05d}.zip'
            with ZipFile(shard) as archive:
                record = json.loads(archive.read('records.json'))['records'][i % 100]
                assert record['video_file'] == row['video_file']
                with np.load(io.BytesIO(archive.read(record['member'])), allow_pickle=False) as loaded:
                    observations = {key: loaded[key] for key in loaded.files}
            features, quality = normalize_sequence(observations, record['frame_shape'])
            prepared = features.astype(np.float16)
            expected = arrays[split][0][i, :len(features)]
            assert np.array_equal(prepared, expected)
            result = recognizer.predict_observations(observations, record['frame_shape'])
            item = {'split': split, 'gloss': name, 'video_file': row['video_file'],
                    'frame_shape': record['frame_shape'], 'fps': record['fps'],
                    'source_frames': record['frames'], 'raw_to_cache_bitwise_equal': True,
                    'quality': quality, 'cached_raw_prediction': result['top5']}
            if args.fresh_videos:
                video = ROOT/'data/asl_citizen/ASL_Citizen/videos'/row['video_file']
                capture = cv2.VideoCapture(str(video)); fps = capture.get(cv2.CAP_PROP_FPS)
                frames = []
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frames.append(frame)
                capture.release()
                fresh = recognizer.predict_frames(frames, [round(t*1000/fps) for t in range(len(frames))])
                item['fresh_prediction'] = fresh
                item['fresh_top5_matches_raw'] = [x['label'] for x in fresh['top5']] == [x['label'] for x in result['top5']]
                item['fresh_max_top5_score_delta'] = max(abs(a['score']-b['score']) for a,b in zip(fresh['top5'], result['top5']))
            fresh_results.append(item)
            save('raw_cache_and_fresh_video_checks', fresh_results)
    after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert before == after
    report.update(status='complete', checkpoint_unchanged=True, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output/'audit.json', report)
    print('AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
