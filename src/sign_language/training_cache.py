"""Prepare checksummed raw shards into resumable, memory-mapped GRU inputs."""

import hashlib
import io
import json
import multiprocessing
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sign_language.dataset import write_json
from sign_language.extraction import fingerprint, load_rows, output_lock, read_completion, shard_plan
from sign_language.preprocessing import CONFIG, INPUT_SIZE, normalize_sequence


def prepare_shard(job):
    path, expected = job
    results = []
    with zipfile.ZipFile(path) as archive:
        records = json.loads(archive.read('records.json'))['records']
        if len(records) != len(expected):
            raise ValueError(f'Wrong raw shard length: {path}')
        for raw, row in zip(records, expected, strict=True):
            if raw['status'] != 'ok' or any(raw.get(k) != v for k, v in row.items()):
                raise ValueError(f'Failed or mismatched raw record: {raw.get("video_file")}')
            with np.load(io.BytesIO(archive.read(raw['member'])), allow_pickle=False) as arrays:
                features, quality = normalize_sequence(arrays, raw['frame_shape'])
            padded = np.zeros((128, INPUT_SIZE), dtype=np.float16)
            padded[:len(features)] = features
            results.append((padded, len(features), quality))
    return results


def prepare_cache(raw, dataset, output, workers=2):
    raw, dataset, output = Path(raw), Path(dataset), Path(output)
    with output_lock(output.parent/'locks', output):
        return _prepare_cache(raw, dataset, output, workers)


def _prepare_cache(raw, dataset, output, workers):
    if workers not in (1, 2, 3, 4):
        raise ValueError('Use 1–4 preparation workers')
    source = json.loads((raw/'run-manifest.json').read_text())
    summary = json.loads((raw/'summary.json').read_text())
    if summary['status'] != 'complete' or summary['failed_clips']:
        raise ValueError('Complete raw landmarks without failed clips are required')
    rows, labels, hashes = load_rows(dataset)
    if hashes != source['identity']['csv_sha256'] or labels != json.loads((raw/'labels.json').read_text()):
        raise ValueError('Raw features and official metadata/labels differ')
    plan = shard_plan(rows, ('train', 'val', 'test'), 100, None)
    identity = {'raw_run_id': source['run_id'], 'preprocessing': CONFIG,
                'preprocessing_sha256': hashlib.sha256(Path(__file__).with_name('preprocessing.py').read_bytes()).hexdigest(),
                'dtype': 'float16', 'numpy': np.__version__}
    cache_id = fingerprint(identity)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['cache_id'] != cache_id:
            raise ValueError('Prepared cache differs; use a new cache directory')
    elif any(output.iterdir()):
        raise ValueError('Nonempty prepared cache without manifest')
    write_json(manifest, {'cache_id': cache_id, 'identity': identity, 'labels': labels})
    arrays, counts = {}, {}
    for split in ('train', 'val', 'test'):
        selected = [r for _, group in plan for r in group if r['split'] == split]
        counts[split] = len(selected)
        write_json(output/f'{split}-index.json', selected)
        arrays[split] = []
        for name, shape, dtype in [('features', (len(selected), 128, INPUT_SIZE), 'float16'),
                                   ('lengths', (len(selected),), 'int16'), ('targets', (len(selected),), 'int64')]:
            path = output/f'{split}-{name}.npy'
            if path.exists():
                a = np.load(path, mmap_mode='r+')
                if a.shape != shape or a.dtype != np.dtype(dtype):
                    raise ValueError(f'Invalid prepared array: {path}')
            else:
                a = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
            arrays[split].append(a)
    offsets = {s: 0 for s in counts}
    pending = []
    reused = 0
    for name, group in plan:
        # Verify every source archive and record before reuse or transformation.
        if read_completion(raw, name, source['run_id'], group) is None:
            raise ValueError(f'Raw shard failed integrity checks: {name}')
        split = group[0]['split']; start = offsets[split]; offsets[split] += len(group)
        marker = output/f'{name}.prepared.json'
        if marker.exists():
            m = json.loads(marker.read_text())
            if m['cache_id'] == cache_id and m['count'] == len(group) and m['start'] == start:
                a, lengths, targets = arrays[split]
                digest = hashlib.sha256(a[start:start+len(group)].tobytes()+lengths[start:start+len(group)].tobytes()+targets[start:start+len(group)].tobytes()).hexdigest()
                if digest == m['sha256']:
                    reused += len(group)
                    continue
        pending.append((name, group, split, start))
    print(f'Cache verified: {reused} reused clips, {sum(len(x[1]) for x in pending)} to prepare; workers={workers}', flush=True)
    began = time.monotonic(); processed = reused
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        # Bounded submissions: at most two shards per worker in flight.
        for begin in range(0, len(pending), workers*2):
            batch = pending[begin:begin+workers*2]
            jobs = [(str(raw/f'{name}.zip'), group) for name, group, _, _ in batch]
            for (name, group, split, start), results in zip(batch, pool.map(prepare_shard, jobs), strict=True):
                a, lengths, targets = arrays[split]
                quality = {'frames': 0, 'frames_with_hands': 0, 'frames_with_body_anchor': 0}
                for i, (feature, length, q) in enumerate(results):
                    a[start+i] = feature; lengths[start+i] = length; targets[start+i] = group[i]['class_id']
                    for key in quality:
                        quality[key] += q[key]
                for x in arrays[split]:
                    x.flush()
                end = start+len(group)
                digest = hashlib.sha256(a[start:end].tobytes()+lengths[start:end].tobytes()+targets[start:end].tobytes()).hexdigest()
                write_json(output/f'{name}.prepared.json', {'cache_id': cache_id, 'count': len(group),
                           'start': start, 'sha256': digest, 'quality': quality})
                processed += len(group)
                write_json(output/'status.json', {'status': 'preparing', 'processed': processed,
                           'total': len(rows), 'elapsed_seconds': time.monotonic()-began})
                print(f'Prepared {name}: {processed}/{len(rows)} clips', flush=True)
    final = {'status': 'complete', 'cache_id': cache_id, 'processed': processed,
             'counts': counts, 'elapsed_seconds': time.monotonic()-began}
    write_json(output/'status.json', final)
    return final
