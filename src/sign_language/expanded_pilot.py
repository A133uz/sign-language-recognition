"""Checksummed 15-sign train/validation subset; preserve official source partitions."""
import hashlib
import json
from pathlib import Path

import numpy as np

from sign_language.five_sign_pilot import sha256, write_json
from sign_language.preprocessing import CONFIG

SIGNS = ('HELLO', 'NO', 'PLEASE', 'THANKYOU', 'YES',
         'ME', 'YOU', 'WE', 'MY', 'YOUR', 'OUR', 'HELP', 'GOOD', 'NEED', 'LOVE')


def digest_chunk(arrays, start, count):
    return hashlib.sha256(b''.join(a[start:start+count].tobytes() for a in arrays)).hexdigest()


def prepare_subset(parent, output, signs=SIGNS):
    parent, output = Path(parent), Path(output)
    if output.exists():
        raise FileExistsError('Subset already exists; preserve it')
    manifest = json.loads((parent / 'manifest.json').read_text())
    status = json.loads((parent / 'status.json').read_text())
    if status['status'] != 'complete' or status['cache_id'] != manifest['cache_id']:
        raise ValueError('Parent cache incomplete or mismatched')
    if manifest['identity']['preprocessing'] != CONFIG or manifest['identity']['preprocessing_sha256'] != sha256(Path(__file__).with_name('preprocessing.py')):
        raise ValueError('Parent preprocessing differs')
    labels = {g: i for i, g in enumerate(signs)}
    if len(labels) != len(signs) or any(g not in manifest['labels'] for g in signs):
        raise ValueError('Duplicate or missing requested signs')
    prepared, participants, checks, metadata = {}, {}, {}, {}
    for split in ('train', 'val'):
        index = json.loads((parent / f'{split}-index.json').read_text())
        arrays = [np.load(parent / f'{split}-{name}.npy', mmap_mode='r') for name in ('features', 'lengths', 'targets')]
        if any(len(a) != len(index) for a in arrays):
            raise ValueError('Parent row count mismatch')
        selected = [i for i, row in enumerate(index) if row['gloss'] in labels]
        needed = set(selected)
        checked = {}
        for marker in sorted(parent.glob(f'{split}-*.prepared.json')):
            record = json.loads(marker.read_text())
            start, count = record['start'], record['count']
            covered = needed.intersection(range(start, start + count))
            if not covered:
                continue
            if record['cache_id'] != manifest['cache_id'] or digest_chunk(arrays, start, count) != record['sha256']:
                raise ValueError('Parent shard checksum differs: ' + marker.name)
            checked[marker.name] = {k: record[k] for k in ('start', 'count', 'sha256')}
            needed.difference_update(covered)
        if needed:
            raise ValueError('Selected rows lack verified prepared-shard coverage')
        old_targets = np.asarray([manifest['labels'][index[i]['gloss']] for i in selected])
        if not np.array_equal(arrays[2][selected], old_targets):
            raise ValueError('Parent index/target mismatch')
        features = np.array(arrays[0][selected], copy=True)
        lengths = np.array(arrays[1][selected], copy=True)
        if features.shape[1:] != (128, 254) or not np.isfinite(features).all() or np.any((lengths < 1) | (lengths > 128)):
            raise ValueError('Invalid prepared features')
        rows = [{**index[i], 'class_id': labels[index[i]['gloss']], 'parent_class_id': int(old_targets[j]), 'parent_index': i}
                for j, i in enumerate(selected)]
        targets = np.asarray([labels[row['gloss']] for row in rows], dtype=np.int64)
        if set(targets) != set(range(len(labels))):
            raise ValueError('A selected class is absent from a split')
        participants[split] = sorted({row['participant_id'] for row in rows})
        prepared[split] = (features, lengths, targets, rows)
        checks[split] = checked
        metadata[split] = {name: sha256(parent / f'{split}-{name}.{ext}') for name, ext in
                           [('index', 'json'), ('lengths', 'npy'), ('targets', 'npy')]}
    if set(participants['train']) & set(participants['val']):
        raise ValueError('Signer overlap')
    identity = {'parent_cache_id': manifest['cache_id'], 'preprocessing': CONFIG,
                'preprocessing_sha256': manifest['identity']['preprocessing_sha256'], 'signs': list(signs),
                'partition': 'official train/val only; test not read or copied',
                'selection': 'all available examples of selected labels; no quality filtering',
                'parent_row_indices': {s: [r['parent_index'] for r in values[3]] for s, values in prepared.items()}}
    cache_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    output.mkdir(parents=True)
    counts = {}
    for split, (features, lengths, targets, rows) in prepared.items():
        for name, value in [('features', features), ('lengths', lengths), ('targets', targets)]:
            np.save(output / f'{split}-{name}.npy', value, allow_pickle=False)
        write_json(output / f'{split}-index.json', rows)
        counts[split] = {g: sum(r['gloss'] == g for r in rows) for g in signs}
    result = {'cache_id': cache_id, 'identity': identity, 'labels': labels, 'counts': counts,
              'participants': participants, 'verified_parent_chunks': checks, 'parent_metadata_sha256': metadata,
              'files_sha256': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / 'manifest.json', result)
    write_json(output / 'status.json', {'status': 'complete', 'cache_id': cache_id, 'counts': counts})
    return result


def verify_parent_chunks(parent, subset):
    parent = Path(parent)
    for split, chunks in subset['verified_parent_chunks'].items():
        arrays = [np.load(parent / f'{split}-{name}.npy', mmap_mode='r') for name in ('features', 'lengths', 'targets')]
        for name, record in chunks.items():
            if digest_chunk(arrays, record['start'], record['count']) != record['sha256']:
                raise ValueError('Parent chunk changed: ' + name)
        for name, digest in subset['parent_metadata_sha256'][split].items():
            extension = 'json' if name == 'index' else 'npy'
            if sha256(parent / f'{split}-{name}.{extension}') != digest:
                raise ValueError('Parent metadata changed')
    return True
