"""Train a separate ST-GCN; reuse the verified landmark cache and preserve GRUs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT/'models/asl-stgcn27-v1'


def cache_verification_stamp(cache):
    """Reuse a full verification only while payload metadata and marker bytes agree."""
    payload = {}
    for split in ('train', 'val'):
        for suffix in ('features.npy','lengths.npy','targets.npy','index.json'):
            path = cache/f'{split}-{suffix}'
            stat = path.stat()
            payload[path.name] = [stat.st_size, stat.st_mtime_ns]
        for path in sorted(cache.glob(f'{split}-*.prepared.json')):
            payload[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('manifest.json','status.json'):
        payload[name] = hashlib.sha256((cache/name).read_bytes()).hexdigest()
    for split in ('train','val','test'):
        path = ROOT/f'data/asl_citizen/ASL_Citizen/splits/{split}.csv'
        payload[f'official-{split}.csv'] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('preprocessing.py','stgcn_graph.py','stgcn_model.py','evaluation.py'):
        payload[f'source-{name}'] = hashlib.sha256((ROOT/'src/sign_language'/name).read_bytes()).hexdigest()
    return payload


def record_cache_verification(cache):
    path = ROOT/'.cache/stgcn-training/cache-verification.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'verified_utc': datetime.now(timezone.utc).isoformat(),
                               'counts': {'train': 40154, 'val': 10304},
                               'stamp': cache_verification_stamp(cache)}, indent=2))


def frozen_source(output, create=False):
    snapshot = output/'engine_snapshot'
    manifest = snapshot/'manifest.json'
    if not manifest.exists() and create:
        if snapshot.exists():
            raise ValueError('Incomplete engine snapshot; inspect before proceeding')
        directory = snapshot/'src/sign_language'
        directory.mkdir(parents=True)
        hashes = {}
        for path in (ROOT/'src/sign_language').glob('*.py'):
            target = directory/path.name
            shutil.copyfile(path, target)
            hashes[path.name] = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest.write_text(json.dumps({'files': hashes}, indent=2))
    if manifest.exists():
        for name, digest in json.loads(manifest.read_text())['files'].items():
            if hashlib.sha256((snapshot/'src/sign_language'/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Frozen engine changed: {name}')
        return snapshot/'src'
    return ROOT/'src'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('train', 'verify-cache'))
    parser.add_argument('--cache', type=Path, default=ROOT/'data/training/asl-gru-128-v1')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, choices=(0,1,2,3,4), default=2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1:
        parser.error('Epochs, batch size and patience must be positive')
    if args.smoke and args.output == DEFAULT_OUTPUT:
        args.output = ROOT/'models/asl-stgcn27-smoke'
    sys.path.insert(0, str(frozen_source(args.output, create=args.action == 'train' and not args.smoke)))
    if args.action == 'verify-cache':
        receipt = ROOT/'.cache/stgcn-training/cache-verification.json'
        if receipt.exists() and json.loads(receipt.read_text()).get('stamp') == cache_verification_stamp(args.cache):
            print('Reused complete train/validation verification; payload metadata, markers and source identities unchanged.', flush=True)
            return
        from sign_language.evaluation import validate_cache
        import torch
        checkpoint = torch.load(ROOT/'models/asl-stgcn27-smoke/best.pt', map_location='cpu', weights_only=True)
        for split in ('train','val'):
            index = validate_cache(args.cache, ROOT/'data/asl_citizen/ASL_Citizen', checkpoint, split)
            print('VERIFIED', split, len(index), flush=True)
        record_cache_verification(args.cache)
    else:
        from sign_language.stgcn_training import train_stgcn
        result = train_stgcn(args.cache, args.output, epochs=1 if args.smoke else args.epochs,
                             batch_size=args.batch_size, workers=args.workers, patience=args.patience,
                             resume=args.resume, smoke=args.smoke)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
