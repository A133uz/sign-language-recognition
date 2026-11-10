"""Plan and run signer-separated regularized GRU CV, then an official-split refit."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sign_language.cross_validation import load_fold, make_plan, summarize
from sign_language.dataset import write_json
from sign_language.extraction import output_lock

CACHE = ROOT/'data/training/asl-gru-128-v1'
OUTPUT = ROOT/'models/asl-gru-128-cv-regularized-v1'


def command(output, fold=None, smoke=False):
    args = [sys.executable, '-u', str(ROOT/'scripts/train_local.py'), 'train',
            '--cache', str(CACHE), '--output', str(output), '--profile', 'regularized-v1',
            '--epochs', '50', '--batch-size', '64', '--workers', '2', '--patience', '10']
    if fold is not None:
        args.extend(['--fold-file', str(OUTPUT/'folds.json'), '--fold', str(fold)])
    if smoke:
        args.append('--smoke')
    elif (output/'last.pt').exists():
        args.append('--resume')
    return args


def verify_smoke():
    status = json.loads((OUTPUT/'smoke/status.json').read_text())
    config = status['config']
    if status['status'] != 'smoke_complete' or config['profile'] != 'regularized-v1':
        raise ValueError('Run the regularized CV smoke check first')
    snapshot = OUTPUT/'engine_snapshot/src/sign_language'
    source = snapshot if snapshot.exists() else ROOT/'src/sign_language'
    for name, expected in config['source_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Source changed after smoke check')
    plan = json.loads((OUTPUT/'folds.json').read_text())
    if config['partition'].get('plan_id') != plan['plan_id']:
        raise ValueError('Smoke check belongs to another fold plan')
    return config


def run_study():
    with output_lock(OUTPUT.parent/'locks', OUTPUT):
        smoke_config = verify_smoke()
        for fold in range(3):
            load_fold(CACHE, OUTPUT/'folds.json', fold)
        state = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                 'profile': 'regularized-v1', 'total_stages': 4, 'test_evaluated': False}
        write_json(OUTPUT/'status.json', state)
        logs = OUTPUT/'logs'; logs.mkdir(exist_ok=True)
        try:
            stages = [(f'fold-{i+1}', i) for i in range(3)]+[('official-refit', None)]
            for index, (name, fold) in enumerate(stages, 1):
                path = OUTPUT/name
                if (path/'status.json').exists() and json.loads((path/'status.json').read_text())['status'] == 'completed':
                    old = json.loads((path/'config.json').read_text())
                    if old['source_sha256'] != smoke_config['source_sha256'] or old['profile'] != 'regularized-v1':
                        raise ValueError(f'{name} was trained with different source/profile settings')
                    print(f'Reusing completed {name}', flush=True)
                    continue
                args = command(path, fold)
                state.update(stage=index, phase=name, current_output=str(path), command=args)
                write_json(OUTPUT/'status.json', state)
                print(f'Starting {name}: independent fresh model or matching epoch resume', flush=True)
                with (logs/f'{name}.stdout.log').open('a', encoding='utf-8') as stdout, \
                     (logs/f'{name}.stderr.log').open('a', encoding='utf-8') as stderr:
                    result = subprocess.run(args, cwd=ROOT, stdout=stdout, stderr=stderr,
                        env={**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONUNBUFFERED': '1'},
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if result.returncode:
                    raise RuntimeError(f'{name} exited {result.returncode}; see {logs/name}.stderr.log')
                if json.loads((path/'status.json').read_text())['status'] != 'completed':
                    raise RuntimeError(f'{name} did not complete')
                summarize(OUTPUT)
            state.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(OUTPUT/'status.json', state)
            return summarize(OUTPUT)
        except BaseException as error:
            state.update(status='failed', error=str(error))
            write_json(OUTPUT/'status.json', state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'smoke', 'run', 'summary'))
    args = parser.parse_args()
    if args.action == 'plan':
        plan = make_plan(CACHE, OUTPUT/'folds.json', folds=3)
        print(json.dumps({'plan_id': plan['plan_id'], 'n_splits': plan['n_splits'],
                          'partitions': [{k: v for k, v in item.items() if k not in ('train', 'validation')}
                                         for item in plan['partitions']]}, indent=2))
    elif args.action == 'smoke':
        raise SystemExit(subprocess.call(command(OUTPUT/'smoke', fold=0, smoke=True), cwd=ROOT))
    elif args.action == 'run':
        print(json.dumps(run_study(), indent=2))
    else:
        print(json.dumps(summarize(OUTPUT), indent=2))


if __name__ == '__main__':
    main()
