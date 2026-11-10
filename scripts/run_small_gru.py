"""Wait for successful CV completion, then train a fresh 64-unit baseline."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sign_language.dataset import write_json

JOB = ROOT/'.cache/small-gru'
OUTPUT = ROOT/'models/asl-gru-64-baseline'
SMOKE = ROOT/'models/asl-gru-64-smoke'


def verify_engine():
    smoke = json.loads((SMOKE/'status.json').read_text())
    config = smoke['config']
    if (smoke['status'] != 'smoke_complete' or config['model']['hidden_size'] != 64
            or config['profile'] != 'baseline' or config['partition']['mode'] != 'official_train_val'):
        raise ValueError('A successful 64-unit baseline smoke check is required')
    for name, expected in config['source_sha256'].items():
        if hashlib.sha256((ROOT/'src/sign_language'/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Training source changed while queued; rerun the smoke check')


def main():
    JOB.mkdir(parents=True, exist_ok=True)
    state = {'status': 'waiting_for_cv', 'queued_utc': datetime.now(timezone.utc).isoformat(),
             'hidden_size': 64, 'profile': 'baseline', 'test_evaluated': False}
    write_json(JOB/'queue.json', state)
    verify_engine()
    print('Queued 64-unit baseline; waiting for the existing CV study to complete successfully.', flush=True)
    try:
        while True:
            result = subprocess.run([sys.executable, str(ROOT/'scripts/local_cv_job.py'), 'status'],
                                    capture_output=True, text=True, check=True,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            cv = json.loads(result.stdout)
            state.update(cv_status=cv['status'], heartbeat_utc=datetime.now(timezone.utc).isoformat())
            write_json(JOB/'queue.json', state)
            if not cv['supervisor_running']:
                if cv['status'] != 'completed':
                    raise RuntimeError('CV did not complete successfully; queue stopped without starting GPU training')
                break
            time.sleep(10)
        verify_engine()
        command = [sys.executable, '-u', str(ROOT/'scripts/train_local.py'), 'train',
                   '--hidden-size', '64', '--profile', 'baseline', '--workers', '2',
                   '--epochs', '50', '--batch-size', '64', '--patience', '10', '--output', str(OUTPUT)]
        if (OUTPUT/'last.pt').exists():
            command.append('--resume')
        state.update(status='training', started_utc=datetime.now(timezone.utc).isoformat(), command=command)
        write_json(JOB/'queue.json', state)
        print('CV completed. Starting 64-unit baseline.', flush=True)
        result = subprocess.run(command, cwd=ROOT, env={**os.environ, 'MPLBACKEND': 'Agg'},
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            raise RuntimeError(f'64-unit training exited with code {result.returncode}')
        state.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(JOB/'queue.json', state)
    except BaseException as error:
        state.update(status='failed', error=str(error))
        write_json(JOB/'queue.json', state)
        raise


if __name__ == '__main__':
    main()
