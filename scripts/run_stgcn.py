"""Verify train/val cache, train ST-GCN, then evaluate the frozen selected model."""

import json
import hashlib
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'models/asl-stgcn27-v1'
PROGRESS = ROOT/'.cache/stgcn-training/pipeline.json'


def main():
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    for phase, command in (
        ('cache_verification', ['train_stgcn.py', 'verify-cache']),
        ('training', ['train_stgcn.py', 'train', *(['--resume'] if (OUTPUT/'last.pt').exists() else [])]),
        ('test_evaluation', ['evaluate_stgcn.py']),
        ('results_report', ['report_stgcn.py'])):
        state = {'phase': phase, 'status': 'running', 'updated_utc': datetime.now(timezone.utc).isoformat()}
        PROGRESS.write_text(json.dumps(state, indent=2))
        if phase == 'training' and (OUTPUT/'status.json').exists():
            if json.loads((OUTPUT/'status.json').read_text()).get('status') == 'completed':
                continue
        if phase == 'test_evaluation':
            report = ROOT/'data/evaluations/asl-stgcn27-test-v1/report.json'
            if report.exists():
                previous = json.loads(report.read_text())
                digest = hashlib.sha256((OUTPUT/'best.pt').read_bytes()).hexdigest()
                if previous.get('status') != 'complete' or previous.get('checkpoint_sha256') != digest:
                    raise ValueError('Existing evaluation differs; inspect it before rerunning')
                continue
        result = subprocess.run([sys.executable, '-u', str(ROOT/'scripts'/command[0]), *command[1:]], cwd=ROOT)
        if result.returncode:
            state.update(status='failed', exit_code=result.returncode)
            PROGRESS.write_text(json.dumps(state, indent=2))
            raise SystemExit(result.returncode)
    state.update(status='complete', phase='complete', updated_utc=datetime.now(timezone.utc).isoformat())
    PROGRESS.write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
