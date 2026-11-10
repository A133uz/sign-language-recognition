"""Start, inspect or stop the local CUDA baseline without losing epoch checkpoints."""

import argparse
import contextlib
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT/'.cache/training'
STATE = JOB/'job.json'
STOP = JOB/'stop.request'
OUTPUT = ROOT/'models/asl-gru-128-baseline'


def save(value):
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    for attempt in range(5):
        try:
            temporary.replace(STATE)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.1)


@contextlib.contextmanager
def lock():
    import msvcrt
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB/'supervisor.lock').open('a+b') as stream:
        if not stream.tell():
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError('Training supervisor is already running') from error
        try:
            yield
        finally:
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def running():
    try:
        with lock():
            return False
    except RuntimeError:
        return True


def kill(process):
    if process.poll() is None:
        result = subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True,
                                text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode and process.poll() is None:
            raise RuntimeError(result.stderr)
        process.wait(timeout=30)


def run(command=None, smoke_path=None):
    with lock():
        smoke_path = smoke_path or ROOT/'models/asl-gru-128-smoke/status.json'
        if not smoke_path.exists() or json.loads(smoke_path.read_text())['status'] != 'smoke_complete':
            raise RuntimeError('Complete the separate CUDA smoke test before launching the full run')
        STOP.unlink(missing_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        if command is None:
            command = [sys.executable, '-u', str(ROOT/'scripts/train_local.py'), 'train',
                       '--workers', '2', '--epochs', '50', '--batch-size', '64']
            if (OUTPUT/'last.pt').exists():
                command.append('--resume')
        state = {'status': 'starting', 'started_utc': datetime.now(timezone.utc).isoformat(),
                 'supervisor_pid': os.getpid(), 'command': command,
                 'stdout_log': str(JOB/f'{stamp}.train.stdout.log'),
                 'stderr_log': str(JOB/f'{stamp}.train.stderr.log')}
        save(state)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
        kernel.SetThreadExecutionState.restype = ctypes.c_uint32
        awake = bool(kernel.SetThreadExecutionState(0x80000001))
        state['sleep_prevention_active'] = awake
        process = None
        try:
            with open(state['stdout_log'], 'w') as stdout, open(state['stderr_log'], 'w') as stderr:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                    env={**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONUNBUFFERED': '1'},
                    creationflags=subprocess.CREATE_NO_WINDOW)
                state.update(status='running', trainer_pid=process.pid)
                save(state)
                while process.poll() is None:
                    if STOP.exists():
                        kill(process); state['status'] = 'stopped'; break
                    state['heartbeat_utc'] = datetime.now(timezone.utc).isoformat()
                    save(state); time.sleep(5)
                code = process.wait()
                if state['status'] != 'stopped':
                    state['status'] = 'completed' if code == 0 else 'failed'
                state['exit_code'] = code
        except BaseException as error:
            if process is not None:
                kill(process)
            state.update(status='failed', error=str(error))
            raise
        finally:
            if awake:
                kernel.SetThreadExecutionState(0x80000000)
            state.update(sleep_prevention_active=False, finished_utc=datetime.now(timezone.utc).isoformat())
            save(state)


def main(run_command=None, smoke_path=None, progress_path=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'status', 'stop'))
    args = parser.parse_args()
    if args.action == 'run':
        run(command=run_command, smoke_path=smoke_path)
    elif args.action == 'stop':
        if running():
            STOP.write_text('stop\n'); print('Stop requested; resume uses the last complete epoch checkpoint.')
        else:
            print('No training supervisor is running.')
    else:
        value = json.loads(STATE.read_text()) if STATE.exists() else {'status': 'not_started'}
        value['supervisor_running'] = running()
        if value['status'] in ('running', 'starting') and not value['supervisor_running']:
            value['status'] = 'not_running_last_state_'+value['status']
        path = OUTPUT/'status.json'
        if path.exists():
            value['training'] = json.loads(path.read_text())
            current = value['training'].get('current_output')
            if current:
                nested = Path(current).resolve()
                if nested.is_relative_to(OUTPUT.resolve()) and (nested/'status.json').exists():
                    value['current_run'] = json.loads((nested/'status.json').read_text())
        if progress_path is not None and progress_path.exists():
            value['queue'] = json.loads(progress_path.read_text())
        print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
