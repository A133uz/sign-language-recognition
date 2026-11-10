"""Run, inspect or stop the resumable local CPU extraction job. No training."""

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
JOB = ROOT / '.cache/local-landmark-extraction'
STATE = JOB / 'state.json'
STOP = JOB / 'stop.request'
OUTPUT = ROOT / 'data/landmarks/asl-citizen-raw-v1-cpu-hands-pose'


def now():
    return datetime.now(timezone.utc).isoformat()


def save_state(value):
    temporary = STATE.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    for attempt in range(5):
        try:
            temporary.replace(STATE)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.1)


@contextlib.contextmanager
def supervisor_lock():
    import msvcrt
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB/'supervisor.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError('The local extraction supervisor is already running.') from error
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def is_running():
    if sys.platform != 'win32':
        return False
    try:
        with supervisor_lock():
            return False
    except RuntimeError:
        return True


def tail(path, lines=8):
    if not path.exists():
        return []
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-12000))
        return stream.read().decode('utf-8', errors='replace').splitlines()[-lines:]


def status():
    if not STATE.exists():
        return {'status':'not_started'}
    value = json.loads(STATE.read_text(encoding='utf-8'))
    value['supervisor_running'] = is_running()
    if value['status'] == 'running' and not value['supervisor_running']:
        value['status'] = 'not_running_last_state_was_running'
    summary = OUTPUT/'summary.json'
    if summary.exists():
        value['committed_progress'] = json.loads(summary.read_text())
    if value.get('stdout_log'):
        value['recent_progress'] = tail(Path(value['stdout_log']))
    return value


def kill_tree(process):
    if process.poll() is None:
        result = subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode and process.poll() is None:
            raise RuntimeError(f'Could not stop extraction process tree: {result.stderr or result.stdout}')
        process.wait(timeout=30)


def read_summary():
    path = OUTPUT/'summary.json'
    return json.loads(path.read_text()) if path.exists() else {}


def attempt_action(code, summary, native_tail, failures, max_retries):
    """Retry only worker-pool crashes; never conceal configuration or I/O errors."""
    if code == 0 and summary.get('status') == 'complete':
        return 'completed'
    if code == 2 and summary.get('status') == 'complete_with_failures':
        return 'completed_with_failures'
    if code in (0, 2) and summary.get('status') == 'partial' and summary.get('newly_processed_clips', 0) > 0:
        return 'refreshing_workers'
    if code != 0 and 'concurrent.futures.process.BrokenProcessPool:' in native_tail and failures <= max_retries:
        return 'retrying_worker_crash'
    return 'failed'


def run_attempt(state, command, environment):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    stdout_path = JOB/f'{stamp}.stdout.log'
    native_path = JOB/f'{stamp}.native.log'
    state.update(stdout_log=str(stdout_path), native_log=str(native_path), command=command)
    process = None
    try:
        with stdout_path.open('w', encoding='utf-8') as stdout, native_path.open('w', encoding='utf-8') as errors:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=errors,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            state.update(status='running', extractor_pid=process.pid, heartbeat_utc=now())
            save_state(state)
            next_heartbeat = time.monotonic()+30
            while process.poll() is None:
                if STOP.exists():
                    kill_tree(process)
                    state['status'] = 'stopped'
                    break
                if time.monotonic() >= next_heartbeat:
                    state['heartbeat_utc'] = now()
                    save_state(state)
                    next_heartbeat = time.monotonic()+30
                time.sleep(2)
            return process.wait()
    except BaseException:
        if process is not None:
            kill_tree(process)
        raise


def extraction_loop(state, command, environment, max_crash_retries):
    failures = 0
    state['attempt_history'] = []
    while True:
        if STOP.exists():
            state['status'] = 'stopped'
            break
        before = read_summary().get('processed_clips', 0)
        state['attempt'] = len(state['attempt_history'])+1
        code = run_attempt(state, command, environment)
        summary = read_summary()
        after = summary.get('processed_clips', 0)
        native_tail = '\n'.join(tail(Path(state['native_log']), 40))
        crashed = code != 0 and 'concurrent.futures.process.BrokenProcessPool:' in native_tail
        failures = (1 if after > before else failures+1) if crashed else 0
        action = attempt_action(code, summary, native_tail, failures, max_crash_retries)
        if state['status'] == 'stopped' or STOP.exists():
            action = 'stopped'
        state['attempt_history'].append({
            'attempt': state['attempt'], 'exit_code': code, 'action': action,
            'processed_before': before, 'processed_after': after,
            'finished_utc': now(), 'stdout_log': state['stdout_log'], 'native_log': state['native_log'],
        })
        state.update(status=action, exit_code=code, heartbeat_utc=now(),
                     consecutive_worker_crashes=failures)
        save_state(state)
        if action not in ('refreshing_workers', 'retrying_worker_crash'):
            break
        if action == 'retrying_worker_crash':
            # Keep stop requests responsive during a short bounded backoff.
            for _ in range(5):
                if STOP.exists():
                    break
                time.sleep(2)


def run(workers, chunk_shards=25, max_crash_retries=2):
    if sys.platform != 'win32':
        raise RuntimeError('This background supervisor targets the local Windows machine.')
    with supervisor_lock():
        # Remove only a previous stop flag in this supervisor's own directory.
        STOP.unlink(missing_ok=True)
        command = [sys.executable, '-u', str(ROOT/'scripts/project.py'), 'extract-landmarks',
                   '--dataset-root', str(ROOT/'data/asl_citizen/ASL_Citizen'),
                   '--models', str(ROOT/'models'), '--output', str(OUTPUT),
                   '--work-dir', str(ROOT/'.cache/extraction-work'),
                   '--workers', str(workers), '--shard-size', '100',
                   '--splits', 'train', 'val', 'test', '--max-shards', str(chunk_shards)]
        state = {'status':'starting', 'started_utc':now(), 'supervisor_pid':os.getpid(),
                 'workers':workers, 'delegate':'CPU', 'include_pose':True,
                 'output':str(OUTPUT), 'chunk_shards':chunk_shards, 'max_crash_retries':max_crash_retries,
                 'command':command, 'training_performed':False}
        save_state(state)
        # A scoped execution request prevents automatic idle sleep while this
        # job runs. No power-plan setting is changed, and the display may sleep.
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
        kernel.SetThreadExecutionState.restype = ctypes.c_uint32
        continuous, system_required = 0x80000000, 0x00000001
        awake = bool(kernel.SetThreadExecutionState(continuous | system_required))
        state['automatic_idle_sleep_prevention'] = awake
        state['sleep_prevention_active'] = awake
        environment = {**os.environ, 'MPLBACKEND':'Agg', 'PYTHONUNBUFFERED':'1',
                       'PYTHONIOENCODING':'utf-8', 'TF_CPP_MIN_LOG_LEVEL':'2'}
        try:
            extraction_loop(state, command, environment, max_crash_retries)
            state.update(finished_utc=now(), heartbeat_utc=now())
        except BaseException as error:
            state.update(status='failed', error=f'{type(error).__name__}: {error}', finished_utc=now())
            raise
        finally:
            if awake:
                kernel.SetThreadExecutionState(continuous)
            state['sleep_prevention_active'] = False
            save_state(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run','status','stop'))
    parser.add_argument('--workers', type=int, choices=(1,2,3,4), default=2)
    args = parser.parse_args()
    if args.action == 'run':
        run(args.workers)
    elif args.action == 'stop':
        if is_running():
            STOP.write_text(now()+'\n')
            print('Stop requested. Committed batches remain resumable; the in-progress batch will repeat.')
        else:
            print('No extraction supervisor is running.')
    else:
        print(json.dumps(status(), indent=2))


if __name__ == '__main__':
    main()
