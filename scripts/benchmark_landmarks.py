"""Benchmark bounded CPU extraction on identical clips with fresh output folders."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def validate_features(folder):
    import io
    import numpy as np

    clips = frames = 0
    for path in sorted(folder.glob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            records = json.loads(archive.read('records.json'))['records']
            for record in records:
                if record['status'] != 'ok':
                    raise RuntimeError(f'Failed benchmark clip: {record}')
                with np.load(io.BytesIO(archive.read(record['member'])), allow_pickle=False) as data:
                    count = record['frames']
                    if (data['hands_xyz'].shape != (count, 2, 21, 3)
                            or data['pose_xyz'].shape != (count, 33, 3)
                            or not np.array_equal(data['frame_index'], np.arange(count))
                            or not np.all(np.diff(data['timestamp_ms']) > 0)
                            or any(not np.isfinite(data[key]).all() for key in data.files)):
                        raise RuntimeError(f'Invalid benchmark feature arrays: {record["video_file"]}')
                    clips += 1
                    frames += count
    return {'validated_clips': clips, 'validated_frames': frames,
            'checks': 'finite numeric arrays, expected hand/pose shapes, frame indices and increasing timestamps'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips', type=int, default=100)
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--dataset-root', type=Path, default=ROOT/'data/asl_citizen/ASL_Citizen')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.clips <= 500 or not args.workers or any(n not in (1,2,3,4) for n in args.workers) or len(set(args.workers)) != len(args.workers):
        parser.error('Use 1-500 clips and unique worker counts from 1-4.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output or ROOT/'.cache/landmark-benchmarks'/f'{stamp}-{uuid.uuid4().hex[:6]}'
    output = output.resolve()
    if output.is_relative_to(args.dataset_root.resolve()):
        parser.error('Keep benchmark output outside the raw dataset.')
    output.mkdir(parents=True, exist_ok=False)
    cpu = platform.processor()
    if sys.platform == 'win32':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'cpu': cpu,
        'logical_cpus': os.cpu_count(), 'platform': platform.platform(),
        'python': sys.version, 'clips_per_run': args.clips, 'delegate': 'CPU',
        'include_pose': True, 'training_performed': False,
        'timing_scope': 'Fresh CLI process, metadata/model checks, tracker initialization, full decoding, hand/pose inference, NPZ compression and verified local shard writes; validation timed separately.',
        'sampling': 'Existing extractor: sorted train filenames, Random(0) shuffle, first N; identical for every worker count.',
        'runs': [],
    }
    report_path = output/'benchmark-report.json'
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print('Benchmark folder:', output, flush=True)
    print('CPU:', cpu, '| logical CPUs:', os.cpu_count(), flush=True)
    environment = {**os.environ, 'MPLBACKEND':'Agg', 'PYTHONUNBUFFERED':'1',
                   'PYTHONIOENCODING':'utf-8', 'TF_CPP_MIN_LOG_LEVEL':'2'}
    previous_id = None
    for workers in args.workers:
        folder = output/f'workers-{workers}'
        stdout_path, native_path = output/f'workers-{workers}.stdout.log', output/f'workers-{workers}.native.log'
        command = [sys.executable, '-u', str(ROOT/'scripts/project.py'), 'extract-landmarks',
                   '--dataset-root', str(args.dataset_root.resolve()), '--models', str(ROOT/'models'),
                   '--output', str(folder), '--work-dir', str(output/f'work-{workers}'),
                   '--splits', 'train', '--limit', str(args.clips), '--shard-size', '100', '--workers', str(workers)]
        print(f'Starting fresh {args.clips}-clip benchmark with {workers} worker(s).', flush=True)
        began = time.perf_counter()
        with stdout_path.open('w', encoding='utf-8') as stdout, native_path.open('w', encoding='utf-8') as native:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                       stderr=native, text=True, encoding='utf-8', errors='replace')
            try:
                for line in process.stdout:
                    stdout.write(line)
                    stdout.flush()
                    if line.startswith(('Delegate ', 'train-', 'Committed ')):
                        print(f'workers={workers}: {line}', end='', flush=True)
                code = process.wait()
            except KeyboardInterrupt:
                if sys.platform == 'win32':
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
                else:
                    process.terminate()
                process.wait()
                raise
            finally:
                process.stdout.close()
        wall = time.perf_counter()-began
        if code:
            raise RuntimeError(f'Benchmark failed (exit {code}); inspect {stdout_path} and {native_path}')
        summary = json.loads((folder/'summary.json').read_text())
        manifest = json.loads((folder/'run-manifest.json').read_text())
        if summary['newly_processed_clips'] != args.clips or summary['failed_clips']:
            raise RuntimeError('Benchmark did not freshly extract every selected clip.')
        if previous_id is not None and manifest['run_id'] != previous_id:
            raise RuntimeError('Worker configurations did not use identical source/models/settings/samples.')
        previous_id = manifest['run_id']
        validation_began = time.perf_counter()
        validation = validate_features(folder)
        if validation['validated_clips'] != args.clips or validation['validated_frames'] != summary['decoded_frames']:
            raise RuntimeError('Validated feature totals differ from extraction summary.')
        native_text = native_path.read_text(encoding='utf-8', errors='replace')
        errors = [line for line in native_text.splitlines() if re.match(r'^E\d{4}|^ERROR', line)]
        run = {
            'workers': workers, 'wall_seconds': wall,
            'extractor_session_seconds': summary['session_elapsed_seconds'],
            'seconds_per_clip': wall/args.clips, 'frames_per_second': summary['decoded_frames']/wall,
            'extrapolated_corpus_hours': wall/args.clips*83399/3600,
            'clips': args.clips, 'frames': summary['decoded_frames'],
            'frames_with_hands': summary['frames_with_hands'], 'frames_with_pose': summary['frames_with_pose'],
            'feature_archive_bytes': summary['feature_archive_bytes'],
            'run_id': manifest['run_id'], 'identity': manifest['identity'],
            'native_error_lines': errors[:20], 'native_error_line_count': len(errors),
            'tensor_write_error_logged': 'Tensors are designed for single writes' in native_text,
            'feature_validation': validation, 'validation_seconds': time.perf_counter()-validation_began,
        }
        report['runs'].append(run)
        report_path.write_text(json.dumps(report, indent=2)+'\n')
        print('RESULT:', json.dumps({k:v for k,v in run.items() if k not in ('identity',)}), flush=True)
    report['fastest_workers'] = min(report['runs'], key=lambda run: run['wall_seconds'])['workers']
    report['status'] = 'complete'
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print('Completed benchmark:', report_path, flush=True)


if __name__ == '__main__':
    main()
