"""Template; run scripts/build_colab_bundle.py and use its generated cell."""

import base64
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import zipfile
from pathlib import Path

# The builder embeds only reviewed source code, not dataset files or model weights.
PAYLOAD = "__SOURCE_BUNDLE_BASE64__"
PAYLOAD_SHA256 = "__SOURCE_BUNDLE_SHA256__"

DATASET_ROOT = Path('/content/asl_citizen/ASL_Citizen')
INCLUDE_POSE = True       # Recommended: keep hand position relative to body/head.
WORKERS = 1              # Start with 1. Try 2 only after timing a pilot.
SHARD_SIZE = 100          # Keep unchanged when resuming a full run.

if sys.version_info[:2] not in ((3, 12), (3, 13)):
    raise RuntimeError(f'This dependency set targets Python 3.12/3.13; runtime is {sys.version}. Report this before installing.')
if not (DATASET_ROOT / 'videos').is_dir() or not (DATASET_ROOT / 'splits/train.csv').is_file():
    raise RuntimeError(f'Dataset not found at {DATASET_ROOT}; keep the runtime containing your extracted data.')

source = base64.b64decode(PAYLOAD)
if hashlib.sha256(source).hexdigest() != PAYLOAD_SHA256:
    raise RuntimeError('Embedded source bundle checksum mismatch')
PROJECT = Path('/content') / f'asl-extractor-{PAYLOAD_SHA256[:10]}'
PROJECT.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(source)) as bundle:
    for name in bundle.namelist():
        target = (PROJECT / name).resolve()
        if not target.is_relative_to(PROJECT.resolve()):
            raise RuntimeError('Unsafe source bundle path')
    bundle.extractall(PROJECT)

# An isolated runtime environment avoids altering packages imported by this notebook.
ENV_PYTHON = PROJECT / '.venv/bin/python'
if not ENV_PYTHON.exists():
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(PROJECT / '.venv')], check=True)
subprocess.run([
    sys.executable, '-m', 'pip', '--python', str(ENV_PYTHON),
    '--disable-pip-version-check', 'install',
    '-r', str(PROJECT / 'requirements-colab.txt'),
], check=True)

def _extraction_environment():
    # Colab's inline backend belongs to the notebook kernel, not this venv.
    # Keep notebook plots inline; only headless extraction children use Agg.
    return {**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONUNBUFFERED': '1',
            'TF_CPP_MIN_LOG_LEVEL': '2'}


def check_extraction_environment(python, log_path):
    # Capture native-library errors as well as Python tracebacks. Notebook output
    # can otherwise show only the parent's unhelpful CalledProcessError wrapper.
    probe = (
        "import importlib, sys\n"
        "print('Python:', sys.version, flush=True)\n"
        "for name in ('numpy', 'cv2', 'mediapipe'):\n"
        "    print('Importing ' + name, flush=True)\n"
        "    module = importlib.import_module(name)\n"
        "    print(name + ': ' + str(module.__version__), flush=True)\n"
    )
    result = subprocess.run([str(python), '-u', '-c', probe],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace',
                            env=_extraction_environment())
    output = result.stdout or '(No output from the dependency check.)'
    log_path.write_text(output, encoding='utf-8')
    print(output, flush=True)
    if result.returncode:
        raise RuntimeError(
            f'Extraction dependency check failed (exit {result.returncode}). '
            f'Diagnostics saved to {log_path}. Dataset files were not changed.\n'
            f'{output}'
        )


check_extraction_environment(ENV_PYTHON, PROJECT / 'environment-check.log')

from google.colab import drive
drive.mount('/content/drive')

mode = 'hands-pose' if INCLUDE_POSE else 'hands-only'
OUTPUT_ROOT = Path('/content/drive/MyDrive/sign-language-recognition/landmarks') / f'raw-v1-{mode}-{PAYLOAD_SHA256[:10]}'
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
WORK_DIR = Path('/content/asl-landmark-work')
WORK_DIR.mkdir(exist_ok=True)
print('Dataset:', DATASET_ROOT)
print('Persistent output:', OUTPUT_ROOT)
print('Extractor delegate: CPU. A T4 is not used by this configuration.')
print('Do not run a second VM against the same output folder.')


def _run_extraction(folder, extra, workers):
    command = [
        str(ENV_PYTHON), str(PROJECT / 'scripts/project.py'), 'extract-landmarks',
        '--dataset-root', str(DATASET_ROOT), '--output', str(folder),
        '--models', str(PROJECT / 'models'), '--work-dir', str(WORK_DIR),
        '--workers', str(workers), *extra,
    ]
    if not INCLUDE_POSE:
        command.append('--hands-only')
    log = PROJECT / 'native-extraction.log'
    print('Starting/resuming landmark extraction only. Native diagnostics:', log, flush=True)
    environment = _extraction_environment()
    with log.open('ab') as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors,
                                   text=True, bufsize=1, env=environment, start_new_session=True)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
            code = process.wait()
        except KeyboardInterrupt:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            process.wait()
            print('Stopped. Completed Drive shards remain resumable.')
            raise
        finally:
            process.stdout.close()
    if code:
        if (folder / 'summary.json').exists():
            print((folder / 'summary.json').read_text())
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 6000))
            print(stream.read().decode(errors='replace'))
        raise RuntimeError(f'Extractor exited with code {code}; inspect summary.json and failed video records before continuing.')
    summary = json.loads((folder / 'summary.json').read_text())
    if summary['newly_processed_clips'] and not summary['failed_clips']:
        seconds = summary['session_elapsed_seconds'] / summary['newly_processed_clips']
        print(f'Pilot/session rate: {seconds:.2f} seconds per newly processed clip.')
        if folder.name.startswith('pilot'):
            print(f'Illustrative full-corpus extrapolation: {seconds * 83399 / 3600:.1f} hours; small pilots are not a reliable full-run forecast.')
            print(f'Illustrative feature storage: {summary["feature_archive_bytes"] / summary["successful_clips"] * 83399 / 1024**3:.2f} GiB; ensure sufficient Drive quota.')
    return summary


def run_pilot(workers=WORKERS, clips=10):
    # Separate from full output: the pilot's limited manifest must never mix with full shards.
    return _run_extraction(OUTPUT_ROOT / f'pilot-w{workers}-n{clips}',
                           ['--splits', 'train', '--limit', str(clips), '--shard-size', '10'], workers)


def run_full_extraction(workers=WORKERS, max_shards=None, retry_failed=False):
    extra = ['--shard-size', str(SHARD_SIZE)]
    if max_shards is not None:
        extra += ['--max-shards', str(max_shards)]
    if retry_failed:
        extra.append('--retry-failed')
    return _run_extraction(OUTPUT_ROOT / 'full', extra, workers)


def inspect_pilot(folder=None, overlay_video=True):
    """Inspect saved coordinates, without loading MediaPipe into the notebook kernel."""
    import numpy as np
    import matplotlib.pyplot as plt
    folder = Path(folder) if folder is not None else OUTPUT_ROOT / f'pilot-w{WORKERS}-n10'
    schema = json.loads((folder / 'feature-schema.json').read_text())
    for path in sorted(folder.glob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            records = json.loads(archive.read('records.json'))['records']
            good = next((r for r in records if r['status'] == 'ok' and r['frames_with_hands']), None)
            if good is None:
                continue
            with np.load(io.BytesIO(archive.read(good['member'])), allow_pickle=False) as arrays:
                frame = int(np.argmax(arrays['hand_present'].sum(axis=1)))
                height, width = good['frame_shape'][:2]
                fig, ax = plt.subplots(figsize=(8, 6))
                if overlay_video:
                    try:
                        import cv2
                        capture = cv2.VideoCapture(str(DATASET_ROOT / 'videos' / good['video_file']))
                        try:
                            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                            ok, pixels = capture.read()
                        finally:
                            capture.release()
                        if ok:
                            ax.imshow(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
                        else:
                            print('Source frame unavailable; showing landmarks only.')
                    except ImportError:
                        print('Notebook OpenCV unavailable; use overlay_video=False for skeleton-only review.')
                pose = arrays['pose_xyz'][frame, :, :2] * [width, height]
                mask = arrays['pose_valid_mask'][frame]
                for a, b in schema['upper_pose_connections']:
                    if mask[a] and mask[b]:
                        ax.plot(pose[[a,b],0], pose[[a,b],1], color='gray')
                for hand in range(2):
                    if arrays['hand_present'][frame, hand]:
                        points = arrays['hands_xyz'][frame, hand, :, :2] * [width, height]
                        for a, b in schema['hand_connections']:
                            ax.plot(points[[a,b],0], points[[a,b],1])
                        ax.scatter(points[:,0], points[:,1], s=12)
                ax.set(xlim=(0,width), ylim=(height,0), title=f"Dataset label: {good['gloss']} | frame {frame} | detector landmarks")
                ax.set_aspect('equal')
                plt.show()
                print('Clip:', good['video_file'], '| frames:', good['frames'],
                      '| frames with hands:', good['frames_with_hands'])
                print('Hands are saved in detection order. Training must canonicalize hand slots.')
                return good
    raise RuntimeError('No pilot clip with hand detections; inspect failures/tracking before the full run.')


pilot_result = run_pilot()
print('Pilot result:', json.dumps(pilot_result, indent=2))
print('After reviewing pilot tracking and failures, run: result = run_full_extraction()')
print('To view saved pilot skeletons first: inspect_pilot()')
print('Re-run the SAME setup cell after VM replacement, once raw data is available again, to resume.')
print('No GRU training is implemented or started by these cells.')
