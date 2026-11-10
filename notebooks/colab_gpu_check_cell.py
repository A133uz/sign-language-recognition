"""Copy this entire file into a new cell AFTER the extraction setup/pilot."""

import csv
import os
from pathlib import Path
import subprocess

if not all(name in globals() for name in ('ENV_PYTHON', 'PROJECT', 'DATASET_ROOT')):
    raise RuntimeError('Run the existing extraction setup cell in this notebook first.')

gpu_check_models = {
    'hands': Path(PROJECT) / 'models/hand_landmarker.task',
    'pose': Path(PROJECT) / 'models/pose_landmarker_lite.task',
}
for path in (Path(ENV_PYTHON), *gpu_check_models.values()):
    if not path.is_file():
        raise FileNotFoundError(f'Missing setup file: {path}; finish the existing pilot first.')
with (Path(DATASET_ROOT) / 'splits/train.csv').open(encoding='utf-8-sig', newline='') as stream:
    gpu_check_video = Path(DATASET_ROOT) / 'videos' / next(csv.DictReader(stream))['Video file']
if not gpu_check_video.is_file():
    raise FileNotFoundError(gpu_check_video)

try:
    info = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=15,
    )
    print('NVIDIA inventory (not proof of MediaPipe GPU use):\n' + info.stdout, flush=True)
except (OSError, subprocess.TimeoutExpired) as error:
    print('NVIDIA inventory unavailable:', error, flush=True)

gpu_check_program = r'''
import json, sys, time
import cv2
import mediapipe as mp

kind, model, video = sys.argv[1:]
print('MediaPipe:', mp.__version__, 'task:', kind, 'requested delegate: GPU', flush=True)
base = mp.tasks.BaseOptions(model_asset_path=model, delegate=mp.tasks.BaseOptions.Delegate.GPU)
vision = mp.tasks.vision
if kind == 'hands':
    options = vision.HandLandmarkerOptions(
        base_options=base, running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=.5, min_hand_presence_confidence=.5,
        min_tracking_confidence=.5,
    )
    tracker_class = vision.HandLandmarker
else:
    options = vision.PoseLandmarkerOptions(
        base_options=base, running_mode=vision.RunningMode.VIDEO, num_poses=1,
        output_segmentation_masks=False, min_pose_detection_confidence=.5,
        min_pose_presence_confidence=.5, min_tracking_confidence=.5,
    )
    tracker_class = vision.PoseLandmarker
capture = cv2.VideoCapture(video)
try:
    if not capture.isOpened():
        raise RuntimeError('Cannot open diagnostic dataset video')
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) // 2))
    frames, detections = 0, 0
    began = time.perf_counter()
    with tracker_class.create_from_options(options) as tracker:
        for index in range(8):
            ok, frame = capture.read()
            if not ok:
                break
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = tracker.detect_for_video(image, index * 100)
            frames += 1
            detections += len(result.hand_landmarks if kind == 'hands' else result.pose_landmarks)
    if not frames:
        raise RuntimeError('No diagnostic frames decoded')
    print('PROBE_RESULT=' + json.dumps({
        'task': kind, 'requested_delegate': 'GPU', 'frames': frames,
        'detections': detections, 'initialization_and_inference_seconds': time.perf_counter() - began,
    }), flush=True)
finally:
    capture.release()
'''

# Override the inline backend only for children. No CPU retry or training.
gpu_check_env = {**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONUNBUFFERED': '1',
                 'TF_CPP_MIN_LOG_LEVEL': '0', 'GLOG_minloglevel': '0'}
gpu_check_results = {}
for task, model in gpu_check_models.items():
    print(f'\n--- {task.upper()}: fresh process, explicit GPU request ---', flush=True)
    try:
        probe = subprocess.run(
            [str(ENV_PYTHON), '-u', '-c', gpu_check_program, task, str(model), str(gpu_check_video)],
            env=gpu_check_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', timeout=120,
        )
        log = probe.stdout or '(No native diagnostic output.)'
        result_line = next((line for line in log.splitlines() if line.startswith('PROBE_RESULT=')), None)
        passed = probe.returncode == 0 and result_line is not None
        gpu_check_results[task] = {'status': 'GPU_REQUEST_PASS' if passed else 'FAIL',
                                  'exit_code': probe.returncode, 'log': log}
        print(log, flush=True)
        print(task.upper() + ': ' + gpu_check_results[task]['status'], flush=True)
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b''
        log = output.decode(errors='replace') if isinstance(output, bytes) else output
        gpu_check_results[task] = {'status': 'TIMEOUT', 'log': log}
        print(log, flush=True)
        print(task.upper() + ': TIMEOUT after 120 seconds', flush=True)

print('\nSUMMARY:', {name: item['status'] for name, item in gpu_check_results.items()})
print('PASS means the GPU request and video inference completed; inspect native GL/EGL/delegate logs for the actual renderer.')
print('A software renderer or CPU fallback does not establish T4 acceleration. This is not a throughput benchmark.')
print('The main extractor is unchanged and still selects CPU. Paste the output here before changing the full run.')
