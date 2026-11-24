"""Collect consented coarse-labeled hand/pose landmarks locally; no video or training."""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import cv2
import torch

from sign_language.activity_collection import SCHEMA, save_trial, validate_trial
from sign_language.dataset import write_json
from sign_language.hands import draw_hands
from sign_language.streaming_face import StreamingFaceTracker
from webcam_language_switch import load_recognizer

PROMPTS = [
    ('REST_VISIBLE', 'rest', 'Rest hands visible; no signing'),
    ('MOVE', 'unrelated_movement', 'Move hands naturally; do not sign'),
    ('HELLO', 'known_sign', 'Sign HELLO once within 3 seconds, then relax'),
    ('YES', 'known_sign', 'Sign YES once within 3 seconds, then relax'),
    ('HOPE', 'unknown_sign', 'Sign HOPE once within 3 seconds, then relax'),
    ('REST_BELOW', 'rest', 'Keep both hands below the camera frame'),
    ('THANKYOU', 'known_sign', 'Sign THANK YOU once within 3 seconds, then relax'),
    ('MOVE', 'unrelated_movement', 'Adjust hands or clothing; do not sign'),
    ('HELLO', 'known_sign', 'Sign HELLO slowly once, then relax'),
    ('HOPE', 'unknown_sign', 'Sign HOPE once within 3 seconds, then relax'),
    ('YES', 'known_sign', 'Sign YES slowly once, then relax'),
    ('THANKYOU', 'known_sign', 'Sign THANK YOU slowly once, then relax'),
    ('HOLD_MY', 'held_pose', 'Form MY pose during GET READY; hold it'),
    ('REST_VISIBLE', 'rest', 'Rest hands visible; no signing'),
]


def run(recognizer, destination, camera=0, duration=7.0):
    if not 3 <= duration <= 15:
        raise ValueError('Duration must be 3–15 seconds')
    destination = Path(destination)
    # The directory is unique and must be created by the caller before capture.
    if not destination.is_dir() or any(destination.iterdir()):
        raise ValueError('Destination must be a new, empty directory')
    manifest_path = destination / 'session.json'
    manifest = {'schema': SCHEMA, 'status': 'starting', 'prompt_confirmation': 'pending',
                'video_saved': False, 'face_mesh_saved': False, 'body_landmarks_saved': True,
                'labels_are_prompt_level_only': True, 'sign_boundaries_annotated': False,
                'signer_count': 1, 'checkpoint_sha256': recognizer.checkpoint_sha256,
                'trial_seconds': duration, 'trials': []}
    cap = cv2.VideoCapture(camera)
    title = 'LANDMARK COLLECTION - local only'
    observations, timestamps, trial_shape = [], [], None
    recording_start = None
    state, index = 'ready', 0
    start = time.monotonic()
    last_stamp = -1
    try:
        if not cap.isOpened(): raise RuntimeError(f'Cannot open camera {camera}')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        manifest['status'] = 'running'
        write_json(manifest_path, manifest)
        with StreamingFaceTracker(recognizer) as tracker:
            print(f'WINDOW_READY | {len(PROMPTS)} trials | local hand/pose landmarks | SPACE starts', flush=True)
            while index < len(PROMPTS):
                ok, frame = cap.read()
                if not ok or frame is None: raise RuntimeError('Camera stopped returning frames')
                now = time.monotonic()
                stamp = int((now - start) * 1000)
                if stamp <= last_stamp: continue
                last_stamp = stamp
                body, _face, _present, _blend, hands = tracker.observe(frame, stamp)
                now = time.monotonic()
                if state == 'countdown' and now >= deadline:
                    state, recording_start = 'recording', now
                    observations, timestamps, trial_shape = [], [], frame.shape
                    first_stamp = None
                if state == 'recording':
                    if now - recording_start < duration:
                        if tuple(frame.shape) != tuple(trial_shape):
                            raise ValueError('Camera dimensions changed during capture')
                        if first_stamp is None: first_stamp = stamp
                        observations.append(body)
                        timestamps.append(stamp - first_stamp)
                    else:
                        if len(observations) < 8:
                            raise ValueError('Too few observations; incomplete trial not saved')
                        prompt, category, _ = PROMPTS[index]
                        metadata = save_trial(destination, index+1, prompt, category,
                                              observations, timestamps, trial_shape)
                        validate_trial(destination, metadata)
                        manifest['trials'].append(metadata)
                        write_json(manifest_path, manifest)
                        observations, timestamps = [], []
                        index += 1
                        state = 'ready'
                        if index >= len(PROMPTS): break
                view = cv2.flip(draw_hands(frame, hands), 1)
                cv2.rectangle(view, (0, 0), (view.shape[1], 119), (16, 16, 16), -1)
                headline = ('GET READY' if state == 'countdown' else
                            f'SIGN NOW: {max(0, duration-(now-recording_start)):.1f}s' if state == 'recording'
                            else 'SPACE starts the next trial')
                lines = [f'TRIAL {index+1}/{len(PROMPTS)}: {PROMPTS[index][0]}',
                         PROMPTS[index][2], headline, 'No video saved | Q stops collection']
                for row, text in enumerate(lines):
                    cv2.putText(view, text, (8, 24+row*27), cv2.FONT_HERSHEY_SIMPLEX,
                                .47, (40, 230, 255) if row == 2 else (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(title, view)
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord('q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    manifest.update(status='interrupted', reason='user_closed',
                                    incomplete_trial_discarded=state == 'recording')
                    break
                if key == ord(' ') and state == 'ready':
                    state, deadline = 'countdown', now + 2.0
            if index >= len(PROMPTS): manifest.update(status='complete', reason='all_trials_completed')
    except BaseException as exc:
        manifest.update(status='failed', reason=type(exc).__name__,
                        incomplete_trial_discarded=state == 'recording')
        raise
    finally:
        observations, timestamps = [], []
        cap.release()
        cv2.destroyAllWindows()
        manifest['elapsed_seconds'] = time.monotonic() - start
        write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    recognizer = load_recognizer('ASL', args.device)
    session_id = f'{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}'
    destination = ROOT / '.cache' / 'sign-activity-recordings' / session_id
    destination.mkdir(parents=True, exist_ok=False)
    print('SESSION', destination, flush=True)
    result = run(recognizer, destination, args.camera)
    print('FINISHED', result['status'], len(result['trials']), flush=True)


if __name__ == '__main__': main()
