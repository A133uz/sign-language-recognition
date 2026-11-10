"""Six prompted sign trials; compare timing/orientation without saving media."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import cv2
import numpy as np
import torch

from sign_language.dataset import write_json
from sign_language.recognition import SignRecognizer
from sign_language.recognition_diagnostics import compare_capture
from sign_language.recognition_webcam import SkeletonPreview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Choose a new report filename')
    torch.set_num_threads(2)
    recognizer = SignRecognizer(ROOT/'models/asl-gru-128-cv-regularized-v1/official-refit/best.pt', ROOT/'models')
    prompts = ['HELLO', 'THANKYOU', 'NO']*2
    report = {'status': 'starting', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'checkpoint_sha256': recognizer.checkpoint_sha256, 'epoch': recognizer.epoch,
              'classes': len(recognizer.names), 'training_performed': False,
              'media_saved': False, 'coordinates_saved': False, 'user_completion_confirmed': None,
              'capture_seconds': 3., 'prompts': prompts, 'trials': []}
    write_json(args.report, report)
    capture = cv2.VideoCapture(args.camera)
    pool = ThreadPoolExecutor(max_workers=1)
    preview = SkeletonPreview(recognizer.hand_model)
    preview_job = prediction_job = None
    annotated = None
    index, state, deadline = 0, 'ready', 0.
    frames, timestamps, last_result = [], [], None
    started, last_timestamp, hands = time.monotonic(), -1, 0
    title = 'ASL diagnosis - timing and orientation - Q closes'
    try:
        if not capture.isOpened():
            raise RuntimeError('Cannot open camera')
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_FPS, 30)
        report['status'] = 'running'
        write_json(args.report, report)
        print('GUIDED_WINDOW_READY | Space for each prompted sign; automatic 3-second capture; Q closes', flush=True)
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('Camera read failed')
            now = time.monotonic()
            timestamp = max(last_timestamp+1, round((now-started)*1000))
            last_timestamp = timestamp
            if preview_job is not None and preview_job.done():
                annotated, hands = preview_job.result()
                preview_job = None
            if prediction_job is not None and prediction_job.done():
                try:
                    last_result = prediction_job.result()
                    report['trials'].append(last_result)
                    print('TRIAL_COMPLETE', index+1, last_result['expected'],
                          {key: value['top5'][0]['label'] for key,value in last_result['variants'].items()}, flush=True)
                except Exception as exc:
                    report['trials'].append({'expected': prompts[index], 'error': str(exc)})
                    print('TRIAL_ERROR', str(exc), flush=True)
                index += 1
                prediction_job, state = None, 'done' if index == len(prompts) else 'ready'
                write_json(args.report, report)
            if preview_job is None and prediction_job is None:
                preview_job = pool.submit(preview.draw, frame.copy(), timestamp)
            if state == 'countdown' and now >= deadline:
                state, deadline, frames, timestamps = 'recording', now+3., [], []
                capture_start = now
            if state == 'recording':
                frames.append(frame.copy())
                timestamps.append(max(timestamps[-1]+1 if timestamps else 0, round((now-capture_start)*1000)))
                if now >= deadline:
                    prediction_job = pool.submit(compare_capture, recognizer, frames, timestamps, prompts[index])
                    state, frames, timestamps = 'processing', [], []
            image = frame if prediction_job is not None or annotated is None else annotated
            view = np.full((frame.shape[0]+300, frame.shape[1], 3), 22, np.uint8)
            view[300:] = cv2.flip(image, 1)
            name = prompts[index] if index < len(prompts) else 'DONE'
            if state == 'recording':
                headline = f'SIGN NOW: {name} - {max(0,deadline-now):.1f}s remaining'
            elif state == 'countdown':
                headline = f'GET READY: {name} - {deadline-now:.1f}s'
            elif state == 'processing':
                headline = 'Comparing same capture; please wait...'
            elif state == 'done':
                headline = 'ALL SIX COMPLETE - Q closes'
            else:
                headline = f'Trial {index+1}/6: {name} - Space when ready'
            lines = [headline, 'Perform ONCE during SIGN NOW; stop is automatic.',
                     'Keep shoulders/hands visible. No keyboard during capture.',
                     f'Hands: {hands}/2 | full 2,731-label GRU | no media saved']
            if last_result:
                lines.append(f'Previous prompt: {last_result["expected"]}')
                for key, value in last_result['variants'].items():
                    lines.append(f'{key}: {value["top5"][0]["label"]} | target rank >= {value["expected_rank_min"]}')
            for row, line in enumerate(lines):
                cv2.putText(view, line, (10, 23+row*26), cv2.FONT_HERSHEY_SIMPLEX,
                            .49, (100, 255, 160), 1, cv2.LINE_AA)
            cv2.imshow(title, view)
            key = cv2.waitKey(1) & 0xff
            if key in (27, ord('q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(' ') and state == 'ready':
                state, deadline = 'countdown', now+1.5
        report['status'] = 'complete' if index == len(prompts) else 'partial'
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        capture.release()
        cv2.destroyAllWindows()
        pool.submit(preview.close)
        pool.shutdown(wait=True)
        if prediction_job is not None:
            try:
                report['trials'].append(prediction_job.result())
            except Exception as exc:
                report['trials'].append({'error': str(exc)})
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(args.report, report)
    print('GUIDED_FINISHED', report['status'], len(report['trials']), flush=True)


if __name__ == '__main__':
    main()
