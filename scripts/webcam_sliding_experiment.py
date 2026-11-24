"""Experimental ASL sliding window; predictions are raw and can be wrong at rest."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import cv2
import torch

from sign_language.dataset import write_json
from sign_language.hands import draw_hands
from sign_language.streaming_face import RollingObservations, StreamingFaceTracker, score_window
from webcam_language_switch import load_recognizer


def run(recognizer, report_path, camera=0, window_seconds=2.0, interval_seconds=.5):
    if not .5 <= window_seconds <= 5 or not .1 <= interval_seconds <= 5:
        raise ValueError('Window must be 0.5–5s and interval 0.1–5s')
    buffer = RollingObservations(round(window_seconds * 1000), max_items=300)
    capture = cv2.VideoCapture(camera)
    title = 'EXPERIMENTAL ASL - raw sliding predictions'
    report = {'status': 'starting', 'checkpoint_sha256': recognizer.checkpoint_sha256,
              'epoch': recognizer.epoch, 'window_seconds': window_seconds,
              'interval_seconds': interval_seconds, 'video_saved': False,
              'landmarks_saved': False, 'processed_frames': 0, 'scored_windows': 0,
              'rejected_windows': 0, 'stale_results': 0, 'inference_seconds_total': 0.0,
              'tracking_seconds_total': 0.0}
    pool = ThreadPoolExecutor(max_workers=1)
    job = None
    started = time.monotonic()
    next_update = started + window_seconds
    result_until = 0.0
    subtitle = ''
    status = 'Collecting initial window'
    last_ms = -1
    try:
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open camera {camera}')
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_FPS, 30)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        report['status'] = 'running'
        with StreamingFaceTracker(recognizer) as tracker:
            print('WINDOW_READY | automatic ASL | raw closed-set guesses | Q closes', flush=True)
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError('Camera stopped returning frames')
                now = time.monotonic()
                timestamp_ms = int((now - started) * 1000)
                # Do not fabricate timestamps or submit a duplicate timestamp.
                if timestamp_ms <= last_ms:
                    continue
                last_ms = timestamp_ms
                tracking_start = time.perf_counter()
                body, face, present, blend, hands = tracker.observe(frame, timestamp_ms)
                report['tracking_seconds_total'] += time.perf_counter() - tracking_start
                report['processed_frames'] += 1
                buffer.append(timestamp_ms, body, face, present, blend, frame.shape)
                now = time.monotonic()
                if job is not None and job.done():
                    try:
                        result = job.result()
                        report['scored_windows'] += 1
                        report['inference_seconds_total'] += result['inference_seconds']
                        age = now - started - result['window_end_ms'] / 1000
                        if age <= 2.0:
                            subtitle = result['label']
                            result_until = now + max(1.0, interval_seconds * 2)
                            status = 'RAW GUESS - NOT AN ACTIVITY DETECTOR'
                        else:
                            report['stale_results'] += 1
                    except ValueError as exc:
                        report['rejected_windows'] += 1
                        subtitle, status = '', str(exc)
                    job = None
                if subtitle and now >= result_until:
                    subtitle, status = '', 'Waiting for next result'
                if now >= next_update:
                    next_update = now + interval_seconds
                    if job is None and len(buffer.items) >= 8:
                        snapshot = buffer.snapshot()
                        # Wait for a substantially filled window; this is not a sign boundary.
                        if snapshot[0][-1] - snapshot[0][0] >= window_seconds * 800:
                            job = pool.submit(score_window, recognizer, snapshot)
                view = cv2.flip(draw_hands(frame, hands), 1)
                cv2.rectangle(view, (0, 0), (view.shape[1], 56), (15, 15, 15), -1)
                cv2.putText(view, 'EXPERIMENTAL ASL | Q: quit', (10, 21),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(view, status[:65], (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
                            .43, (70, 220, 255), 1, cv2.LINE_AA)
                if subtitle:
                    from webcam_700_subtitles import draw_subtitle
                    view = draw_subtitle(view, subtitle)
                cv2.imshow(title, view)
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord('q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
        report.update(status='complete', reason='user_closed')
    except BaseException as exc:
        report.update(status='failed', reason=str(exc))
        raise
    finally:
        capture.release()
        cv2.destroyAllWindows()
        pool.shutdown(wait=True, cancel_futures=True)
        report['elapsed_seconds'] = time.monotonic() - started
        write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--window-seconds', type=float, default=2.0)
    parser.add_argument('--interval-seconds', type=float, default=.5)
    args = parser.parse_args()
    torch.set_num_threads(2)
    recognizer = load_recognizer('ASL', args.device)
    report = ROOT / '.cache' / f'sliding-asl-{datetime.now():%Y%m%d-%H%M%S-%f}.json'
    run(recognizer, report, args.camera, args.window_seconds, args.interval_seconds)


if __name__ == '__main__':
    main()
