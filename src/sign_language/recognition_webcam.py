"""Interactive isolated-sign capture; frames/landmarks remain in memory only."""

from concurrent.futures import ThreadPoolExecutor
import time

import cv2
import numpy as np

from sign_language.dataset import write_json
from sign_language.hands import HandTracker, draw_hands


class SkeletonPreview:
    """One worker owns its tracker; preview never blocks raw camera capture."""
    def __init__(self, model):
        self.model, self.tracker = model, None

    def draw(self, frame, timestamp):
        if self.tracker is None:
            self.tracker = HandTracker(self.model)
        result = self.tracker.detect(frame, timestamp)
        return draw_hands(frame, result), len(result.hand_landmarks)

    def close(self):
        if self.tracker:
            self.tracker.close()


def model_display_name(recognizer):
    """Describe the actual loaded class, rather than a default or filename."""
    name = type(getattr(recognizer, 'model', None)).__name__
    return {'SignSTGCN': 'ST-GCN', 'SignGRU': 'GRU'}.get(name, 'Unknown model')


def state_headline(state, now, deadline, capture_started, model_name):
    if state == 'countdown':
        return f'GET READY: {max(1, int(deadline-now)+1)}'
    if state == 'recording':
        return f'SIGN NOW - Space to finish ({now-capture_started:.1f}s)'
    if state == 'processing':
        return f'PROCESSING hands + body, then {model_name}...'
    return 'READY - Space, wait for SIGN NOW, perform ONE sign'


def run_recognition(recognizer, report_path, camera=0, max_seconds=8.0):
    if not 1 <= max_seconds <= 10:
        raise ValueError('Capture limit must be between one and ten seconds')
    model_name = model_display_name(recognizer)
    session_id = report_path.stem
    report = {'status': 'starting', 'checkpoint': str(recognizer.checkpoint),
              'checkpoint_sha256': recognizer.checkpoint_sha256, 'epoch': recognizer.epoch,
              'classes': len(recognizer.names), 'device': str(recognizer.device),
              'architecture': getattr(recognizer, 'config', {}).get('architecture', 'gru'),
              'loaded_model_class': type(getattr(recognizer, 'model', None)).__name__,
              'display_model': model_name, 'session_id': session_id,
              'media_saved': False, 'coordinates_saved': False, 'training_performed': False,
              'trials': []}
    write_json(report_path, report)
    capture = cv2.VideoCapture(camera)
    title = f'{model_name} webcam | epoch {recognizer.epoch} | {session_id}'
    pool = ThreadPoolExecutor(max_workers=1)
    preview = SkeletonPreview(recognizer.hand_model)
    preview_job = prediction_job = None
    frames, timestamps = [], []
    state, deadline, note, result, capture_started = 'ready', 0., '', None, 0.
    started, last_timestamp = time.monotonic(), -1
    preview_frame, hands = None, 0
    reason = 'user_closed'
    try:
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open camera {camera}')
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_FPS, 30)
        report.update(status='running', camera_open=True)
        write_json(report_path, report)
        print(f'WINDOW_READY | {title} | model={model_name} | Space: start/stop | Q: exit', flush=True)
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError('Camera stopped returning frames')
            now = time.monotonic()
            timestamp = max(last_timestamp+1, round((now-started)*1000))
            last_timestamp = timestamp
            if preview_job is not None and preview_job.done():
                preview_frame, hands = preview_job.result()
                preview_job = None
            if prediction_job is not None and prediction_job.done():
                try:
                    result = prediction_job.result()
                    result['capture_id'] = len(report['trials'])+1
                    report['trials'].append(result)
                    note = 'Top five model scores; not calibrated confidence'
                    print('PREDICTION:', result, flush=True)
                except Exception as exc:
                    note = f'Capture failed: {exc}'
                    report['trials'].append({'capture_id': len(report['trials'])+1, 'error': str(exc)})
                    result = None
                prediction_job, state = None, 'ready'
                write_json(report_path, report)
            if preview_job is None and prediction_job is None:
                preview_job = pool.submit(preview.draw, frame.copy(), timestamp)
            if state == 'countdown' and now >= deadline:
                state, frames, timestamps = 'recording', [], []
                capture_started = now
            if state == 'recording':
                frames.append(frame.copy())
                timestamps.append(round((now-capture_started)*1000))
                if len(timestamps) > 1:
                    timestamps[-1] = max(timestamps[-1], timestamps[-2]+1)
            # Display only is mirrored. All classifier/MediaPipe inputs remain unmirrored.
            image = frame if prediction_job is not None or preview_frame is None else preview_frame
            view_image = cv2.flip(image, 1)
            h, w = frame.shape[:2]
            view = np.full((h+250, w, 3), 22, dtype=np.uint8)
            view[250:] = view_image
            headline = state_headline(state, now, deadline, capture_started, model_name)
            capture_number = result['capture_id'] if result else len(report['trials'])+1
            lines = [f'MODEL: {model_name} | epoch {recognizer.epoch} | {str(recognizer.device).upper()} | {len(recognizer.names):,} labels',
                     headline, f'Capture #{capture_number} | Space: start/stop | R: clear | Q: close',
                     f'{session_id} | uncalibrated scores | hands {hands}/2']
            if result:
                lines += [f'{i+1}. {item["label"]}   {item["score"]*100:.1f}%'
                          for i, item in enumerate(result['top5'])]
                quality = result['quality']
                lines += [f'Frames: {result["captured_frames"]} | hand frames: {quality["frames_with_hands"]}'
                          f' | shoulder frames: {quality["frames_with_body_anchor"]}']
            elif note:
                lines += [note[:85]]
            else:
                lines += ['Keep face, shoulders and both hands visible.',
                          'Frames stay in memory; only predictions/timing saved.']
            for row, text in enumerate(lines):
                cv2.putText(view, text, (10, 22+row*22), cv2.FONT_HERSHEY_SIMPLEX,
                            .47, (80, 255, 120) if row < 4 else (245, 235, 180), 1, cv2.LINE_AA)
            cv2.imshow(title, view)
            key = cv2.waitKey(1) & 0xff
            if key in (27, ord('q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('r') and state != 'processing':
                state, frames, timestamps, result, note = 'ready', [], [], None, ''
            elif key == ord(' ') and state == 'ready':
                state, deadline, result, note = 'countdown', now+1.5, None, ''
            elif state == 'recording' and (key == ord(' ') or now-capture_started >= max_seconds):
                if len(frames) < 8:
                    note = 'Too short: capture at least eight frames'
                    state, frames, timestamps = 'ready', [], []
                else:
                    prediction_job = pool.submit(recognizer.predict_frames, frames, timestamps)
                    state, frames, timestamps = 'processing', [], []
        report.update(status='complete', reason=reason)
    except Exception as exc:
        report.update(status='failed', reason=str(exc))
        raise
    finally:
        capture.release()
        cv2.destroyAllWindows()
        # Finish any submitted inference before disposing of trackers; no captures are persisted.
        pool.submit(preview.close)
        pool.shutdown(wait=True)
        if prediction_job is not None and prediction_job.done():
            try:
                value = prediction_job.result()
                value['capture_id'] = len(report['trials'])+1
                report['trials'].append(value)
            except Exception as exc:
                report['trials'].append({'capture_id': len(report['trials'])+1, 'error': str(exc)})
        report['camera_open'] = False
        report['elapsed_seconds'] = time.monotonic()-started
        write_json(report_path, report)
    return report
