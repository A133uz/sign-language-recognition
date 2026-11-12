"""Interactive isolated-sign capture; frames/landmarks remain in memory only."""

from concurrent.futures import ThreadPoolExecutor
import time

import cv2
import numpy as np

from sign_language.dataset import write_json
from sign_language.hands import HandTracker, draw_hands
from sign_language.frame_rate import FrameRateLimiter, capture_rate_metrics


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
    return {'SignSTGCN': 'ST-GCN', 'SignGRU': 'GRU', 'SignMeanPoolGRU': 'GRU mean-pool',
            'SignGeometryMeanPoolGRU': 'GRU mean-pool + geometry',
            'SignTrajectoryMeanPoolGRU': 'GRU mean-pool + hand/path geometry',
            'SignTrajectoryReadoutGRU': 'Frozen GRU + trajectory readout'}.get(name, 'Unknown model')


def state_headline(state, now, deadline, capture_started, model_name, auto_finish=False, max_seconds=8.):
    if state == 'countdown':
        return f'GET READY: {max(1, int(deadline-now)+1)}'
    if state == 'recording':
        if auto_finish:
            return f'SIGN NOW - auto stop in {max(0.,max_seconds-now+capture_started):.1f}s'
        return f'SIGN NOW - Space to finish ({now-capture_started:.1f}s)'
    if state == 'processing':
        return f'PROCESSING hands + body, then {model_name}...'
    return 'READY - Space, wait for SIGN NOW, perform ONE sign'


def _predict_capture(recognizer, frames, timestamps, metrics):
    return {**recognizer.predict_frames(frames, timestamps), **metrics}


def _write_report(path, report, recognizer):
    images=getattr(recognizer,'saved_still_images',None)
    if images is not None:
        report.update(saved_still_images=list(images),media_saved=bool(images),video_saved=False)
    write_json(path,report)


def run_recognition(recognizer, report_path, camera=0, max_seconds=8.0, capture_fps=30., expected_sign=None, auto_finish=False):
    if not 1 <= max_seconds <= 10:
        raise ValueError('Capture limit must be between one and ten seconds')
    limiter = FrameRateLimiter(capture_fps)
    if expected_sign is not None and expected_sign not in recognizer.names:
        raise ValueError('Expected sign is not a checkpoint label')
    model_name = model_display_name(recognizer)
    session_id = report_path.stem
    report = {'status': 'starting', 'checkpoint': str(recognizer.checkpoint),
              'checkpoint_sha256': recognizer.checkpoint_sha256, 'epoch': recognizer.epoch,
              'classes': len(recognizer.names), 'device': str(recognizer.device),
              'architecture': getattr(recognizer, 'config', {}).get('architecture', 'gru'),
              'loaded_model_class': type(getattr(recognizer, 'model', None)).__name__,
              'display_model': model_name, 'session_id': session_id,
              'capture_fps_limit': float(capture_fps),
              'capture_clock': 'perf_counter',
              'capture_clock_resolution_seconds': time.get_clock_info('perf_counter').resolution,
              'expected_label': expected_sign,
              'max_capture_seconds':max_seconds,'automatic_capture_finish':auto_finish,
              'media_saved': False, 'coordinates_saved': False, 'training_performed': False,
              'trials': []}
    _write_report(report_path,report,recognizer)
    capture = cv2.VideoCapture(camera)
    title = f'{model_name} webcam | epoch {recognizer.epoch} | {session_id}'
    pool = ThreadPoolExecutor(max_workers=1)
    preview = SkeletonPreview(recognizer.hand_model)
    preview_job = prediction_job = None
    frames, timestamps = [], []
    state, deadline, note, result, capture_started = 'ready', 0., '', None, 0.
    source_frames = 0
    started, last_timestamp = time.perf_counter(), -1
    preview_frame, hands = None, 0
    reason = 'user_closed'
    try:
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open camera {camera}')
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_FPS, capture_fps)
        device_fps = float(capture.get(cv2.CAP_PROP_FPS))
        report['camera_reported_fps'] = device_fps if np.isfinite(device_fps) and device_fps > 0 else None
        report.update(status='running', camera_open=True)
        _write_report(report_path,report,recognizer)
        print(f'WINDOW_READY | {title} | model={model_name} | Space: start/stop | Q: exit', flush=True)
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError('Camera stopped returning frames')
            now = time.perf_counter()
            timestamp = max(last_timestamp+1, round((now-started)*1000))
            last_timestamp = timestamp
            if preview_job is not None and preview_job.done():
                preview_frame, hands = preview_job.result()
                preview_job = None
            if prediction_job is not None and prediction_job.done():
                try:
                    result = prediction_job.result()
                    result['capture_id'] = len(report['trials'])+1
                    if expected_sign is not None:result['expected_label']=expected_sign
                    report['trials'].append(result)
                    note = 'Top five model scores; not calibrated confidence'
                    print('PREDICTION:', result, flush=True)
                except Exception as exc:
                    note = f'Capture failed: {exc}'
                    report['trials'].append({'capture_id': len(report['trials'])+1, 'error': str(exc)})
                    result = None
                prediction_job, state = None, 'ready'
                _write_report(report_path,report,recognizer)
            if preview_job is None and prediction_job is None:
                preview_job = pool.submit(preview.draw, frame.copy(), timestamp)
            if state == 'countdown' and now >= deadline:
                state, frames, timestamps = 'recording', [], []
                capture_started = now
                source_frames = 0
                limiter = FrameRateLimiter(capture_fps)
            if state == 'recording':
                source_frames += 1
                if limiter.accept(now-capture_started):
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
            headline = state_headline(state, now, deadline, capture_started, model_name,auto_finish,max_seconds)
            if expected_sign is not None:headline+=f' | sign {expected_sign}'
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
                measured = result.get('effective_capture_fps')
                if measured is not None:
                    lines += [f'Capture: {measured:.1f} FPS | limit {capture_fps:g} | skipped {result["fps_limit_dropped_frames"]}']
            elif note:
                lines += [note[:85]]
            else:
                lines += ['Keep face, shoulders and both hands visible.',
                          f'Capture target: {capture_fps:g} FPS; slower cameras are not upsampled.',
                          'Two annotated stills saved for review; no video.' if getattr(recognizer,'review_stills',False)
                          else 'Frames stay in memory; only predictions/timing saved.']
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
            elif state == 'recording' and ((key == ord(' ') and not auto_finish) or now-capture_started >= max_seconds):
                if len(frames) < 8:
                    note = 'Too short: capture at least eight frames'
                    state, frames, timestamps = 'ready', [], []
                else:
                    metrics = capture_rate_metrics(timestamps, source_frames, now-capture_started, capture_fps)
                    metrics.update(capture_stop_reason='time_limit' if now-capture_started >= max_seconds else 'manual',
                                   capture_wall_seconds=now-capture_started)
                    prediction_job = pool.submit(_predict_capture, recognizer, frames, timestamps, metrics)
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
                if expected_sign is not None:value['expected_label']=expected_sign
                report['trials'].append(value)
            except Exception as exc:
                report['trials'].append({'capture_id': len(report['trials'])+1, 'error': str(exc)})
        report['camera_open'] = False
        report['elapsed_seconds'] = time.perf_counter()-started
        _write_report(report_path,report,recognizer)
    return report
