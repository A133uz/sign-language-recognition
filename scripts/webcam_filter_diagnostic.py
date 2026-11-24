"""Guided matched comparison of provisional sliding-window filters."""

import argparse
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
from sign_language.sliding_filter_diagnostics import CandidateFilters
from sign_language.streaming_face import RollingObservations, StreamingFaceTracker, score_window
from webcam_language_switch import load_recognizer

PROMPTS = [
    ('REST', 'Keep hands relaxed and visible; no sign'),
    ('MOVE', 'Move hands naturally; do not sign'),
    ('HELLO', 'Sign HELLO once in first 3 seconds, then relax'),
    ('THANKYOU', 'Sign THANK YOU once in first 3 seconds, then relax'),
    ('PLEASE', 'Sign PLEASE once in first 3 seconds, then relax'),
    ('YES', 'Sign YES once in first 3 seconds, then relax'),
    ('NO', 'Sign NO once in first 3 seconds, then relax'),
    ('HOLD', 'Form HELLO end pose during GET READY; hold it'),
    ('REST', 'Keep hands relaxed and visible; no sign'),
]

ACTIVITY_PROMPTS = [
    ('REST', 'Hands relaxed and visible: do not sign'),
    ('MOVE', 'Move hands naturally without signing'),
    ('HELLO', 'Sign HELLO once in first 3 seconds, then relax'),
    ('YES', 'Sign YES once in first 3 seconds, then relax'),
    ('UNKNOWN', 'Sign an ASL sign outside our 15, then relax'),
    ('REST', 'Hold a comfortable resting position; do not sign'),
    ('THANKYOU', 'Sign THANK YOU once in first 3 seconds, then relax'),
    ('MOVE', 'Adjust hands or clothing naturally; do not sign'),
    ('HELLO', 'Sign HELLO once in first 3 seconds, then relax'),
    ('UNKNOWN', 'Sign any ASL sign outside our 15, then relax'),
    ('YES', 'Sign YES once in first 3 seconds, then relax'),
    ('THANKYOU', 'Sign THANK YOU once in first 3 seconds, then relax'),
    ('HOLD_MY', 'Form the MY hand pose during GET READY; hold it'),
    ('REST', 'Hands relaxed and visible: do not sign'),
]

HEALTH_PROMPTS = [
    ('REST_VISIBLE', 'Hands relaxed but visible: do not sign'),
    ('REST_BELOW', 'Lower hands below frame; do not sign'),
    ('HELLO', 'Sign HELLO once in first 3 seconds, then relax'),
    ('MOVE', 'Move hands naturally without signing'),
    ('HELLO', 'Sign HELLO once in first 3 seconds, then relax'),
    ('REST_NATURAL', 'Rest hands naturally as you did before'),
]


def run(recognizer, report_path, camera=0, trial_seconds=7.0, activity=False, health=False):
    if activity and health:
        raise ValueError('Choose one diagnostic mode')
    prompts = HEALTH_PROMPTS if health else ACTIVITY_PROMPTS if activity else PROMPTS
    if activity:
        from sign_language.activity_metrics import motion_summary
    if health:
        from sign_language.window_health import window_health, failure_category
    cap = cv2.VideoCapture(camera)
    title = 'ASL FILTER DIAGNOSTIC - guided'
    report = {'status': 'starting', 'checkpoint_sha256': recognizer.checkpoint_sha256,
              'epoch': recognizer.epoch, 'threshold_experimental': .60,
              'stable_consecutive_windows': 2, 'window_seconds': 2,
              'interval_seconds': .5, 'trial_seconds': trial_seconds,
              'confirmation': 'pending_user_confirmation', 'video_saved': False,
              'landmarks_saved': False, 'activity_measurements': activity,
              'health_measurements': health, 'trials': []}
    state = 'ready'
    index = 0
    started = time.monotonic()
    last_ms = -1
    trial = None
    status = 'Press SPACE to begin this trial'
    last_result = None

    def save(): write_json(report_path, report)

    def finish_trial(completed):
        nonlocal trial
        if trial is None: return
        trial['completed'] = completed
        trial['duration_seconds'] = time.monotonic() - trial_start
        trial['processed_frames'] = len(buffer.items) if trial['frames_seen'] == 0 else trial['frames_seen']
        report['trials'].append(trial)
        trial = None
        save()

    try:
        if not cap.isOpened(): raise RuntimeError(f'Cannot open camera {camera}')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        report['status'] = 'running'
        save()
        with StreamingFaceTracker(recognizer) as tracker:
            print(f'WINDOW_READY | {len(prompts)} guided trials | SPACE starts | Q aborts', flush=True)
            while index < len(prompts):
                ok, frame = cap.read()
                if not ok or frame is None: raise RuntimeError('Camera stopped returning frames')
                now = time.monotonic()
                stamp = int((now - started) * 1000)
                if stamp <= last_ms: continue
                last_ms = stamp
                body, face, present, blend, hands = tracker.observe(frame, stamp)
                now = time.monotonic()
                if state == 'countdown' and now >= deadline:
                    state, trial_start, next_score = 'recording', now, now + 2
                    buffer = RollingObservations(2000, max_items=300)
                    filters = CandidateFilters()
                    trial = {'number': index + 1, 'prompt': prompts[index][0],
                             'completed': False, 'frames_seen': 0, 'windows': []}
                    last_result = None
                if state == 'recording':
                    if now - trial_start < trial_seconds:
                        buffer.append(stamp, body, face, present, blend, frame.shape)
                        trial['frames_seen'] += 1
                        if now >= next_score:
                            next_score = now + .5
                            snapshot = buffer.snapshot()
                            if len(snapshot[0]) >= 8 and snapshot[0][-1] - snapshot[0][0] >= 1600:
                                coverage = window_health(snapshot) if health else None
                                try:
                                    result = score_window(recognizer, snapshot)
                                    motion = motion_summary(snapshot) if activity else None
                                    result_time = time.monotonic() - trial_start
                                    decisions = filters.update(result['label'], result['score'], result_time)
                                    last_result = decisions
                                    window = {'end_seconds': float(result['window_end_ms']/1000 - (trial_start-started)),
                                        'result_seconds': result_time, 'score': result['score'],
                                        'decisions': decisions, 'inference_seconds': result['inference_seconds'],
                                        'frames': result['observed_frames']}
                                    if activity:
                                        window['motion'] = motion
                                    if health:
                                        window['health'] = coverage
                                    trial['windows'].append(window)
                                except ValueError as exc:
                                    filters.reset()
                                    last_result = None
                                    trial.setdefault('unscored_windows', 0)
                                    trial['unscored_windows'] += 1
                                    reasons = trial.setdefault('unscored_reasons', {})
                                    text = str(exc)
                                    reason = ('hands_or_shoulders' if text == 'Keep hands and shoulders visible'
                                              else 'face' if text == 'Keep face visible'
                                              else 'other_value_error')
                                    reasons[reason] = reasons.get(reason, 0) + 1
                                    if health:
                                        trial.setdefault('unscored_details', []).append({
                                            'end_seconds': float(snapshot[0][-1]/1000 - (trial_start-started)),
                                            'reason': failure_category(exc, coverage),
                                            'health': coverage,
                                        })
                    else:
                        finish_trial(True)
                        index += 1
                        state, last_result = 'ready', None
                        status = 'Press SPACE for the next trial'
                        if index >= len(prompts): break
                view = cv2.flip(draw_hands(frame, hands), 1)
                cv2.rectangle(view, (0, 0), (view.shape[1], 145), (16, 16, 16), -1)
                lines = [f'TRIAL {index+1}/{len(prompts)}: {prompts[index][0]}', prompts[index][1],
                         'GET READY' if state == 'countdown' else
                         f'SIGN NOW: {max(0, trial_seconds-(now-trial_start)):.1f}s' if state == 'recording' else status,
                         'Raw / threshold / stable: ' + (' / '.join(last_result[k] or '-' for k in
                           ('raw', 'threshold', 'threshold_and_stability')) if last_result else '- / - / -'),
                         'SPACE starts | Q stops | provisional filters']
                for row, line in enumerate(lines):
                    cv2.putText(view, line, (8, 23 + row*26), cv2.FONT_HERSHEY_SIMPLEX,
                                .46, (255, 255, 255) if row != 2 else (40, 230, 255), 1, cv2.LINE_AA)
                cv2.imshow(title, view)
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord('q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    if state == 'recording': finish_trial(False)
                    report.update(status='interrupted', reason='user_closed')
                    break
                if key == ord(' ') and state == 'ready':
                    state, deadline = 'countdown', now + 2.0
            else:
                report.update(status='complete', reason='all_trials_completed')
            if index >= len(prompts): report.update(status='complete', reason='all_trials_completed')
    except BaseException as exc:
        if trial is not None: finish_trial(False)
        report.update(status='failed', reason=str(exc))
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        report['elapsed_seconds'] = time.monotonic() - started
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--activity', action='store_true', help='Compare aggregate hand activity across guided prompts')
    parser.add_argument('--health', action='store_true', help='Explain unscored windows with aggregate tracking counts')
    args = parser.parse_args()
    if args.report.exists(): parser.error('Report already exists')
    torch.set_num_threads(2)
    recognizer = load_recognizer('ASL', args.device)
    print('MODEL_READY | ASL | epoch', recognizer.epoch, '| labels', len(recognizer.names), flush=True)
    run(recognizer, args.report, args.camera, activity=args.activity, health=args.health)
    print('FINISHED |', args.report, flush=True)


if __name__ == '__main__': main()
