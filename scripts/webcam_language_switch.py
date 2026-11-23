"""Recognize one isolated sign with an explicitly selected ASL or RSL model."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from sign_language.dataset import write_json
from sign_language.frame_rate import FrameRateLimiter
from sign_language.recognition_webcam import SkeletonPreview, _font

ASL_CHECKPOINT = ROOT / 'models/asl-gru-fifteen-signs-face-50epochs-v1/model.pt'
RSL_CHECKPOINT = ROOT / 'models/slovo-gru-v1/best.pt'
MODELS = ROOT / 'models'


def load_recognizer(language, device, asl_checkpoint=ASL_CHECKPOINT,
                    rsl_checkpoint=RSL_CHECKPOINT, models=MODELS):
    """Load each model with the preprocessing and label map it was trained for."""
    if language == 'ASL':
        from sign_language.face_gru_recognition import FaceAwareRecognizer
        recognizer = FaceAwareRecognizer(asl_checkpoint, models, device=device)
        if len(recognizer.names) != 15 or recognizer.epoch != 36:
            raise ValueError('Unexpected ASL face-aware checkpoint')
        return recognizer
    if language == 'RSL':
        if not Path(rsl_checkpoint).is_file():
            raise FileNotFoundError('No trained SLOVO checkpoint installed')
        from sign_language.slovo_recognition import SlovoRecognizer
        return SlovoRecognizer(rsl_checkpoint, models, device=device)
    raise ValueError(f'Unsupported sign language: {language}')


class LanguageSelector:
    """Never change model while a sign is being captured or scored."""

    def __init__(self, loader):
        self.loader = loader
        self.language = 'ASL'
        self.models = {'ASL': loader('ASL')}

    @property
    def recognizer(self):
        return self.models[self.language]

    def select(self, language, state):
        if state != 'ready':
            return False, 'Finish the current capture before switching language'
        if language == self.language:
            return False, f'{language} already selected'
        try:
            if language not in self.models:
                self.models[language] = self.loader(language)
        except Exception as exc:
            return False, f'{language} unavailable: {exc}'
        self.language = language
        return True, f'{language} selected'


def top_prediction(result):
    """Both adapters report their own model's best gloss."""
    if 'top1' in result:
        return result['top1']
    return result['top5'][0]


def overlay(frame, language, status, subtitle):
    """Show the active language and one Unicode-capable subtitle."""
    canvas = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    small, large = _font(17), _font(28)
    draw.rectangle((0, 0, canvas.width, 53), fill=(20, 20, 20))
    draw.text((10, 4), f'LANGUAGE: {language}   |   1: ASL   2: RSL   |   SPACE: SIGN   Q: QUIT',
              font=small, fill='white')
    status_size = 17
    while status_size > 10 and draw.textbbox((0, 0), status, font=_font(status_size))[2] > canvas.width - 20:
        status_size -= 1
    draw.text((10, 27), status, font=_font(status_size), fill=(245, 220, 110))
    if subtitle:
        # Fit even long Russian or English glosses without changing their characters.
        size = 28
        while size > 10 and draw.textbbox((0, 0), subtitle, font=_font(size))[2] > canvas.width - 30:
            size -= 1
        large = _font(size)
        box = draw.textbbox((0, 0), subtitle, font=large)
        width, height = box[2] - box[0], box[3] - box[1]
        x, y = max(10, (canvas.width - width) // 2), canvas.height - height - 30
        draw.rectangle((x-8, y-4, min(canvas.width-1, x+width+8), canvas.height-12), fill=(20, 20, 20))
        draw.text((x, y-box[1]), subtitle, font=large, fill='white')
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def run(selector, report_path, camera=0, capture_seconds=3.0, capture_fps=30.0):
    capture = cv2.VideoCapture(camera)
    pool = ThreadPoolExecutor(max_workers=2)
    preview = SkeletonPreview(selector.recognizer.hand_model)
    preview_job = prediction_job = None
    preview_frame = None
    state, status, subtitle = 'ready', 'Ready', ''
    frames, timestamps = [], []
    deadline = started = 0.0
    limiter = FrameRateLimiter(capture_fps)
    last_preview = -1
    title = 'Sign recognition - language selector'
    report = {'status': 'starting', 'video_saved': False, 'landmarks_saved': False, 'trials': []}

    def save():
        write_json(report_path, report)

    def completed(job, language, model):
        result = job.result()
        best = top_prediction(result)
        item = {'capture_id': len(report['trials']) + 1, 'language': language,
                'checkpoint_sha256': model.checkpoint_sha256, 'epoch': model.epoch,
                'prediction': best['label'], 'captured_frames': result['captured_frames'],
                'capture_seconds': result['capture_seconds'],
                'processing_seconds': result['processing_seconds']}
        report['trials'].append(item)
        save()
        return best['label']

    try:
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open camera {camera}')
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_FPS, capture_fps)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        report.update(status='running', camera_open=True)
        save()
        print('WINDOW_READY | 1: ASL | 2: RSL | Space: start | Q: quit', flush=True)
        begin = time.perf_counter()
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError('Camera stopped returning frames')
            now = time.perf_counter()
            timestamp = max(last_preview + 1, round((now - begin) * 1000))
            last_preview = timestamp
            if preview_job is not None and preview_job.done():
                try:
                    preview_frame, _ = preview_job.result()
                except Exception as exc:
                    print('PREVIEW_ERROR', str(exc), flush=True)
                preview_job = None
            if preview_job is None:
                preview_job = pool.submit(preview.draw, frame.copy(), timestamp)
            if prediction_job is not None and prediction_job.done():
                try:
                    subtitle = completed(prediction_job, trial_language, trial_model)
                    status = 'Ready'
                except Exception as exc:
                    subtitle, status = '', f'Prediction failed: {exc}'
                    report['trials'].append({'capture_id': len(report['trials']) + 1,
                                             'language': trial_language, 'error': str(exc)})
                    save()
                prediction_job = None
                state = 'ready'
            if state == 'countdown' and now >= deadline:
                state, frames, timestamps, started = 'recording', [], [], now
                limiter = FrameRateLimiter(capture_fps)
                status = 'SIGN NOW'
            if state == 'recording':
                if now - started <= capture_seconds and limiter.accept(now - started):
                    frames.append(frame.copy())
                    stamp = round((now - started) * 1000)
                    timestamps.append(max(stamp, timestamps[-1] + 1 if timestamps else 0))
                if now - started >= capture_seconds:
                    if len(frames) < 8:
                        state, status = 'ready', 'Too few frames; try again'
                    else:
                        trial_language, trial_model = selector.language, selector.recognizer
                        prediction_job = pool.submit(trial_model.predict_frames, frames, timestamps)
                        state, status = 'processing', 'Recognizing...'
                    frames, timestamps = [], []
            image = preview_frame if preview_frame is not None else frame
            view = overlay(cv2.flip(image, 1), selector.language, status, subtitle)
            cv2.imshow(title, view)
            key = cv2.waitKey(1) & 0xff
            if key in (27, ord('q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord('1'), ord('2')):
                changed, status = selector.select('ASL' if key == ord('1') else 'RSL', state)
                if changed:
                    subtitle = ''
            if key == ord(' ') and state == 'ready':
                state, deadline, status, subtitle = 'countdown', now + 1.5, 'GET READY', ''
        report.update(status='complete', reason='user_closed')
    except BaseException as exc:
        report.update(status='failed', reason=str(exc))
        raise
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if prediction_job is not None:
            try:
                completed(prediction_job, trial_language, trial_model)
            except Exception as exc:
                print('PREDICTION_ERROR', str(exc), flush=True)
        if preview_job is not None:
            try:
                preview_job.result()
            except Exception:
                pass
        preview.close()
        pool.shutdown(wait=True)
        report['camera_open'] = False
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    selector = LanguageSelector(lambda language: load_recognizer(language, args.device))
    report = ROOT / '.cache' / f'webcam-language-{datetime.now():%Y%m%d-%H%M%S-%f}.json'
    run(selector, report, camera=args.camera)


if __name__ == '__main__':
    main()
