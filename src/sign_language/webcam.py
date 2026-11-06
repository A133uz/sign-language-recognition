"""Hand skeleton preview; no sign classification or training."""

import time
from collections import Counter

import cv2
import numpy as np

from sign_language.hands import HandTracker, draw_hands


def run_webcam(camera: int = 0, duration: float = 0) -> dict:
    if duration < 0:
        raise ValueError("Duration must be nonnegative; zero means until Q")
    capture = cv2.VideoCapture(camera)
    started = time.monotonic()
    counts = Counter()
    frames, last_timestamp, detection_seconds = 0, -1, 0.0
    error, reason = None, "user_closed"
    title = "MediaPipe hand skeleton - Q to close"
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open camera {camera}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        with HandTracker() as tracker:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera stopped returning frames")
                now = time.monotonic()
                timestamp = max(last_timestamp + 1, int((now - started) * 1000))
                last_timestamp = timestamp
                before = time.perf_counter()
                result = tracker.detect(frame, timestamp)
                detection_seconds += time.perf_counter() - before
                hands = len(result.hand_landmarks)
                counts[hands] += 1
                frames += 1
                annotated = cv2.flip(draw_hands(frame, result), 1)
                view = np.full((annotated.shape[0] + 112, annotated.shape[1], 3), 20, np.uint8)
                view[112:] = annotated
                elapsed = time.monotonic() - started
                lines = [f"Hands detected: {hands}/2 | Points: {21 * hands} | FPS: {frames/max(elapsed,.001):.1f}",
                         "Move both hands, spread fingers, turn palms and make a fist",
                         "Hand tracking only | No sign classifier | No video saved | Q: close"]
                for row, text in enumerate(lines):
                    cv2.putText(view, text, (12, 29 + row * 32), cv2.FONT_HERSHEY_SIMPLEX,
                                .55, (80, 255, 120), 1, cv2.LINE_AA)
                cv2.imshow(title, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if duration and elapsed >= duration:
                    reason = "duration_reached"
                    break
    except (RuntimeError, ValueError, OSError, cv2.error) as exc:
        error, reason = str(exc), "failed"
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return {
        "status": "failed" if error else "complete", "reason": reason, "error": error,
        "frames": frames, "elapsed_seconds": time.monotonic() - started,
        "frames_by_hand_count": {str(k): counts[k] for k in range(3)},
        "mean_detection_ms": detection_seconds / frames * 1000 if frames else None,
        "media_saved": False, "coordinates_saved": False, "training_performed": False,
    }
