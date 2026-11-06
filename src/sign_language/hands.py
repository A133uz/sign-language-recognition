"""MediaPipe Tasks hand detector shared by previews and future extraction."""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from sign_language.paths import MODELS

MODEL = MODELS / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


class HandTracker:
    def __init__(self, model: Path = MODEL):
        if not model.is_file():
            raise FileNotFoundError("Hand model missing. Run: python scripts/project.py setup-hands")
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.last_timestamp = -1

    def detect(self, frame: np.ndarray, timestamp_ms: int):
        if timestamp_ms <= self.last_timestamp:
            raise ValueError("Video timestamps must increase strictly")
        self.last_timestamp = timestamp_ms
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.detector.detect_for_video(image, timestamp_ms)

    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def draw_hands(frame: np.ndarray, result) -> np.ndarray:
    """Draw on unmirrored pixels; the caller mirrors the finished preview only."""
    view = frame.copy()
    height, width = frame.shape[:2]
    for index, landmarks in enumerate(result.hand_landmarks):
        color = ((0, 230, 255), (255, 180, 0))[index % 2]
        points = [(round(p.x * (width - 1)), round(p.y * (height - 1))) for p in landmarks]
        for first, second in CONNECTIONS:
            cv2.line(view, points[first], points[second], color, 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(view, point, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(view, point, 2, color, -1, cv2.LINE_AA)
    return view
