"""Experimental streaming inference using the existing face-aware checkpoint."""

from collections import deque
from contextlib import ExitStack
import time

import cv2
import mediapipe as mp
import numpy as np
import torch

from sign_language.face_gru_recognition import normalize_joined
from sign_language.hands import HandTracker
from sign_language.raw_landmarks import frame_observation


class RollingObservations:
    """Bounded, timestamp-ordered raw observations; contains no images."""

    def __init__(self, duration_ms=2000, max_items=180):
        if duration_ms <= 0 or max_items < 8:
            raise ValueError('Invalid window limits')
        self.duration_ms = duration_ms
        self.items = deque(maxlen=max_items)
        self.shape = None

    def append(self, timestamp_ms, observation, face, face_present, blend, frame_shape):
        if self.items and timestamp_ms <= self.items[-1][0]:
            raise ValueError('Timestamps must strictly increase')
        if self.shape is not None and tuple(frame_shape) != self.shape:
            raise ValueError('Camera dimensions changed')
        self.shape = tuple(frame_shape)
        # Own the arrays because MediaPipe and callers may reuse buffers.
        body = {key: np.array(value, copy=True) for key, value in observation.items()}
        self.items.append((timestamp_ms, body, np.array(face, copy=True),
                           bool(face_present), np.array(blend, copy=True)))
        cutoff = timestamp_ms - self.duration_ms
        while self.items and self.items[0][0] < cutoff:
            self.items.popleft()

    def snapshot(self):
        if not self.items:
            raise ValueError('Window is empty')
        items = list(self.items)
        body = {key: np.stack([item[1][key] for item in items]) for key in items[0][1]}
        return (np.asarray([item[0] for item in items]), body,
                np.stack([item[2] for item in items]),
                np.asarray([item[3] for item in items]),
                np.stack([item[4] for item in items]), self.shape)


class StreamingFaceTracker:
    """Persistent MediaPipe trackers configured like the face-aware adapter."""

    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.resources = ExitStack()
        self.last_timestamp = -1
        try:
            self.hands = self.resources.enter_context(HandTracker(recognizer.hand_model))
            pose_options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(recognizer.pose_model),
                    delegate=mp.tasks.BaseOptions.Delegate.CPU),
                running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1,
                output_segmentation_masks=False, min_pose_detection_confidence=.5,
                min_pose_presence_confidence=.5, min_tracking_confidence=.5)
            self.pose = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)
            self.resources.callback(self.pose.close)
            face_options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(recognizer.face_model),
                    delegate=mp.tasks.BaseOptions.Delegate.CPU),
                running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1,
                output_face_blendshapes=True, min_face_detection_confidence=.5,
                min_face_presence_confidence=.5, min_tracking_confidence=.5)
            self.face = mp.tasks.vision.FaceLandmarker.create_from_options(face_options)
            self.resources.callback(self.face.close)
        except BaseException:
            self.resources.close()
            raise

    def observe(self, frame, timestamp_ms):
        if timestamp_ms <= self.last_timestamp:
            raise ValueError('Timestamps must strictly increase')
        self.last_timestamp = timestamp_ms
        hand_result = self.hands.detect(frame, timestamp_ms)
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        body = frame_observation(hand_result, self.pose.detect_for_video(image, timestamp_ms))
        result = self.face.detect_for_video(image, timestamp_ms)
        points, scores = np.zeros((478, 3), np.float32), np.zeros(52, np.float32)
        present = bool(result.face_landmarks)
        if present:
            if (len(result.face_landmarks) != 1 or len(result.face_landmarks[0]) != 478 or
                    len(result.face_blendshapes) != 1 or len(result.face_blendshapes[0]) != 52):
                raise ValueError('Unexpected face output')
            categories = result.face_blendshapes[0]
            if [item.category_name for item in categories] != self.recognizer.config['blendshape_names']:
                raise ValueError('Blendshape order differs')
            points[:] = [(point.x, point.y, point.z) for point in result.face_landmarks[0]]
            scores[:] = [item.score for item in categories]
        return body, points, present, scores, hand_result

    def close(self):
        self.resources.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def score_window(recognizer, snapshot):
    """Score one immutable window with the unchanged per-window preprocessing."""
    started = time.perf_counter()
    timestamps, body, face, face_present, blend, shape = snapshot
    if len(timestamps) < 8 or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Window needs eight frames and increasing timestamps')
    joined, quality = normalize_joined(body, face, face_present, blend, shape)
    if not quality['frames_with_hands'] or not quality['frames_with_body_anchor']:
        raise ValueError('Keep hands and shoulders visible')
    if not quality['frames_with_face']:
        raise ValueError('Keep face visible')
    x = torch.from_numpy(joined).unsqueeze(0).to(recognizer.device)
    lengths = torch.tensor([len(joined)], dtype=torch.long)
    with torch.inference_mode():
        logits = recognizer.model.score(x, lengths)
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite prediction')
    probabilities = logits.softmax(-1)
    index = int(probabilities.argmax(-1).item())
    return {'label': recognizer.names[index], 'score': float(probabilities[0, index]),
            'window_end_ms': int(timestamps[-1]), 'window_start_ms': int(timestamps[0]),
            'input_frames': len(joined), 'observed_frames': len(timestamps),
            'quality': quality, 'inference_seconds': time.perf_counter() - started}
