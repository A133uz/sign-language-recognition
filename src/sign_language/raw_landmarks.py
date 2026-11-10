"""Raw hand/body observations; no resampling, augmentation or model training."""

import io
import math
from contextlib import ExitStack
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from sign_language.hands import CONNECTIONS, HandTracker

SCHEMA = {
    "version": "asl-citizen-raw-hands-pose-v1",
    "coordinates": "MediaPipe normalized image xyz; no input mirroring/resizing",
    "hand_order": "detection order, NOT stable left/right slots; use handedness and pose wrists later",
    "handedness": {"missing": -1, "Left": 0, "Right": 1},
    "depth": "hand z is wrist-relative; pose z has a different origin; NOT shared metric/world depth",
    "missing": "absent groups are zero-filled with explicit presence masks",
    "pose_mask": "finite coordinates and visibility/presence >= 0.5; low-confidence raw points retained",
    "temporal": "all decoded frames; timestamp_ms from source fps, strictly increasing",
    "preprocessing": "none beyond BGR->RGB for MediaPipe; normalize/resample in future trainer",
    "hand_connections": [list(edge) for edge in CONNECTIONS],
    "upper_pose_connections": [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24]],
    "arrays": {
        "hands_xyz": "float32 [T,2,21,3]",
        "hand_present": "bool [T,2]",
        "handedness": "int8 [T,2]",
        "handedness_score": "float32 [T,2]; category score, not joint tracking confidence",
        "pose_xyz": "float32 [T,33,3]",
        "pose_visibility": "float32 [T,33]",
        "pose_presence": "float32 [T,33]",
        "pose_detected": "bool [T]",
        "pose_valid_mask": "bool [T,33]",
        "timestamp_ms": "int64 [T]",
        "frame_index": "int32 [T]",
    },
}


def frame_observation(hands, pose=None) -> dict:
    out = {
        "hands_xyz": np.zeros((2, 21, 3), np.float32),
        "hand_present": np.zeros(2, bool),
        "handedness": np.full(2, -1, np.int8),
        "handedness_score": np.zeros(2, np.float32),
        "pose_xyz": np.zeros((33, 3), np.float32),
        "pose_visibility": np.zeros(33, np.float32),
        "pose_presence": np.zeros(33, np.float32),
        "pose_detected": np.asarray(False),
        "pose_valid_mask": np.zeros(33, bool),
    }
    if len(hands.hand_landmarks) > 2:
        raise ValueError("More than two hands returned")
    for slot, points in enumerate(hands.hand_landmarks):
        if len(points) != 21:
            raise ValueError("Expected 21 hand points")
        out["hands_xyz"][slot] = [(p.x, p.y, p.z) for p in points]
        out["hand_present"][slot] = True
        category = hands.handedness[slot][0]
        out["handedness"][slot] = {"Left": 0, "Right": 1}.get(category.category_name, -1)
        out["handedness_score"][slot] = category.score
    if pose is not None and pose.pose_landmarks:
        points = pose.pose_landmarks[0]
        if len(points) != 33:
            raise ValueError("Expected 33 pose points")
        out["pose_detected"] = np.asarray(True)
        out["pose_xyz"][:] = [(p.x, p.y, p.z) for p in points]
        out["pose_visibility"][:] = [p.visibility or 0.0 for p in points]
        out["pose_presence"][:] = [p.presence or 0.0 for p in points]
        out["pose_valid_mask"][:] = (
            (out["pose_visibility"] >= .5) & (out["pose_presence"] >= .5)
        )
    if any(not np.isfinite(value).all() for value in out.values()):
        raise ValueError("Nonfinite landmark output")
    return out


def extract_clip(video: Path, hand_model: Path, pose_model: Path | None, max_frames=3600):
    capture = cv2.VideoCapture(str(video))
    frames, timestamps = [], []
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open video")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid source FPS; no silent FPS substitution")
        with ExitStack() as resources:
            # New tracker state for every clip prevents cross-signer tracking carryover.
            hands = resources.enter_context(HandTracker(hand_model))
            pose = None
            if pose_model is not None:
                options = mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(
                        model_asset_path=str(pose_model), delegate=mp.tasks.BaseOptions.Delegate.CPU
                    ),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_poses=1, output_segmentation_masks=False,
                    min_pose_detection_confidence=.5, min_pose_presence_confidence=.5,
                    min_tracking_confidence=.5,
                )
                pose = mp.tasks.vision.PoseLandmarker.create_from_options(options)
                resources.callback(pose.close)
            shape = None
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if len(frames) >= max_frames:
                    raise ValueError(f"Exceeded {max_frames} frames; video was not truncated")
                if shape is not None and frame.shape != shape:
                    raise ValueError("Video dimensions changed during decoding")
                shape = frame.shape
                timestamp = max(timestamps[-1] + 1 if timestamps else 0,
                                round(len(frames) * 1000 / fps))
                hand_result = hands.detect(frame, timestamp)
                pose_result = None
                if pose is not None:
                    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    pose_result = pose.detect_for_video(image, timestamp)
                frames.append(frame_observation(hand_result, pose_result))
                timestamps.append(timestamp)
        if not frames or (declared > 0 and len(frames) != declared):
            raise ValueError(f"Incomplete decode: {len(frames)} frames; declared {declared}")
    finally:
        capture.release()
    arrays = {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}
    arrays["timestamp_ms"] = np.asarray(timestamps, np.int64)
    arrays["frame_index"] = np.arange(len(frames), dtype=np.int32)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    metadata = {
        "frames": len(frames), "declared_frames": declared, "fps": fps,
        "frame_shape": list(shape), "source_bytes": video.stat().st_size,
        "frames_with_hands": int(arrays["hand_present"].any(axis=1).sum()),
        "frames_with_pose": int(arrays["pose_detected"].sum()),
        "hand_detections": int(arrays["hand_present"].sum()),
        "pose_valid_points": int(arrays["pose_valid_mask"].sum()),
    }
    return buffer.getvalue(), metadata


def extract_job(job):
    """Spawn-safe worker; failures remain explicit records, never fake feature rows."""
    row, video_root, models, max_frames = job
    try:
        data, quality = extract_clip(
            Path(video_root) / row["video_file"], Path(models["hands"]),
            Path(models["pose"]) if "pose" in models else None, max_frames,
        )
        return {**row, **quality, "status": "ok"}, data
    except Exception as exc:
        return {**row, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}, None
