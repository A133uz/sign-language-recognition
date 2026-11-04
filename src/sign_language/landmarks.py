"""Shared hand-landmark representation for video and webcam inputs."""

from pathlib import Path

import numpy as np

SEQUENCE_LENGTH = 32
FEATURE_SIZE = 128  # two hands: 21 xyz landmarks plus one presence bit each
FEATURE_SCHEMA = "hands-image-xyz-xsorted-v1"
POSE_FEATURE_SCHEMA = "hands-pose-image-xyz-relative-v1"
POSE_FEATURE_SIZE = 160
WORLD_FEATURE_SCHEMA = "hands-pose-world-wrist-normalized-v1"
WORLD_FEATURE_SIZE = 288
FEATURE_SCHEMAS = {
    FEATURE_SCHEMA: FEATURE_SIZE,
    POSE_FEATURE_SCHEMA: POSE_FEATURE_SIZE,
    WORLD_FEATURE_SCHEMA: WORLD_FEATURE_SIZE,
}
POSE_INDICES = (0, 11, 12, 13, 14, 15, 16)  # nose, shoulders, elbows and wrists


class CaptureRejected(ValueError):
    """A capture lacks sufficient tracked hands and must not receive a forced label."""


def encode_hands(hands: list) -> np.ndarray:
    """Encode up to two hands, ordered by image x; retain position and missing masks."""
    encoded = np.zeros((2, 64), dtype=np.float32)
    for slot, hand in enumerate(sorted(hands, key=lambda points: points[0].x)[:2]):
        if len(hand) != 21:
            raise ValueError("Each detected hand must contain 21 landmarks.")
        encoded[slot, :63] = np.array([(p.x, p.y, p.z) for p in hand], dtype=np.float32).ravel()
        encoded[slot, 63] = 1.0
    return encoded.ravel()


def sample_sequence(features: np.ndarray, length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Sample frames uniformly, including both endpoints; short sequences repeat frames."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] not in FEATURE_SCHEMAS.values() or not len(features):
        raise ValueError("Expected at least one frame with a supported feature width.")
    if length < 1 or not np.isfinite(features).all():
        raise ValueError("Sequence length and values must be valid and finite.")
    indices = np.linspace(0, len(features) - 1, length).round().astype(int)
    return features[indices]


class HandExtractor:
    """Use the same unmirrored RGB image input for files and webcam frames."""

    def __init__(
        self, model_path: Path, pose_model_path: Path | None = None, *, include_world: bool = False
    ):
        if include_world and pose_model_path is None:
            raise ValueError("World-hand representation requires a pose model")
        self.include_world = include_world
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError("MediaPipe is required; install the vision profile.") from exc
        if not model_path.is_file():
            raise FileNotFoundError(f"MediaPipe hand model not found: {model_path}")
        if pose_model_path is not None and not pose_model_path.is_file():
            raise FileNotFoundError(f"MediaPipe pose model not found: {pose_model_path}")
        self.mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.pose_landmarker = None
        if pose_model_path is not None:
            try:
                pose_options = vision.PoseLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(pose_model_path)),
                    running_mode=vision.RunningMode.IMAGE,
                    num_poses=1,
                )
                self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
            except Exception:
                self.landmarker.close()
                raise

    def extract(self, bgr_frame: np.ndarray) -> np.ndarray:
        import cv2

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(image)
        hands = encode_hands(result.hand_landmarks)
        if self.pose_landmarker is None:
            return hands
        pose = self.pose_landmarker.detect(image)
        context = encode_body_context(hands, pose.pose_landmarks)
        if self.include_world:
            return np.concatenate(
                (context, encode_world_hands(result.hand_landmarks, result.hand_world_landmarks))
            )
        return context

    def close(self) -> None:
        try:
            self.landmarker.close()
        finally:
            if self.pose_landmarker is not None:
                self.pose_landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def encode_body_context(hands: np.ndarray, poses: list) -> np.ndarray:
    """Append upper-body landmarks and wrist-to-nose offsets in shoulder widths."""
    body = np.zeros(32, dtype=np.float32)
    if poses:
        pose = poses[0]
        if len(pose) < 17:
            raise ValueError("Pose must contain at least 17 landmarks.")
        for slot, index in enumerate(POSE_INDICES):
            point = pose[index]
            body[slot * 4 : slot * 4 + 4] = [point.x, point.y, point.z, point.visibility or 0.0]
        nose, left, right = pose[0], pose[11], pose[12]
        width = ((left.x - right.x) ** 2 + (left.y - right.y) ** 2) ** 0.5
        if width > 0.03 and all((p.visibility or 0) >= 0.5 for p in (nose, left, right)):
            for slot in range(2):
                if hands[slot * 64 + 63]:
                    body[28 + slot * 2] = (hands[slot * 64] - nose.x) / width
                    body[29 + slot * 2] = (hands[slot * 64 + 1] - nose.y) / width
    return np.concatenate((hands, body))


def encode_world_hands(image_hands: list, world_hands: list) -> np.ndarray:
    """Keep image-x slot pairing; center each estimated 3D hand at its wrist and normalize size."""
    if len(image_hands) != len(world_hands):
        raise ValueError("Image and world hand results do not match")
    encoded = np.zeros((2, 64), dtype=np.float32)
    order = sorted(range(len(image_hands)), key=lambda index: image_hands[index][0].x)[:2]
    for slot, index in enumerate(order):
        hand = world_hands[index]
        if len(hand) != 21:
            raise ValueError("Each world hand must contain 21 landmarks")
        points = np.array([(point.x, point.y, point.z) for point in hand], dtype=np.float32)
        if not np.isfinite(points).all():
            raise ValueError("World hand coordinates must be finite")
        scale = np.linalg.norm(points[9] - points[0])
        if scale > 1e-6:
            encoded[slot, :63] = ((points - points[0]) / scale).ravel()
            encoded[slot, 63] = 1
    return encoded.ravel()
