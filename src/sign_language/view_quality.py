"""Aggregate hand/pose geometry in the video model's crop, without storing landmarks."""

import numpy as np


def crop_geometry(width: int, height: int, backend: str) -> tuple[int, int, int, int]:
    """Return resized width/height and crop origin using each backend's rounding rules."""
    if width < 1 or height < 1:
        raise ValueError("Frame dimensions must be positive")
    if backend == "wlasl":
        scale = 256 / min(width, height)
        resized_width, resized_height = (
            max(256, round(width * scale)),
            max(256, round(height * scale)),
        )
        left, top = (resized_width - 224) // 2, (resized_height - 224) // 2
    elif backend == "citizen":
        scale = max(1.0, 226 / min(width, height))
        resized_width, resized_height = round(width * scale), round(height * scale)
        if max(width, height) > 256:
            resized_width = resized_height = 256
        left, top = (
            int(np.round((resized_width - 224) / 2)),
            int(np.round((resized_height - 224) / 2)),
        )
    else:
        raise ValueError("Backend must be wlasl or citizen")
    return resized_width, resized_height, left, top


def _quantiles(values: np.ndarray) -> dict | None:
    if not len(values):
        return None
    quantiles = np.quantile(values, [0.1, 0.5, 0.9])
    return {key: float(value) for key, value in zip(("p10", "p50", "p90"), quantiles, strict=True)}


def summarize_model_view(
    features: np.ndarray, width: int, height: int, *, backend: str = "wlasl"
) -> dict:
    """Describe all detected hands, not an inferred active hand or recognition quality."""
    features = np.asarray(features, dtype=np.float64)
    if (
        features.ndim != 2
        or features.shape[1] not in (128, 160)
        or not len(features)
        or not np.isfinite(features).all()
    ):
        raise ValueError("Expected finite per-frame hand or hand/pose features")
    resized_width, resized_height, left, top = crop_geometry(width, height, backend)
    hands = features[:, :128].reshape(-1, 2, 64)
    present = hands[:, :, 63] > 0.5
    points = hands[:, :, :63].reshape(-1, 2, 21, 3)[..., :2]
    points = points * np.array([resized_width, resized_height]) - np.array([left, top])
    detected = points[present]
    spans = np.ptp(detected, axis=1) if len(detected) else np.empty((0, 2))
    inside = np.all((detected >= 0) & (detected < 224), axis=-1)
    result = {
        "backend": backend,
        "sampled_frames": len(features),
        "frames_with_hands": int(present.any(axis=1).sum()),
        "detected_hand_instances": int(present.sum()),
        "all_hand_points_inside_fraction": float(inside.all(axis=1).mean())
        if len(detected)
        else None,
        "hand_points_inside_fraction": float(inside.mean()) if len(detected) else None,
        "hand_bbox_short_side_px": _quantiles(spans.min(axis=1)),
        "hand_bbox_long_side_px": _quantiles(spans.max(axis=1)),
        "resized_frame": [resized_width, resized_height],
        "crop_origin": [left, top],
        "model_crop_size": [224, 224],
        "pose_context_frames": 0,
        "shoulder_span_px": None,
        "nose_y_px": None,
    }
    if features.shape[1] == 160:
        pose = features[:, 128:156].reshape(-1, 7, 4)
        valid = (pose[:, :3, 3] >= 0.5).all(axis=1)
        pose_xy = pose[valid, :, :2] * np.array([resized_width, resized_height]) - np.array(
            [left, top]
        )
        result["pose_context_frames"] = int(valid.sum())
        result["shoulder_span_px"] = _quantiles(
            np.linalg.norm(pose_xy[:, 1] - pose_xy[:, 2], axis=1)
        )
        result["nose_y_px"] = _quantiles(pose_xy[:, 0, 1])
    return result


def analyze_capture_view(frames: list[np.ndarray], extractor, backend: str) -> dict:
    """Extract 32 in-memory samples and return aggregate geometry only."""
    features = extract_capture_features(frames, extractor)
    height, width = frames[0].shape[:2]
    return summarize_model_view(features, width, height, backend=backend)


def extract_capture_features(
    frames: list[np.ndarray], extractor, *, unique: bool = False
) -> np.ndarray:
    """Share ephemeral samples between geometry analysis and optional body alignment."""
    import cv2

    if (
        not frames
        or frames[0].ndim != 3
        or frames[0].shape[2] != 3
        or any(frame.shape != frames[0].shape for frame in frames)
    ):
        raise ValueError("View analysis needs nonempty, consistently sized color frames")
    height, width = frames[0].shape[:2]
    indexes = (
        np.linspace(0, len(frames) - 1, min(32, len(frames)) if unique else 32).round().astype(int)
    )
    features = []
    for index in indexes:
        frame = frames[index]
        scale = min(1.0, 480 / max(height, width))
        if scale < 1:
            frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        features.append(extractor.extract(frame))
    return np.stack(features)
