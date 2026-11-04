"""Experimental fixed body-framing alignment; sign motion is preserved across a capture."""

import hashlib
import json
from pathlib import Path

import numpy as np


def load_reference(path: Path) -> dict:
    """Read an explicit local calibration artifact, not a new model or label mapping."""
    reference = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(reference, dict)
        or type(reference.get("schema_version")) is not int
        or reference.get("schema_version") != 1
        or reference.get("crop_size") != 224
        or not {"nose_xy_px", "shoulder_span_px"} <= reference.keys()
    ):
        raise ValueError("Invalid body-framing reference file")
    try:
        nose = np.asarray(reference["nose_xy_px"], dtype=np.float64)
        span = float(reference["shoulder_span_px"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid numeric framing reference") from exc
    if (
        nose.shape != (2,)
        or not np.isfinite(nose).all()
        or not np.all((nose > 0) & (nose < 224))
        or not np.isfinite(span)
        or not 5 < span < 224
    ):
        raise ValueError("Invalid numeric framing reference")
    return reference


def build_reference(features_path: Path, videos: Path, output: Path) -> dict:
    """Fit framing statistics on training clips only, verifying source pixel hashes."""
    import cv2

    from sign_language.landmarks import POSE_FEATURE_SCHEMA
    from sign_language.view_quality import crop_geometry

    if output.exists():
        raise FileExistsError(f"Reference already exists: {output}")
    with np.load(features_path, allow_pickle=False) as data:
        features, splits = data["features"], data["splits"]
        report = json.loads(str(data["report"]))
    if report.get("feature_schema") != POSE_FEATURE_SCHEMA or features.shape[1:] != (32, 160):
        raise ValueError("Reference requires prepared hand/pose sequences")
    if len(features) != len(splits) or len(features) != len(report["records"]):
        raise ValueError("Feature and split records do not align")
    values, used, skipped = [], [], []
    for record, sequence, split in zip(report["records"], features, splits, strict=True):
        if split != record["assigned_split"]:
            raise ValueError("Feature split disagrees with its manifest")
        if split != "train":
            continue
        video_id = record["video_id"]
        if not isinstance(video_id, str) or not video_id.isascii() or not video_id.isalnum():
            raise ValueError("Invalid training video ID")
        path = videos / f"{video_id}.mp4"
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != record["content_sha256"]:
                raise ValueError(f"Training video changed since feature extraction: {video_id}")
        capture = cv2.VideoCapture(str(path))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise ValueError(f"Cannot decode training video: {video_id}")
        height, width = frame.shape[:2]
        resized_width, resized_height, left, top = crop_geometry(width, height, "wlasl")
        pose = sequence[:, 128:156].reshape(32, 7, 4)
        valid = (pose[:, :3, 3] >= 0.5).all(axis=1)
        if valid.sum() < 8:
            skipped.append(video_id)
            continue
        xy = pose[valid, :, :2] * [resized_width, resized_height] - [left, top]
        values.append(
            [
                *np.median(xy[:, 0], axis=0),
                np.median(np.linalg.norm(xy[:, 1] - xy[:, 2], axis=1)),
            ]
        )
        used.append(video_id)
    if not values or not np.isfinite(values).all():
        raise ValueError("No valid training body-framing measurements")
    target = np.median(values, axis=0)
    with features_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    reference = {
        "schema_version": 1,
        "status": "experimental_train_only",
        "crop_size": 224,
        "feature_sha256": digest,
        "training_clips": len(used),
        "training_video_ids": used,
        "skipped_training_video_ids": skipped,
        "nose_xy_px": target[:2].tolist(),
        "shoulder_span_px": float(target[2]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(reference, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return reference


def estimate_alignment(
    features: np.ndarray, width: int, height: int, reference: dict
) -> tuple[np.ndarray, dict]:
    """Map median nose and shoulder span to a reference model view, without rotating hands."""
    if width < 1 or height < 1:
        raise ValueError("Frame dimensions must be positive")
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 160 or not np.isfinite(features).all():
        raise ValueError("Alignment requires finite hand/pose features")
    try:
        target_nose = np.asarray(reference["nose_xy_px"], dtype=np.float64)
        target_span = float(reference["shoulder_span_px"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid body-framing reference") from exc
    if (
        target_nose.shape != (2,)
        or not np.isfinite(target_nose).all()
        or not np.all((target_nose > 0) & (target_nose < 224))
        or not np.isfinite(target_span)
        or not 5 < target_span < 224
    ):
        raise ValueError("Invalid body-framing reference")
    pose = features[:, 128:156].reshape(-1, 7, 4)[:, :3]
    valid = (pose[:, :, 3] >= 0.5).all(axis=1)
    valid &= ((pose[:, :, :2] >= 0) & (pose[:, :, :2] <= 1)).all(axis=(1, 2))
    if valid.sum() < 8:
        raise ValueError("Alignment needs visible nose and shoulders in at least eight samples")
    xy = pose[valid, :, :2] * [width, height]
    nose = np.median(xy[:, 0], axis=0)
    span = float(np.median(np.linalg.norm(xy[:, 1] - xy[:, 2], axis=1)))
    if span < 5:
        raise ValueError("Shoulder span is too small for alignment")
    scale = target_span / span
    effective_zoom = scale / (256 / min(width, height))
    if not 0.5 <= effective_zoom <= 2:
        raise ValueError("Alignment zoom exceeds the supported 0.5 to 2 range")
    translation = target_nose + 16 - scale * nose
    matrix = np.array([[scale, 0, translation[0]], [0, scale, translation[1]]], dtype=np.float32)
    return matrix, {"valid_pose_samples": int(valid.sum()), "effective_zoom": effective_zoom}


def align_frames(frames: list[np.ndarray], matrix: np.ndarray) -> tuple[list[np.ndarray], float]:
    """Warp every frame with the same transform; report padding in the final model crop."""
    import cv2

    if not frames or any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("Alignment needs consistently sized frames")
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("Invalid alignment matrix")
    mask = cv2.warpAffine(
        np.ones(frames[0].shape[:2], dtype=np.uint8), matrix, (256, 256), flags=cv2.INTER_NEAREST
    )
    padding = float(1 - mask[16:240, 16:240].mean())
    if padding > 0.35:
        raise ValueError("Alignment would pad more than 35% of the model view")
    return [
        cv2.warpAffine(frame, matrix, (256, 256), borderValue=(127, 127, 127)) for frame in frames
    ], padding
