"""Inventory locally obtained WLASL videos without downloading them."""

import hashlib
import math
from collections import Counter
from pathlib import Path

from sign_language.wlasl import find_sign


def inspect_local_video(path: Path) -> dict:
    """Decode sequentially and fingerprint a local clip; do not assess its label."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required; install the vision profile first.") from exc

    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {"status": "decode_failed", "sha256": digest}
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        declared_frames = (
            int(frame_count) if math.isfinite(frame_count) and frame_count > 0 else None
        )
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = 0
        dimensions = None
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames += 1
            if dimensions is None:
                height, width = frame.shape[:2]
                dimensions = {"width": width, "height": height}
    finally:
        capture.release()
    return {
        "status": "decoded" if frames else "decode_failed",
        "sha256": digest,
        "decoded_frames": frames,
        "reported_frames": declared_frames,
        "reported_fps": fps if math.isfinite(fps) and fps > 0 else None,
        **(dimensions or {}),
    }


def inventory_videos(signs: list[dict], labels: tuple[str, ...], directory: Path) -> dict:
    """Look for <video_id>.mp4 files, retaining a result for every selected record."""
    if not directory.is_dir():
        raise ValueError(f"Video directory does not exist: {directory}")
    records = []
    seen = set()
    for label in labels:
        sign = find_sign(signs, label)
        if sign is None:
            raise ValueError(f"Vocabulary gloss not found in metadata: {label}")
        for instance in sign["instances"]:
            video_id = instance.get("video_id")
            if not isinstance(video_id, str) or not video_id.isascii() or not video_id.isalnum():
                raise ValueError(f"Invalid video ID for gloss: {label}")
            if video_id in seen:
                raise ValueError(f"Duplicate video ID in selected metadata: {video_id}")
            seen.add(video_id)
            filename = f"{video_id}.mp4"
            path = directory / filename
            result = inspect_local_video(path) if path.is_file() else {"status": "missing"}
            records.append(
                {
                    "gloss": sign["gloss"],
                    "video_id": video_id,
                    "signer_id": instance.get("signer_id"),
                    "file": filename,
                    "frame_start": instance.get("frame_start"),
                    "frame_end": instance.get("frame_end"),
                    **result,
                }
            )
    return {
        "status": "local_decode_inventory_not_training_ready",
        "video_directory": str(directory.resolve()),
        "counts": dict(sorted(Counter(record["status"] for record in records).items())),
        "records": records,
    }
