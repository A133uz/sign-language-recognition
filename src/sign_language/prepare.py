"""Extract hand sequences from locally obtained, already trimmed WLASL clips."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from sign_language.config import load_vocabulary
from sign_language.landmarks import (
    FEATURE_SCHEMA,
    POSE_FEATURE_SCHEMA,
    HandExtractor,
    sample_sequence,
)
from sign_language.splits import split_with_coverage
from sign_language.wlasl import load_metadata


def read_video(path: Path) -> list[np.ndarray]:
    """Decode and resize frames in order, rejecting unreasonable clip lengths."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError("Video could not be opened")
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            height, width = frame.shape[:2]
            scale = min(1.0, 480 / max(height, width))
            if scale < 1:
                frame = cv2.resize(
                    frame, (max(1, round(width * scale)), max(1, round(height * scale)))
                )
            frames.append(frame)
            if len(frames) > 600:
                raise ValueError("Clip exceeds the 600-frame safety limit")
    finally:
        capture.release()
    if len(frames) < 2:
        raise ValueError("Fewer than two frames decoded")
    return frames


def prepare_wlasl(
    metadata: Path,
    vocabulary: Path,
    videos: Path,
    model: Path,
    output: Path,
    seed: int,
    pose_model: Path | None = None,
) -> dict:
    """Extract selected clips, then split only examples with visible hand landmarks."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not videos.is_dir():
        raise ValueError(f"Video directory does not exist: {videos}")
    vocab = load_vocabulary(vocabulary)
    if vocab.dataset.casefold() != "wlasl":
        raise ValueError("Vocabulary dataset must be WLASL.")
    labels = {label.casefold() for label in vocab.labels}
    signs = [sign for sign in load_metadata(metadata) if sign["gloss"].casefold() in labels]
    selected, features, skipped = [], {}, []
    with HandExtractor(model, pose_model) as extractor:
        for sign in signs:
            usable = []
            for instance in sign["instances"]:
                video_id = instance.get("video_id")
                if (
                    not isinstance(video_id, str)
                    or not video_id.isascii()
                    or not video_id.isalnum()
                ):
                    raise ValueError(f"Invalid video ID: {video_id}")
                path = videos / f"{video_id}.mp4"
                if not path.is_file():
                    skipped.append({"video_id": video_id, "reason": "missing"})
                    continue
                try:
                    frames = read_video(path)
                    indexes = np.linspace(0, len(frames) - 1, 32).round().astype(int)
                    sequence = sample_sequence(
                        np.stack([extractor.extract(frames[i]) for i in indexes])
                    )
                    if int(np.count_nonzero((sequence[:, 63] + sequence[:, 127]) > 0)) < 4:
                        raise ValueError("Hand landmarks found in fewer than four sampled frames")
                    with path.open("rb") as source:
                        digest = hashlib.file_digest(source, "sha256").hexdigest()
                except (OSError, ValueError) as exc:
                    skipped.append({"video_id": video_id, "reason": str(exc)})
                    continue
                if video_id in features:
                    raise ValueError(f"Duplicate video ID: {video_id}")
                features[video_id] = sequence
                usable.append({**instance, "content_sha256": digest})
            selected.append({"gloss": sign["gloss"], "instances": usable})
    report = split_with_coverage(selected, vocab.labels, seed=seed)
    if report["missing_glosses"]["train"]:
        raise ValueError(
            f"Classes missing from training split: {report['missing_glosses']['train']}"
        )
    for record in report["records"]:
        record["availability"] = "decoded_hand_landmarks_unreviewed"
    report["status"] = "experimental_unreviewed_clips"
    report["feature_schema"] = POSE_FEATURE_SCHEMA if pose_model else FEATURE_SCHEMA
    report["skipped"] = skipped
    report["input_sha256"] = {}
    for name, path in (("metadata", metadata), ("vocabulary", vocabulary), ("hand_model", model)):
        with path.open("rb") as source:
            report["input_sha256"][name] = hashlib.file_digest(source, "sha256").hexdigest()
    if pose_model:
        with pose_model.open("rb") as source:
            report["input_sha256"]["pose_model"] = hashlib.file_digest(source, "sha256").hexdigest()
    report["skipped_counts"] = dict(Counter(item["reason"] for item in skipped))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        np.savez_compressed(
            destination,
            features=np.stack([features[r["video_id"]] for r in report["records"]]),
            labels=np.array([r["class_id"] for r in report["records"]], dtype=np.int64),
            splits=np.array([r["assigned_split"] for r in report["records"]]),
            report=np.array(json.dumps(report)),
        )
    return report
