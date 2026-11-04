"""Summarize pretrained video-model predictions against local WLASL metadata."""

import hashlib
from collections import Counter
from pathlib import Path

from sign_language.video_recognizer import candidate_indices, read_video_clip
from sign_language.wlasl import find_sign


def evaluate_wlasl_videos(
    signs: list[dict],
    vocabulary: tuple[str, ...],
    videos: Path,
    recognizer,
    *,
    source_split: str,
    restricted: bool,
    limit: int | None = None,
) -> dict:
    """Evaluate existing local clips; missing clips are counted and never guessed."""
    if source_split not in ("train", "val", "test"):
        raise ValueError("Source split must be train, val or test")
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")
    if not videos.is_dir():
        raise ValueError(f"Video directory does not exist: {videos}")
    labels = list(vocabulary)
    candidate_indices(recognizer.labels, labels)
    records = []
    seen = set()
    exhausted_limit = False
    for label in labels:
        sign = find_sign(signs, label)
        if sign is None:
            raise ValueError(f"Vocabulary gloss not found in metadata: {label}")
        for instance in sign["instances"]:
            if instance.get("split") != source_split:
                continue
            if limit is not None and len(records) >= limit:
                exhausted_limit = True
                break
            video_id = instance.get("video_id")
            if not isinstance(video_id, str) or not video_id.isascii() or not video_id.isalnum():
                raise ValueError(f"Invalid video ID: {video_id}")
            if video_id in seen:
                raise ValueError(f"Duplicate video ID in selected metadata: {video_id}")
            seen.add(video_id)
            record = {
                "video_id": video_id,
                "gloss": sign["gloss"],
                "signer_id": instance.get("signer_id"),
            }
            path = videos / f"{video_id}.mp4"
            if not path.is_file():
                records.append({**record, "status": "missing"})
                continue
            try:
                with path.open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                frames = read_video_clip(path)
            except (OSError, ValueError) as exc:
                records.append({**record, "status": "decode_failed", "detail": str(exc)})
                continue
            guesses = recognizer.predict(
                frames, min(3, len(labels)) if restricted else 3, labels if restricted else None
            )
            predicted = [guess["gloss"] for guess in guesses]
            records.append(
                {
                    **record,
                    "status": "predicted",
                    "sha256": digest,
                    "decoded_frames": len(frames),
                    "predictions": guesses,
                    "top1_correct": predicted[0].casefold() == label.casefold(),
                    "top3_correct": label.casefold() in [item.casefold() for item in predicted],
                }
            )
        if exhausted_limit:
            break
    counts = Counter(record["status"] for record in records)
    predicted_records = [record for record in records if record["status"] == "predicted"]
    per_gloss = []
    for label in labels:
        subset = [record for record in records if record["gloss"].casefold() == label.casefold()]
        done = [record for record in subset if record["status"] == "predicted"]
        per_gloss.append(
            {
                "gloss": label,
                "listed": len(subset),
                "predicted": len(done),
                "top1_correct": sum(record["top1_correct"] for record in done),
                "top3_correct": sum(record["top3_correct"] for record in done),
            }
        )
    return {
        "status": "exploratory_not_independent",
        "reason": (
            "The pretrained model may have seen these videos or signers; "
            "original splits overlap by signer ID."
        ),
        "source_split": source_split,
        "vocabulary_mode": "restricted" if restricted else "full_wlasl2000",
        "limit": limit,
        "limit_reached": exhausted_limit,
        "counts": dict(counts),
        "top1_correct": sum(record["top1_correct"] for record in predicted_records),
        "top3_correct": sum(record["top3_correct"] for record in predicted_records),
        "per_gloss": per_gloss,
        "records": records,
    }
