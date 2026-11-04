"""Prepare native ASL Citizen landmarks while preserving the publisher's signer splits."""

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from sign_language.config import load_vocabulary
from sign_language.landmarks import WORLD_FEATURE_SCHEMA, HandExtractor, sample_sequence
from sign_language.prepare import read_video


def load_citizen_records(splits_dir: Path, labels: tuple[str, ...]) -> list[dict]:
    """Reject unsafe paths, duplicate clips and signer leakage across the selected source splits."""
    wanted = set(labels)
    records, seen = [], set()
    signers = {split: set() for split in ("train", "val", "test")}
    for split in signers:
        with (splits_dir / f"{split}.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if set(reader.fieldnames or []) != {
                "Participant ID",
                "Video file",
                "Gloss",
                "ASL-LEX Code",
            }:
                raise ValueError("Unexpected ASL Citizen CSV columns")
            for row in reader:
                if any(not isinstance(value, str) for value in row.values()):
                    raise ValueError("Malformed ASL Citizen CSV row")
                gloss = row["Gloss"].strip()
                if gloss not in wanted:
                    continue
                name, signer = row["Video file"], row["Participant ID"].strip()
                if (
                    not name
                    or Path(name).name != name
                    or not name.endswith(".mp4")
                    or any(char in name for char in ("/", "\\", ":"))
                    or not signer
                ):
                    raise ValueError("Unsafe clip filename or missing signer")
                if name in seen:
                    raise ValueError(f"Duplicate source clip across selected records: {name}")
                seen.add(name)
                signers[split].add(signer)
                records.append(
                    {
                        "video_id": Path(name).stem,
                        "filename": name,
                        "gloss": gloss,
                        "signer_id": signer,
                        "assigned_split": split,
                        "class_id": labels.index(gloss),
                    }
                )
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if signers[left] & signers[right]:
            raise ValueError(f"Signer overlap between {left} and {right}")
    return records


def prepare_citizen(
    splits_dir: Path,
    vocabulary: Path,
    videos: Path,
    hand_model: Path,
    pose_model: Path,
    output: Path,
) -> dict:
    """Keep native labels/splits; reject long clips instead of silently truncating their signs."""
    if output.exists():
        raise FileExistsError(f"Feature output already exists: {output}")
    if not videos.is_dir():
        raise ValueError(f"Video directory does not exist: {videos}")
    vocab = load_vocabulary(vocabulary)
    if vocab.dataset.casefold() != "asl-citizen":
        raise ValueError("Vocabulary must use native asl-citizen labels")
    selected = load_citizen_records(splits_dir, vocab.labels)
    records, features, skipped = [], [], []
    hashes_seen = {}
    with HandExtractor(hand_model, pose_model, include_world=True) as extractor:
        for index, record in enumerate(selected, 1):
            path = videos / record["filename"]
            try:
                frames = read_video(path)
                indexes = np.unique(
                    np.linspace(0, len(frames) - 1, min(32, len(frames))).round().astype(int)
                )
                raw = np.stack([extractor.extract(frames[i]) for i in indexes])
                if np.count_nonzero(raw[:, 63] + raw[:, 127]) < 4:
                    raise ValueError("Hands found in fewer than four distinct sampled frames")
                if np.count_nonzero(raw[:, 223] + raw[:, 287]) < 4:
                    raise ValueError("World hands found in fewer than four distinct sampled frames")
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest in hashes_seen and hashes_seen[digest] != record["assigned_split"]:
                    raise RuntimeError("Identical video bytes appear across source splits")
                hashes_seen[digest] = record["assigned_split"]
                records.append(dict(record, content_sha256=digest, decoded_frames=len(frames)))
                features.append(sample_sequence(raw))
            except (OSError, ValueError) as exc:
                skipped.append(dict(record, reason=str(exc)))
            if index % 20 == 0 or index == len(selected):
                print(
                    f"Prepared {index}/{len(selected)}; accepted {len(records)}",
                    file=sys.stderr,
                    flush=True,
                )
    missing = {
        split: [
            label
            for label in vocab.labels
            if not any(r["assigned_split"] == split and r["gloss"] == label for r in records)
        ]
        for split in ("train", "val", "test")
    }
    if missing["train"]:
        raise ValueError(f"Classes missing from training: {missing['train']}")
    input_paths = {
        "vocabulary": vocabulary,
        "hand_model": hand_model,
        "pose_model": pose_model,
        **{f"{split}_csv": splits_dir / f"{split}.csv" for split in ("train", "val", "test")},
    }
    hashes = {}
    for name, path in input_paths.items():
        with path.open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "status": "native_source_splits_unreviewed_clips",
        "dataset": "asl-citizen",
        "feature_schema": WORLD_FEATURE_SCHEMA,
        "label_order": list(vocab.labels),
        "clip_counts": dict(Counter(r["assigned_split"] for r in records)),
        "missing_glosses": missing,
        "records": records,
        "input_sha256": hashes,
        "skipped": skipped,
        "skipped_counts": dict(Counter(r["reason"] for r in skipped)),
        "signer_counts": {
            s: len({r["signer_id"] for r in records if r["assigned_split"] == s})
            for s in ("train", "val", "test")
        },
        "limitations": [
            "Source signer IDs are preserved, not independently verified.",
            "Clip labels and sign intervals are not certified by landmark extraction.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(
            stream,
            features=np.stack(features),
            labels=np.array([r["class_id"] for r in records]),
            splits=np.array([r["assigned_split"] for r in records]),
            report=np.array(json.dumps(report)),
        )
    return report
