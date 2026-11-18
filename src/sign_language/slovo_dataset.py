"""Slovo RSL landmark import and split metadata.

This path is deliberately separate from the ASL Citizen loader and preprocessor.
It reads the tab-delimited Kaggle annotations export and streams the large
MediaPipe JSON one clip at a time.
"""

import csv
import hashlib
import json
import math
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from json import JSONDecodeError, JSONDecoder
from pathlib import Path

import numpy as np


NO_EVENT = "no_event"
MAX_FRAMES = 128
INPUT_SIZE = 132
SCHEMA = {
    "version": "slovo-hands-xy-local-v1",
    "input_size": INPUT_SIZE,
    "max_frames": MAX_FRAMES,
    "frame_order": "source JSON order; uniformly sample endpoints above max_frames; no upsampling",
    "hand_slots": "left-to-right by wrist x in each frame; a lone hand uses the side of frame center",
    "extra_hands": "deduplicate nearest wrist pair within 0.05 frame units; otherwise retain the most spatially separated pair",
    "hand_features": "per hand: 21 wrist-relative XYZ joints divided by clip median palm scale, wrist image XY, presence mask",
    "coordinates": "x normalized by image width; y multiplied by height/width; z retained from MediaPipe",
    "missing": "zeros plus one explicit per-hand presence flag; no body/pose inputs",
    "label": "annotations.csv text; no_event rows are excluded from isolated-sign recognition",
}
REQUIRED_COLUMNS = {
    "attachment_id", "text", "user_id", "height", "width", "length",
    "train", "begin", "end",
}
HAND_NAME = re.compile(r"hand\s*(\d+)", re.IGNORECASE)


def sha256_file(path: Path, chunk_size=8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def read_annotations(path: Path) -> list[dict]:
    """Read the Kaggle annotations export (currently TSV despite its .csv name)."""
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        header = stream.readline()
        delimiter = "\t" if "\t" in header else ","
        stream.seek(0)
        reader = csv.DictReader(stream, delimiter=delimiter)
        columns = set(reader.fieldnames or ())
        if not REQUIRED_COLUMNS.issubset(columns):
            missing = sorted(REQUIRED_COLUMNS - columns)
            raise ValueError(f"Slovo annotations missing columns: {missing}")
        rows = []
        seen = set()
        for line_number, source in enumerate(reader, 2):
            row = {key: (value or "").strip() for key, value in source.items() if key}
            clip_id = row["attachment_id"].lower()
            if not clip_id or clip_id in seen:
                raise ValueError(f"Blank or duplicate attachment_id at line {line_number}")
            seen.add(clip_id)
            if not row["text"] or not row["user_id"]:
                raise ValueError(f"Blank label or user_id at line {line_number}")
            if row["train"].lower() not in ("true", "false"):
                raise ValueError(f"Invalid train flag at line {line_number}: {row['train']!r}")
            try:
                width, height = int(row["width"]), int(row["height"])
                length = float(row["length"])
            except ValueError as exc:
                raise ValueError(f"Invalid dimensions or length at line {line_number}") from exc
            if width <= 0 or height <= 0 or not math.isfinite(length) or length <= 0:
                raise ValueError(f"Invalid dimensions or length at line {line_number}")
            row.update(attachment_id=clip_id, width=width, height=height, length=length,
                       train=row["train"].lower() == "true")
            rows.append(row)
    if not rows:
        raise ValueError("Slovo annotations contain no rows")
    return rows


def iter_json_object(path: Path, chunk_size=2 * 1024 * 1024):
    """Yield (top-level key, decoded value) from a large JSON object.

    Only the current clip value is held in memory. This avoids materializing
    the 1.2 GB landmark file as one nested Python object.
    """
    decoder = JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as stream:
        buffer = stream.read(chunk_size)
        position = 0
        eof = not buffer

        def refill():
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = stream.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        def whitespace():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                refill()

        def expect(character):
            nonlocal position
            whitespace()
            if position >= len(buffer) or buffer[position] != character:
                raise ValueError(f"Malformed landmark JSON; expected {character!r}")
            position += 1

        def value():
            nonlocal position
            while True:
                whitespace()
                try:
                    result, end = decoder.raw_decode(buffer, position)
                except JSONDecodeError:
                    if eof:
                        raise ValueError("Truncated or malformed landmark JSON")
                    refill()
                else:
                    position = end
                    return result

        expect("{")
        whitespace()
        if position < len(buffer) and buffer[position] == "}":
            position += 1
            return
        while True:
            key = value()
            if not isinstance(key, str):
                raise ValueError("Landmark JSON keys must be strings")
            expect(":")
            record = value()
            yield key.lower(), record
            whitespace()
            if position >= len(buffer):
                refill()
                whitespace()
            if position < len(buffer) and buffer[position] == ",":
                position += 1
                continue
            if position < len(buffer) and buffer[position] == "}":
                position += 1
                break
            raise ValueError("Malformed landmark JSON; expected ',' or '}'")
        while True:
            whitespace()
            if position < len(buffer):
                raise ValueError("Unexpected content after landmark JSON object")
            if eof:
                break
            refill()


def _hand_points(points, clip_id, frame_index, hand_name):
    if not isinstance(points, list) or len(points) != 21:
        raise ValueError(f"{clip_id} frame {frame_index}: {hand_name} must have 21 points")
    result = np.empty((21, 3), dtype=np.float32)
    for index, point in enumerate(points):
        if not isinstance(point, dict) or not all(key in point for key in ("x", "y", "z")):
            raise ValueError(f"{clip_id} frame {frame_index}: malformed {hand_name} point {index}")
        result[index] = (float(point["x"]), float(point["y"]), float(point["z"]))
    if not np.isfinite(result).all():
        raise ValueError(f"{clip_id} frame {frame_index}: nonfinite {hand_name} coordinates")
    return result


def normalize_clip(frames, width: int, height: int, clip_id="<unknown>", max_frames=MAX_FRAMES,
                   stats=None):
    """Convert a Slovo frame list to unpadded [time,132] hand-only features."""
    if not isinstance(frames, list) or not frames or not 1 <= max_frames <= MAX_FRAMES:
        raise ValueError(f"{clip_id}: empty/invalid frame sequence or frame limit")
    if width <= 0 or height <= 0:
        raise ValueError(f"{clip_id}: invalid frame dimensions")
    raw = np.zeros((len(frames), 2, 21, 3), dtype=np.float32)
    present = np.zeros((len(frames), 2), dtype=bool)
    aspect = height / width
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"{clip_id} frame {frame_index}: expected an object")
        hands = []
        for name, points in frame.items():
            match = HAND_NAME.fullmatch(str(name).strip())
            if not match:
                raise ValueError(f"{clip_id} frame {frame_index}: unknown landmark group {name!r}")
            value = _hand_points(points, clip_id, frame_index, name)
            value[:, 1] *= aspect
            xy = value[:, :2]
            area = float((xy[:, 0].max() - xy[:, 0].min()) * (xy[:, 1].max() - xy[:, 1].min()))
            hands.append((float(value[0, 0]), int(match.group(1)), value, area))
        while len(hands) > 2:
            distances = []
            for left in range(len(hands)):
                for right in range(left + 1, len(hands)):
                    delta = np.asarray(hands[left][2][0, :2]) - np.asarray(hands[right][2][0, :2])
                    distances.append((float(np.linalg.norm(delta)), left, right))
            closest, left, right = min(distances)
            if closest < 0.05:
                # Near-coincident wrists are duplicate detections. Keep the
                # larger observed hand and discard the lower-area duplicate.
                remove = left if hands[left][3] < hands[right][3] else right
                if stats is not None:
                    stats["near_duplicate_hand_detections_dropped"] += 1
            else:
                # No duplicate pair is apparent. Retain the pair spanning the
                # widest area of the frame and count the ambiguous selection.
                _, left, right = max(distances)
                remove = next(index for index in range(len(hands)) if index not in (left, right))
                if stats is not None:
                    stats["ambiguous_extra_hand_detections_dropped"] += 1
            if stats is not None:
                stats["frames_with_extra_hands"] += 1
            hands.pop(remove)
        hands.sort(key=lambda item: (item[0], item[1]))
        if len(hands) == 1:
            slots = [0 if hands[0][0] < 0.5 else 1]
        else:
            slots = list(range(len(hands)))
        for (_, _, points, _), slot in zip(hands, slots, strict=True):
            raw[frame_index, slot] = points
            present[frame_index, slot] = True

    palm_indices = (5, 9, 13, 17)
    scales = np.ones(2, dtype=np.float32)
    for slot in (0, 1):
        valid = present[:, slot]
        if valid.any():
            selected = raw[valid, slot]
            distances = np.linalg.norm(selected[:, list(palm_indices), :2] -
                                       selected[:, :1, :2], axis=-1).mean(axis=-1)
            usable = distances[np.isfinite(distances) & (distances >= 0.005)]
            if len(usable):
                scales[slot] = float(np.median(usable))

    features = np.zeros((len(frames), INPUT_SIZE), dtype=np.float32)
    for slot in (0, 1):
        offset = slot * 66
        for frame_index in np.flatnonzero(present[:, slot]):
            points = raw[frame_index, slot]
            wrist = points[0]
            features[frame_index, offset:offset + 63] = ((points - wrist) / scales[slot]).reshape(-1)
            features[frame_index, offset + 63:offset + 65] = wrist[:2]
            features[frame_index, offset + 65] = 1.0
    features = np.clip(features, -8.0, 8.0)
    if len(features) > max_frames:
        indices = np.linspace(0, len(features) - 1, max_frames).round().astype(np.int64)
        features = features[indices]
    if not np.isfinite(features).all():
        raise ValueError(f"{clip_id}: normalized features are nonfinite")
    return features


def choose_validation_users(train_rows, fraction=0.2, seed=42, attempts=512):
    """Choose signer groups for validation while retaining every train class."""
    if not 0.05 <= fraction <= 0.4:
        raise ValueError("Validation fraction must be between 0.05 and 0.4")
    grouped = defaultdict(list)
    label_totals = Counter()
    for row in train_rows:
        grouped[row["user_id"]].append(row)
        label_totals[row["text"]] += 1
    if len(grouped) < 2:
        raise ValueError("At least two training signers are required")
    group_labels = {user: Counter(row["text"] for row in items)
                    for user, items in grouped.items()}
    target = round(len(train_rows) * fraction)
    best = None
    for trial in range(attempts):
        users = list(grouped)
        random.Random(seed + trial).shuffle(users)
        selected, selected_count = set(), 0
        validation_labels = set()
        for user in users:
            if selected_count >= target:
                break
            counts = group_labels[user]
            # Keep at least one training sample for every label after holdout.
            if any(label_totals[label] - counts[label] <= 0 for label in counts):
                continue
            selected.add(user)
            selected_count += len(grouped[user])
            validation_labels.update(counts)
        if not selected:
            continue
        remaining_labels = {row["text"] for row in train_rows if row["user_id"] not in selected}
        if len(remaining_labels) != len(label_totals):
            continue
        score = (len(validation_labels), -abs(selected_count - target), -selected_count)
        if best is None or score > best[0]:
            best = (score, selected, selected_count, validation_labels)
    if best is None:
        raise ValueError("Could not form a signer-disjoint validation split")
    _, users, count, validation_labels = best
    return users, {
        "method": "deterministic signer-group holdout; randomized greedy search",
        "seed": seed, "attempts": attempts, "target_fraction": fraction,
        "actual_samples": count, "actual_fraction": count / len(train_rows),
        "validation_signers": len(users), "training_signers": len(set(r["user_id"] for r in train_rows) - users),
        "validation_classes": len(validation_labels), "total_classes": len(label_totals),
        "validation_missing_classes": sorted(set(label_totals) - validation_labels),
    }


def metadata_summary(rows):
    gesture = [row for row in rows if row["text"] != NO_EVENT]
    train = [row for row in gesture if row["train"]]
    test = [row for row in gesture if not row["train"]]
    train_users = {row["user_id"] for row in train}
    test_users = {row["user_id"] for row in test}
    labels = {row["text"] for row in gesture}
    unseen_test = [row for row in test if row["user_id"] not in train_users]
    unseen_labels = {row["text"] for row in unseen_test}
    no_event = [row for row in rows if row["text"] == NO_EVENT]
    return {
        "annotation_rows": len(rows), "gesture_rows": len(gesture), "no_event_rows": len(no_event),
        "gesture_classes": len(labels), "gesture_train_rows": len(train), "gesture_test_rows": len(test),
        "train_signers": len(train_users), "test_signers": len(test_users),
        "shared_train_test_signers": len(train_users & test_users),
        "unseen_signer_test_rows": len(unseen_test),
        "unseen_signer_test_signers": len({row["user_id"] for row in unseen_test}),
        "unseen_signer_test_classes": len(unseen_labels),
        "unseen_signer_test_missing_classes": sorted(labels - unseen_labels),
        "rows_by_official_flag": dict(Counter(str(row["train"]) for row in rows)),
        "no_event_by_official_flag": dict(Counter(str(row["train"]) for row in no_event)),
    }


def audit_slovo(annotations_path: Path, landmarks_path: Path) -> dict:
    """Validate annotation/landmark joins and summarize both split protocols."""
    annotations_path, landmarks_path = Path(annotations_path), Path(landmarks_path)
    rows = read_annotations(annotations_path)
    summary = metadata_summary(rows)
    by_id = {row["attachment_id"]: row for row in rows}
    gesture_ids = {key for key, row in by_id.items() if row["text"] != NO_EVENT}
    seen, lengths, empty_sequences = set(), [], []
    normalization_stats = Counter()
    unknown, malformed = [], []
    for clip_id, frames in iter_json_object(landmarks_path):
        if clip_id in seen:
            raise ValueError(f"Duplicate top-level landmark ID: {clip_id}")
        seen.add(clip_id)
        row = by_id.get(clip_id)
        if row is None:
            unknown.append(clip_id)
            continue
        if row["text"] == NO_EVENT:
            continue
        if isinstance(frames, list) and not frames:
            empty_sequences.append(clip_id)
            continue
        try:
            normalize_clip(frames, row["width"], row["height"], clip_id,
                           stats=normalization_stats)
        except (TypeError, ValueError, OverflowError) as exc:
            malformed.append({"attachment_id": clip_id, "error": str(exc)})
        if isinstance(frames, list):
            lengths.append(len(frames))
        if len(seen) % 2000 == 0:
            print(f"Audited {len(seen)} landmark records", flush=True)
    missing = sorted(gesture_ids - seen)
    no_event_missing = sorted(key for key, row in by_id.items()
                              if row["text"] == NO_EVENT and key not in seen)
    if malformed or unknown or missing:
        raise ValueError(json.dumps({"malformed_examples": malformed[:5], "unknown_examples": unknown[:5],
                                     "missing_gesture_examples": missing[:5]}, ensure_ascii=False))
    if len(seen & gesture_ids) != len(gesture_ids):
        raise ValueError("Not all gesture annotation IDs matched landmark records")
    summary.update({
        "landmark_json_ids": len(seen), "matched_gesture_ids": len(seen & gesture_ids),
        "json_only_ids": len(unknown), "missing_gesture_landmarks": len(missing),
        "no_event_without_landmarks": len(no_event_missing),
        "no_event_with_landmarks": len(set(by_id) & seen - gesture_ids),
        "frame_count_min": min(lengths) if lengths else None,
        "frame_count_median": float(np.median(lengths)) if lengths else None,
        "frame_count_max": max(lengths) if lengths else None,
        "empty_landmark_sequences": len(empty_sequences),
        "empty_landmark_examples": empty_sequences[:20],
        "extra_hand_handling": dict(normalization_stats),
        "input_schema": SCHEMA,
        "annotations_sha256": sha256_file(annotations_path),
        "landmarks_sha256": sha256_file(landmarks_path),
        "status": "complete",
    })
    return summary


def prepare_cache(annotations_path: Path, landmarks_path: Path, output: Path,
                  *, validation_fraction=0.2, seed=42, max_frames=MAX_FRAMES) -> dict:
    """Create a split-preserving, memory-mapped Slovo GRU cache atomically."""
    annotations_path, landmarks_path, output = map(Path, (annotations_path, landmarks_path, output))
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    if max_frames != MAX_FRAMES:
        raise ValueError(f"This Slovo schema is fixed at {MAX_FRAMES} frames")
    rows = read_annotations(annotations_path)
    gesture_rows = [row for row in rows if row["text"] != NO_EVENT]
    official_train = [row for row in gesture_rows if row["train"]]
    official_test = [row for row in gesture_rows if not row["train"]]
    labels = sorted({row["text"] for row in official_train})
    if not labels or {row["text"] for row in official_test} != set(labels):
        raise ValueError("Official train/test gesture vocabularies differ")
    label_map = {label: index for index, label in enumerate(labels)}
    no_event_ids = {row["attachment_id"] for row in rows if row["text"] == NO_EVENT}
    source_hashes = {
        "annotations_sha256": sha256_file(annotations_path),
        "landmarks_sha256": sha256_file(landmarks_path),
    }
    validation_users, validation = choose_validation_users(official_train, validation_fraction, seed)
    split_rows = {
        "train": [row for row in official_train if row["user_id"] not in validation_users],
        "val": [row for row in official_train if row["user_id"] in validation_users],
        "test": official_test,
    }
    all_train_users = {row["user_id"] for row in official_train}
    unseen_test_ids = {row["attachment_id"] for row in official_test
                       if row["user_id"] not in all_train_users}
    row_locations = {}
    for split, group in split_rows.items():
        for index, row in enumerate(group):
            row_locations[row["attachment_id"]] = (split, index)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{output.name}.tmp-", dir=output.parent))
    arrays = {}
    try:
        for split, group in split_rows.items():
            count = len(group)
            index_rows = []
            for row in group:
                index_rows.append({
                    "attachment_id": row["attachment_id"], "user_id": row["user_id"],
                    "text": row["text"], "class_id": label_map[row["text"]],
                    "official_train": row["train"], "split": split,
                })
            write_json(temporary / f"{split}-index.json", index_rows)
            features = np.lib.format.open_memmap(temporary / f"{split}-features.npy", mode="w+",
                                                  dtype=np.float16, shape=(count, MAX_FRAMES, INPUT_SIZE))
            lengths = np.lib.format.open_memmap(temporary / f"{split}-lengths.npy", mode="w+",
                                                dtype=np.int16, shape=(count,))
            targets = np.lib.format.open_memmap(temporary / f"{split}-targets.npy", mode="w+",
                                                dtype=np.int64, shape=(count,))
            arrays[split] = (features, lengths, targets)
            for index, row in enumerate(group):
                features[index] = 0
                lengths[index] = 0
                targets[index] = label_map[row["text"]]

        by_id = {row["attachment_id"]: row for row in gesture_rows}
        seen, processed, empty_by_split = set(), 0, Counter()
        normalization_stats = Counter()
        for clip_id, frames in iter_json_object(landmarks_path):
            if clip_id in seen:
                raise ValueError(f"Duplicate top-level landmark ID: {clip_id}")
            seen.add(clip_id)
            row = by_id.get(clip_id)
            if row is None:
                if clip_id in no_event_ids:
                    continue
                raise ValueError(f"Landmark ID is absent from gesture metadata: {clip_id}")
            split, index = row_locations[clip_id]
            processed += 1
            if isinstance(frames, list) and not frames:
                empty_by_split[split] += 1
                if processed % 100 == 0:
                    for value in arrays.values():
                        value[0].flush()
                    write_json(temporary / "status.json", {
                        "status": "preparing", "processed_gesture_clips": processed,
                        "expected_gesture_clips": len(gesture_rows),
                        "empty_landmark_clips": dict(empty_by_split),
                    })
                    if processed % 1000 == 0:
                        print(f"Prepared {processed}/{len(gesture_rows)} gesture clips", flush=True)
                continue
            feature = normalize_clip(frames, row["width"], row["height"], clip_id,
                                     stats=normalization_stats)
            features, lengths, targets = arrays[split]
            features[index, :len(feature)] = feature.astype(np.float16)
            lengths[index] = len(feature)
            if processed % 100 == 0:
                for value in arrays.values():
                    value[0].flush()
                write_json(temporary / "status.json", {
                    "status": "preparing", "processed_gesture_clips": processed,
                    "expected_gesture_clips": len(gesture_rows),
                    "empty_landmark_clips": dict(empty_by_split),
                })
                if processed % 1000 == 0:
                    print(f"Prepared {processed}/{len(gesture_rows)} gesture clips", flush=True)

        missing = sorted(set(by_id) - seen)
        if missing or processed != len(gesture_rows):
            raise ValueError(f"Feature import incomplete: {processed}/{len(gesture_rows)}; missing {missing[:5]}")
        for value in arrays.values():
            for array in value:
                array.flush()
        unseen_indices = [i for i, row in enumerate(split_rows["test"])
                          if row["attachment_id"] in unseen_test_ids]
        test_labels = {split_rows["test"][i]["text"] for i in unseen_indices}
        split_info = {
            "official_split": "annotations.csv train boolean preserved",
            "validation": validation,
            "unseen_signer_test": {
                "selection": "official test user_id absent from every official training user_id",
                "rows": len(unseen_indices), "signers": len({split_rows['test'][i]['user_id'] for i in unseen_indices}),
                "classes": len(test_labels), "missing_classes": sorted(set(labels) - test_labels),
            },
            "shared_official_train_test_signers": len(all_train_users &
                {row["user_id"] for row in official_test}),
            "empty_landmark_clips_by_cache_split": dict(empty_by_split),
            "extra_hand_handling": dict(normalization_stats),
            "validation_user_ids": sorted(validation_users),
        }
        write_json(temporary / "test-unseen-indices.json", unseen_indices)
        write_json(temporary / "labels.json", label_map)
        write_json(temporary / "split.json", split_info)
        manifest = {
            "dataset": "Slovo Russian Sign Language",
            "cache_id": hashlib.sha256(json.dumps({
                **source_hashes,
                "schema": SCHEMA, "split": split_info,
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "identity": {
                **source_hashes,
                "feature_schema": SCHEMA,
                "feature_schema_sha256": hashlib.sha256(json.dumps(SCHEMA, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest(),
                "numpy": np.__version__,
            },
            "labels": label_map,
            "counts": {split: len(group) for split, group in split_rows.items()},
            "usable_counts": {split: len(group) - empty_by_split[split]
                              for split, group in split_rows.items()},
            "empty_counts": {split: empty_by_split[split] for split in split_rows},
            "input_size": INPUT_SIZE, "max_frames": MAX_FRAMES,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(temporary / "manifest.json", manifest)
        write_json(temporary / "status.json", {
            "status": "complete", "processed_gesture_clips": processed,
            "usable_gesture_clips": processed - sum(empty_by_split.values()),
            "empty_landmark_clips": dict(empty_by_split),
            "counts": manifest["counts"], "cache_id": manifest["cache_id"],
        })
        for group in arrays.values():
            for array in group:
                array.flush()
                array._mmap.close()
        arrays.clear()
        temporary.replace(output)
        return {"status": "complete", "output": str(output), **manifest["counts"],
                "usable_counts": manifest["usable_counts"],
                "empty_counts": manifest["empty_counts"],
                "classes": len(labels), "validation": validation,
                "unseen_signer_test": split_info["unseen_signer_test"]}
    except BaseException:
        for group in arrays.values():
            for array in group:
                try:
                    array.flush()
                finally:
                    if getattr(array, "_mmap", None) is not None:
                        array._mmap.close()
        arrays.clear()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
