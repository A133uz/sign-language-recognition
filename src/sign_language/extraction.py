"""Resumable, checksummed landmark shards for ASL Citizen. Never trains a model."""

import csv
import contextlib
import hashlib
import importlib.metadata
import json
import multiprocessing
import platform
import random
import shutil
import tempfile
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from sign_language.dataset import EXPECTED_COUNTS, write_json
from sign_language.landmark_models import file_hash, setup_models
from sign_language.raw_landmarks import SCHEMA, extract_job

CLASS_COUNT = 2731


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_rows(root: Path):
    """Validate metadata without repeatedly opening 83k raw videos."""
    all_rows, people, labels, csv_hashes, seen = [], {}, {}, {}, set()
    for split, expected in EXPECTED_COUNTS.items():
        path = root / "splits" / f"{split}.csv"
        csv_hashes[split] = file_hash(path)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames[:3] != ["Participant ID", "Video file", "Gloss"]:
                raise ValueError(f"Unexpected {split} CSV columns")
            rows = list(reader)
        if len(rows) != expected:
            raise ValueError(f"Unexpected {split} sample count: {len(rows)}")
        people[split], labels[split] = set(), set()
        for row in rows:
            person, name, gloss = (row[k].strip() for k in ("Participant ID", "Video file", "Gloss"))
            relative = PurePosixPath(name.replace("\\", "/"))
            if (not person or not gloss or not name or name in seen or
                    relative.is_absolute() or ".." in relative.parts or ":" in name or len(relative.parts) != 1):
                raise ValueError("Blank, duplicate or unsafe video metadata")
            seen.add(name)
            people[split].add(person)
            labels[split].add(gloss)
            all_rows.append({"video_file": name, "participant_id": person, "gloss": gloss, "split": split})
    if any(people[a] & people[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Signer leakage between official splits")
    if len(labels["train"]) != CLASS_COUNT or any(labels[s] != labels["train"] for s in ("val", "test")):
        raise ValueError("Unexpected official vocabulary")
    mapping = {gloss: index for index, gloss in enumerate(sorted(labels["train"]))}
    for row in all_rows:
        row["class_id"] = mapping[row["gloss"]]
    return all_rows, mapping, csv_hashes


def shard_plan(rows, splits, shard_size, limit):
    plan = []
    remaining = limit
    for split in splits:
        selected = sorted((r for r in rows if r["split"] == split), key=lambda r: r["video_file"])
        random.Random(0).shuffle(selected)
        if remaining is not None:
            selected = selected[:remaining]
            remaining -= len(selected)
        for offset in range(0, len(selected), shard_size):
            plan.append((f"{split}-{offset // shard_size:05d}", selected[offset:offset + shard_size]))
    if not plan:
        raise ValueError("No samples selected")
    return plan


def read_completion(output: Path, name: str, run_id: str, rows):
    marker, archive = output / f"{name}.complete.json", output / f"{name}.zip"
    if not marker.exists() or not archive.exists():
        return None
    try:
        data = json.loads(marker.read_text())
        if (data["run_id"] != run_id or data["rows_sha256"] != fingerprint(rows) or
                data["bytes"] != archive.stat().st_size or data["sha256"] != file_hash(archive)):
            return None
        with zipfile.ZipFile(archive) as bundle:
            info = json.loads(bundle.read("records.json"))
            if info["run_id"] != run_id or info["rows_sha256"] != fingerprint(rows):
                return None
            records = info["records"]
            if len(records) != len(rows):
                return None
            for expected, actual in zip(rows, records, strict=True):
                if any(actual.get(key) != value for key, value in expected.items()):
                    return None
                if actual["status"] not in ("ok", "failed"):
                    return None
                if actual["status"] == "ok" and actual["member"] not in bundle.namelist():
                    return None
        return records
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None


def commit_shard(local: Path, output: Path, name: str, run_id: str, rows):
    """Marker is installed last; incomplete or damaged copies are recomputed on resume."""
    digest, size = file_hash(local), local.stat().st_size
    target = output / f"{name}.zip"
    uploading = target.with_suffix(".zip.uploading")
    shutil.copyfile(local, uploading)
    if uploading.stat().st_size != size or file_hash(uploading) != digest:
        raise OSError("Shard copy verification failed; no completion marker written")
    uploading.replace(target)
    write_json(output / f"{name}.complete.json", {
        "run_id": run_id, "rows_sha256": fingerprint(rows), "sha256": digest, "bytes": size,
    })


def summarize(records, planned, complete_shards, total_shards, elapsed):
    ok = [r for r in records if r["status"] == "ok"]
    failed = [r for r in records if r["status"] == "failed"]
    counts = Counter(r["split"] for r in ok)
    finished = complete_shards == total_shards
    return {
        "status": ("complete_with_failures" if failed else "complete") if finished else "partial",
        "planned_clips": planned, "processed_clips": len(records),
        "successful_clips": len(ok), "failed_clips": len(failed),
        "successful_by_split": dict(counts),
        "successful_classes_by_split": {
            s: len({r["class_id"] for r in ok if r["split"] == s}) for s in EXPECTED_COUNTS
        },
        "decoded_frames": sum(r["frames"] for r in ok),
        "frames_with_hands": sum(r["frames_with_hands"] for r in ok),
        "frames_with_pose": sum(r["frames_with_pose"] for r in ok),
        "completed_shards": complete_shards, "total_shards": total_shards,
        "failed_videos": [{k: r[k] for k in ("video_file", "split", "error")} for r in failed],
        "session_elapsed_seconds": elapsed,
        "training_performed": False,
        "note": "tracking counts are detector outputs, not recognition or tracking accuracy",
    }


@contextlib.contextmanager
def output_lock(work_dir: Path, output: Path):
    """Prevent duplicate writers on one host; use one Colab VM per Drive output."""
    work_dir.mkdir(parents=True, exist_ok=True)
    lock_path = work_dir / (fingerprint(str(output.resolve())) + ".lock")
    with lock_path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if platform.system() == "Windows":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another extractor is writing to this output directory") from exc
        try:
            yield
        finally:
            if platform.system() == "Windows":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def extract_dataset_landmarks(root: Path, output: Path, model_dir: Path, work_dir: Path, **options):
    with output_lock(work_dir, output):
        return _extract_dataset_landmarks(root, output, model_dir, work_dir, **options)


def _extract_dataset_landmarks(
    root: Path, output: Path, model_dir: Path, work_dir: Path, *,
    splits=("train", "val", "test"), shard_size=100, limit=None,
    workers=1, include_pose=True, max_shards=None, retry_failed=False, max_frames=3600,
):
    if not 1 <= workers <= 4 or not 1 <= shard_size <= 500:
        raise ValueError("Use 1–4 workers and 1–500 clips per shard")
    if (not splits or len(set(splits)) != len(splits) or any(s not in EXPECTED_COUNTS for s in splits)
            or (limit is not None and limit < 1) or (max_shards is not None and max_shards < 1)
            or max_frames < 1):
        raise ValueError("Invalid split, limit, shard or frame configuration")
    root, output = root.resolve(), output.resolve()
    if not (root / "videos").is_dir():
        raise ValueError(f"Missing videos directory: {root / 'videos'}")
    if output.is_relative_to(root):
        raise ValueError("Keep feature output outside the raw dataset directory")
    rows, labels, csv_hashes = load_rows(root)
    plan = shard_plan(rows, splits, shard_size, limit)
    models, provenance = setup_models(model_dir, include_pose)
    source_hashes = {
        name: hashlib.sha256((Path(__file__).parent / name).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        for name in ("extraction.py", "raw_landmarks.py", "hands.py", "landmark_models.py")
    }
    identity = {
        "schema": SCHEMA, "models": provenance, "csv_sha256": csv_hashes,
        "source_sha256": source_hashes,
        "libraries": {n: importlib.metadata.version(n) for n in ("mediapipe", "numpy", "opencv-contrib-python")},
        "python": platform.python_version_tuple()[:2], "platform": platform.system(),
        "delegate": "CPU", "include_pose": include_pose,
        "splits": list(splits), "shard_size": shard_size, "limit": limit,
        "max_frames": max_frames, "rows_sha256": fingerprint([r for _, group in plan for r in group]),
        "video_content_hashing": False,
    }
    run_id = fingerprint(identity)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run-manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["run_id"] != run_id:
            raise ValueError("Cache configuration differs: choose a new output directory; do not mix feature runs")
    else:
        if any(output.iterdir()):
            raise ValueError("Nonempty output directory without a run manifest")
        write_json(manifest_path, {
            "run_id": run_id, "identity": identity, "created_utc": datetime.now(timezone.utc).isoformat(),
            "planned_clips": sum(len(group) for _, group in plan),
            "total_shards": len(plan), "class_count": len(labels),
        })
    write_json(output / "labels.json", labels)
    write_json(output / "feature-schema.json", SCHEMA)
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Delegate CPU; pose={include_pose}; workers={workers}; output={output}", flush=True)
    print("Run one writer per output directory. Raw video bytes must remain unchanged.", flush=True)
    records, new_records, generated, completed = [], [], 0, 0
    began = time.monotonic()
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) if workers > 1 else None
    try:
        for name, group in plan:
            cached = read_completion(output, name, run_id, group)
            if cached is not None and not (retry_failed and any(r["status"] == "failed" for r in cached)):
                records.extend(cached)
                completed += 1
                continue
            if max_shards is not None and generated >= max_shards:
                continue
            jobs = [(r, str(root / "videos"), models, max_frames) for r in group]
            results = pool.map(extract_job, jobs) if pool is not None else map(extract_job, jobs)
            with tempfile.TemporaryDirectory(prefix="landmark-shard-", dir=work_dir) as temporary:
                path = Path(temporary) / f"{name}.zip"
                shard_records = []
                with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
                    for index, (record, data) in enumerate(results):
                        if data is not None:
                            record["member"] = f"clips/{index:05d}.npz"
                            archive.writestr(record["member"], data)
                        shard_records.append(record)
                        if (index + 1) % 10 == 0 or index + 1 == len(group):
                            print(f"{name}: processed {index + 1}/{len(group)} clips; batch not committed yet", flush=True)
                    archive.writestr("records.json", json.dumps({
                        "run_id": run_id, "rows_sha256": fingerprint(group), "records": shard_records,
                    }, allow_nan=False))
                with zipfile.ZipFile(path) as archive:
                    if archive.testzip() is not None:
                        raise OSError("Local shard CRC failure")
                commit_shard(path, output, name, run_id, group)
            records.extend(shard_records)
            new_records.extend(shard_records)
            generated, completed = generated + 1, completed + 1
            summary = summarize(records, sum(len(g) for _, g in plan), completed, len(plan), time.monotonic()-began)
            write_json(output / "summary.json", summary)
            print(f"Committed {name}: {summary['successful_clips']} ok, {summary['failed_clips']} failed; "
                  f"{completed}/{len(plan)} shards", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    summary = summarize(records, sum(len(g) for _, g in plan), completed, len(plan), time.monotonic()-began)
    summary["output"] = str(output)
    summary["generated_shards_this_session"] = generated
    summary["newly_processed_clips"] = len(new_records)
    summary["newly_decoded_frames"] = sum(r["frames"] for r in new_records if r["status"] == "ok")
    summary["feature_archive_bytes"] = sum(p.stat().st_size for p in output.glob("*.zip"))
    write_json(output / "summary.json", summary)
    return summary
