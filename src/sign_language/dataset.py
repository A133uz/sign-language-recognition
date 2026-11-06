"""Official ASL Citizen acquisition and metadata checks; never trains a model."""

import csv
import hashlib
import json
import shutil
import stat
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from sign_language.paths import DATA

URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
ARCHIVE = DATA / "downloads/ASL_Citizen.zip"
EXPECTED_COUNTS = {"train": 40154, "val": 10304, "test": 32941}


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def download_dataset(workers: int = 6) -> dict:
    """Resume an official archive using bounded, independently validated ranges."""
    if not 1 <= workers <= 8:
        raise ValueError("Workers must be 1–8")
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=60) as response:
        total = int(response.headers["Content-Length"])
        etag = response.headers.get("ETag")
    manifest_path = ARCHIVE.with_suffix(".source.json")
    manifest = {"url": URL, "bytes": total, "etag": etag}
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Remote archive changed; existing download must be reviewed")
    write_json(manifest_path, manifest)
    if ARCHIVE.exists():
        if ARCHIVE.stat().st_size != total:
            raise ValueError("Existing archive has the wrong size")
        return manifest
    partial = ARCHIVE.with_suffix(".zip.part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > total:
        raise ValueError("Partial archive exceeds remote size")
    if shutil.disk_usage(ARCHIVE.parent).free < total - offset + 1024**3:
        raise ValueError("Insufficient disk space for the archive")
    ranges = ARCHIVE.parent / "ranges"
    ranges.mkdir(exist_ok=True)

    def response_for(start, end):
        request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
        response = urllib.request.urlopen(request, timeout=90)
        expected = f"bytes {start}-{end}/{total}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            response.close()
            raise ValueError("Server did not honor the exact requested archive range")
        if etag and response.headers.get("ETag") != etag:
            response.close()
            raise ValueError("Archive version changed during download")
        return response

    # Validate an existing partial from an earlier downloader before extending it.
    if offset:
        start = max(0, offset - 65536)
        with response_for(start, offset - 1) as response, partial.open("rb") as stream:
            stream.seek(start)
            if stream.read(offset - start) != response.read():
                raise ValueError("Existing partial does not match the official archive")

    def fetch(bounds):
        start, end = bounds
        path = ranges / f"{start}-{end}.part"
        expected = end - start + 1
        for attempt in range(5):
            try:
                have = path.stat().st_size if path.exists() else 0
                if have > expected:
                    raise ValueError("Oversized partial range")
                if have < expected:
                    with response_for(start + have, end) as response, path.open("ab") as stream:
                        shutil.copyfileobj(response, stream, length=1024 * 1024)
                if path.stat().st_size != expected:
                    raise OSError("Archive range ended early")
                return path
            except (OSError, ValueError):
                if attempt == 4:
                    raise
                time.sleep(min(2**attempt, 8))

    chunk = 128 * 1024**2
    bounds = [(start, min(total - 1, start + chunk - 1)) for start in range(offset, total, chunk)]
    began = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool, partial.open("ab") as stream:
        # Ordered merging keeps .zip.part a valid resumable prefix after interruption.
        for path in pool.map(fetch, bounds):
            with path.open("rb") as source:
                shutil.copyfileobj(source, stream, length=1024 * 1024)
            stream.flush()
            path.unlink()
            size = partial.stat().st_size
            print(f"Archive {size/1024**3:.2f}/{total/1024**3:.2f} GiB "
                  f"({size/total:.1%}); elapsed {time.monotonic()-began:.0f}s", flush=True)
    if partial.stat().st_size != total:
        raise ValueError("Archive size mismatch")
    partial.replace(ARCHIVE)
    return manifest


def safe_member(root: Path, name: str) -> Path:
    relative = PurePosixPath(name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or any(":" in part for part in relative.parts):
        raise ValueError(f"Unsafe archive member: {name}")
    destination = root.joinpath(*relative.parts).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError(f"Archive member escapes destination: {name}")
    return destination


def extract_dataset() -> dict:
    """Read every ZIP member with CRC checking, then install an audited dataset."""
    destination = DATA / "asl_citizen"
    complete = destination / "setup-report.json"
    if complete.exists():
        return json.loads(complete.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    with ARCHIVE.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = archive.infolist()
        needed = sum(item.file_size for item in members)
        if shutil.disk_usage(destination).free < needed + 1024**3:
            raise ValueError("Insufficient disk space for extracted dataset")
        for number, item in enumerate(members, start=1):
            path = safe_member(destination, item.filename)
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Dataset ZIP contains a symbolic link")
            if item.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".extracting")
            with archive.open(item) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            temporary.replace(path)
            if number % 1000 == 0:
                print(f"Extracted and CRC-checked {number}/{len(members)} ZIP entries", flush=True)
        print(f"Extracted and CRC-checked {len(members)} ZIP entries", flush=True)
    result = audit_dataset(destination)
    result.update(archive_sha256=digest, archive_bytes=ARCHIVE.stat().st_size,
                  extracted_bytes=needed, zip_crc_verified=True,
                  video_decoding_verified=False, training_performed=False)
    write_json(complete, result)
    return result


def audit_dataset(root: Path = DATA / "asl_citizen") -> dict:
    """Preserve official splits; reject missing files and participant leakage."""
    train_files = list(root.rglob("train.csv"))
    if len(train_files) != 1:
        raise ValueError("Expected one official splits/train.csv in the extracted dataset")
    splits = train_files[0].parent
    videos = splits.parent / "videos"
    return audit_metadata(splits, videos)


def audit_metadata(splits: Path, videos: Path | None = None) -> dict:
    """Audit official CSVs; check files only when an extracted video root is given."""
    participants, labels, filenames, counts = {}, {}, set(), {}
    for split, expected in EXPECTED_COUNTS.items():
        with (splits / f"{split}.csv").open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            if header[:3] != ["Participant ID", "Video file", "Gloss"]:
                raise ValueError("Unexpected dataset CSV header")
            people, glosses, count = set(), set(), 0
            for row in reader:
                person, filename, gloss = (value.strip() for value in row[:3])
                if not person or not filename or not gloss or filename in filenames:
                    raise ValueError("Blank metadata or duplicate video across splits")
                if videos is not None:
                    path = safe_member(videos, filename)
                    if not path.is_file() or path.stat().st_size == 0:
                        raise ValueError(f"Missing or empty dataset video: {filename}")
                filenames.add(filename)
                people.add(person)
                glosses.add(gloss)
                count += 1
        if count != expected:
            raise ValueError(f"Unexpected {split} count: {count}; expected {expected}")
        counts[split], participants[split], labels[split] = count, people, glosses
    if any(participants[a] & participants[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Participant leakage between official splits")
    if len(labels['train']) != 2731 or any(labels[s] - labels['train'] for s in ('val', 'test')):
        raise ValueError("Unexpected training vocabulary or unseen evaluation label")
    if len(set.union(*participants.values())) != 52:
        raise ValueError("Unexpected number of participants")
    label_map = {label: index for index, label in enumerate(sorted(labels["train"]))}
    write_json(DATA / "metadata/label-map.json", label_map)
    return {"root": str(splits.parent), "counts": counts, "vocabulary": len(label_map),
            "participants_by_split": {s: len(p) for s, p in participants.items()},
            "participant_disjoint": True, "metadata_files_present": True,
            "video_files_present": len(filenames) if videos is not None else None,
            "video_presence_checked": videos is not None}
