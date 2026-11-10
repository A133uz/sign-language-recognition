"""Versioned official task bundles for offline extraction; no training."""

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from sign_language.hands import MODEL_URL

POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def setup_models(directory: Path, include_pose=True):
    directory.mkdir(parents=True, exist_ok=True)
    specs = {"hands": ("hand_landmarker.task", MODEL_URL)}
    if include_pose:
        specs["pose"] = ("pose_landmarker_lite.task", POSE_URL)
    paths, provenance = {}, {}
    for name, (filename, url) in specs.items():
        path = directory / filename
        if not path.exists():
            partial = path.with_suffix(".task.part")
            with urllib.request.urlopen(url, timeout=90) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            with zipfile.ZipFile(partial) as bundle:
                if bundle.testzip() is not None or len(bundle.namelist()) < 2:
                    raise ValueError(f"Invalid official model bundle: {name}")
            partial.replace(path)
        paths[name] = str(path.resolve())
        provenance[name] = {"url": url, "sha256": file_hash(path), "bytes": path.stat().st_size}
    return paths, provenance
