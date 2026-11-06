"""Download the published hand detector, not a sign classifier."""

import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime, timezone

from sign_language.hands import MODEL, MODEL_URL


def setup_hands() -> dict:
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL.exists():
        partial = MODEL.with_suffix(".task.part")
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, partial.open("wb") as target:
            while block := response.read(1024 * 1024):
                target.write(block)
        with zipfile.ZipFile(partial) as bundle:
            if bundle.testzip() is not None or len(bundle.namelist()) < 2:
                raise ValueError("Invalid MediaPipe hand model bundle")
        partial.replace(MODEL)
    with MODEL.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"url": MODEL_URL, "sha256": digest, "bytes": MODEL.stat().st_size,
              "checked_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "pretrained hand landmark detection only; no sign recognition"}
    MODEL.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
