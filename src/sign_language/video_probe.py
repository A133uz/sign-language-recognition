"""Check small direct MP4 files by decoding one frame with OpenCV."""

import tempfile
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

MAX_VIDEO_BYTES = 16 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 10


def is_direct_mp4(url: object) -> bool:
    """Recognize direct HTTP(S) MP4 links; pages such as YouTube are excluded."""
    if not isinstance(url, str):
        return False
    try:
        parts = urlsplit(url)
        return (
            parts.scheme.lower() in {"http", "https"}
            and bool(parts.hostname)
            and parts.username is None
            and parts.password is None
            and parts.path.lower().endswith(".mp4")
        )
    except ValueError:
        return False


def decode_first_frame(video_bytes: bytes) -> tuple[int, int] | None:
    """Decode one frame from a temporary file and remove it afterward."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required; install the vision profile first.") from exc

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(video_bytes)

        capture = cv2.VideoCapture(str(temporary_path))
        try:
            success, frame = capture.read()
        finally:
            capture.release()
        if not success or frame is None:
            return None
        height, width = frame.shape[:2]
        return width, height
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def probe_video(instance: dict) -> dict:
    """Fetch one direct MP4, within a size/time limit, and try one decoded frame."""
    result = {"video_id": instance.get("video_id"), "source": instance.get("source")}
    url = instance.get("url")
    if not is_direct_mp4(url):
        return {**result, "status": "not_direct_mp4"}

    request = Request(url, headers={"User-Agent": "sign-language-project/0.1"})
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            length = response.headers.get("Content-Length")
            if length and length.isdecimal() and int(length) > MAX_VIDEO_BYTES:
                return {**result, "status": "too_large"}
            data = response.read(MAX_VIDEO_BYTES + 1)
    except HTTPError as exc:
        exc.close()
        return {**result, "status": "http_error", "http_status": exc.code}
    except (URLError, TimeoutError, OSError, HTTPException) as exc:
        return {**result, "status": "request_error", "detail": str(exc)}

    if len(data) > MAX_VIDEO_BYTES:
        return {**result, "status": "too_large"}
    dimensions = decode_first_frame(data)
    if dimensions is None:
        return {**result, "status": "decode_failed", "bytes": len(data)}
    width, height = dimensions
    return {**result, "status": "decoded", "bytes": len(data), "width": width, "height": height}


def probe_gloss(sign: dict, limit: int) -> dict:
    """Probe up to limit direct MP4 examples in their recorded metadata order."""
    candidates = [item for item in sign["instances"] if is_direct_mp4(item.get("url"))]
    return {
        "gloss": sign["gloss"],
        "listed_clips": len(sign["instances"]),
        "direct_mp4_candidates": len(candidates),
        "tested": min(len(candidates), limit),
        "results": [probe_video(item) for item in candidates[:limit]],
    }
