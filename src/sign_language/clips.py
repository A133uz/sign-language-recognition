"""Extract WLASL's one-based, inclusive frame intervals from source videos."""

import math
import shutil
import tempfile
from pathlib import Path


def extract_interval(source: Path, output: Path, start: int, end: int, fps: float = 25) -> int:
    """Resample native frames onto the annotation FPS, then apply inclusive bounds."""
    import cv2

    if start < 1 or end < start or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid one-based, inclusive frame interval or output FPS")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    writer = None
    temp_path = None
    count = 0
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open source video: {source}")
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("Source FPS is unavailable; frame annotations cannot be aligned")
        source_index = -1
        frame = None
        for index in range(start, end + 1):
            # Annotation frame 1 is time zero. Choose the native frame at or
            # immediately before each annotation timestamp, repeating if needed.
            target = math.floor((index - 1) * source_fps / fps + 1e-8)
            while source_index < target:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise ValueError(f"Source ended before annotated frame {end}")
                source_index += 1
            if writer is None:
                height, width = frame.shape[:2]
                with tempfile.NamedTemporaryFile(
                    suffix=".mp4", dir=output.parent, delete=False
                ) as f:
                    temp_path = Path(f.name)
                writer = cv2.VideoWriter(
                    str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError("Cannot create MP4 video writer")
            writer.write(frame)
            count += 1
        writer.release()
        writer = None
        with temp_path.open("rb") as source_file:
            with output.open("xb") as destination:
                try:
                    shutil.copyfileobj(source_file, destination)
                except Exception:
                    destination.close()
                    output.unlink(missing_ok=True)
                    raise
        return count
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
