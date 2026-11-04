"""Experimental inference with pretrained WLASL2000 I3D video weights."""

import hashlib
import time
from pathlib import Path

import numpy as np

WEIGHTS_SHA256 = "243a19e6deef3becffbfc5b7dd8adb32916c8bee482565ca243c082584732620"
CLASSES_SHA256 = "2c131fb7f052340aa7ae8579c8cae9746028740b8e62efeb957a7790dbefcf4a"


def load_classes(path: Path) -> list[str]:
    """Load the model's ordered class IDs, retaining spaces in glosses."""
    items = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        index, label = line.split(maxsplit=1)
        index = int(index)
        if index in items or not label.strip():
            raise ValueError("Invalid or duplicate class mapping")
        items[index] = label.strip()
    if set(items) != set(range(2000)):
        raise ValueError("Expected exactly the 2,000 ordered WLASL class IDs")
    return [items[i] for i in range(2000)]


def resize_frame(frame: np.ndarray, mode: str = "aspect") -> np.ndarray:
    """Resize a BGR frame; keep aspect ratio unless an explicit comparison requests square."""
    import cv2

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frames must be BGR color images")
    height, width = frame.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Frames must not be empty")
    if mode == "square":
        return cv2.resize(frame, (256, 256))
    if mode != "aspect":
        raise ValueError("Resize mode must be aspect or square")
    scale = 256 / min(height, width)
    return cv2.resize(frame, (max(256, round(width * scale)), max(256, round(height * scale))))


def read_video_clip(path: Path, max_frames: int = 600) -> list[np.ndarray]:
    """Decode a short local video without opening a camera or retaining raw-size frames."""
    import cv2

    if max_frames < 1:
        raise ValueError("Frame limit must be positive")
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if len(frames) >= max_frames:
                raise ValueError(f"Video exceeds the {max_frames}-frame limit")
            frames.append(resize_frame(frame))
    finally:
        capture.release()
    if not frames:
        raise ValueError("Video contains no decodable frames")
    return frames


def prepare_frames(
    frames: list[np.ndarray], limit: int = 64, *, resize: str = "aspect"
) -> np.ndarray:
    """Resize BGR frames, center crop and uniformly sample the isolated sign."""
    if not frames or limit < 16:
        raise ValueError("At least one frame and a frame limit of at least 16 are required")
    indexes = np.linspace(0, len(frames) - 1, min(limit, max(16, len(frames)))).round().astype(int)
    processed = []
    for i in indexes:
        processed.append(crop_frame(frames[i], resize))
    array = np.stack(processed).astype(np.float32) / 127.5 - 1
    return np.ascontiguousarray(array.transpose(3, 0, 1, 2)[None])


def crop_frame(frame: np.ndarray, resize: str = "aspect") -> np.ndarray:
    """Return the exact spatial crop used for each frame of an I3D input."""
    resized = resize_frame(frame, resize)
    height, width = resized.shape[:2]
    top, left = (height - 224) // 2, (width - 224) // 2
    return resized[top : top + 224, left : left + 224]


def sample_live_frames(
    frames: list[np.ndarray], timestamps: list[float], limit: int = 64
) -> list[np.ndarray]:
    """Sample webcam frames by elapsed time when their arrival rate varies."""
    times = np.asarray(timestamps, dtype=np.float64)
    if (
        len(frames) == 0
        or len(frames) != len(times)
        or limit < 16
        or not np.isfinite(times).all()
        or np.any(np.diff(times) < 0)
    ):
        raise ValueError("Live frames need valid, ordered timestamps and a limit of at least 16")
    desired = np.linspace(times[0], times[-1], min(limit, max(16, len(frames))))
    right = np.clip(np.searchsorted(times, desired), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(desired - times[left] <= times[right] - desired, left, right)
    return [frames[i] for i in nearest]


class VideoRecognizer:
    """Run the 2,000-class video model with its supplied class order."""

    weights_sha256 = WEIGHTS_SHA256
    classes_sha256 = CLASSES_SHA256
    class_count = 2000
    dataset = "wlasl"
    comparison_variants = ("aspect_mean", "aspect_max", "square_mean", "square_max")
    inference_description = "aspect256 or square256, center crop224, mean or max temporal logits"
    read_classes = staticmethod(load_classes)
    frame_crop = staticmethod(crop_frame)

    def __init__(self, weights: Path, classes: Path, device: str = "auto"):
        import torch

        from sign_language._vendor.i3d import InceptionI3d

        for path, expected in ((weights, self.weights_sha256), (classes, self.classes_sha256)):
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != expected:
                raise ValueError(f"Unexpected pretrained model asset checksum: {path}")
        self.labels = self.read_classes(classes)
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("Device must be auto, cpu or cuda")
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self.model = InceptionI3d(num_classes=self.class_count)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()

    def predict(
        self, frames: list[np.ndarray], top_k: int = 5, candidates: list[str] | None = None
    ) -> list[dict]:
        import torch

        indices = candidate_indices(self.labels, candidates)
        if not 1 <= top_k <= len(indices):
            raise ValueError("Invalid number of predictions")
        with torch.inference_mode():
            logits = self._video_logits(frames).mean(dim=1)
        return self._rank(logits, indices, top_k)

    def _video_logits(self, frames: list[np.ndarray], resize: str = "aspect"):
        """Return time-resolved CPU logits, synchronizing GPU work before reporting latency."""
        import torch

        inputs = torch.from_numpy(prepare_frames(frames, resize=resize)).to(self.device)
        with torch.inference_mode():
            return self.model(inputs)[0].cpu()

    def extract_embedding(self, frames: list[np.ndarray]) -> np.ndarray:
        """Return a frozen 1,024-value video descriptor using the default preprocessing."""
        import torch

        inputs = torch.from_numpy(prepare_frames(frames)).to(self.device)
        with torch.inference_mode():
            embedding = self.model.extract_features(inputs).mean(dim=(2, 3, 4))[0].cpu().numpy()
        if embedding.shape != (1024,) or not np.isfinite(embedding).all():
            raise ValueError("Invalid I3D video descriptor")
        return embedding

    def _rank(self, logits, indices: list[int], top_k: int) -> list[dict]:
        scores = logits[indices].softmax(dim=0)
        values, positions = scores.topk(top_k)
        return [
            {"gloss": self.labels[indices[position]], "score": float(score)}
            for score, position in zip(values.cpu().tolist(), positions.cpu().tolist(), strict=True)
        ]

    def predict_comparison(
        self, frames: list[np.ndarray], candidates: list[str], top_k: int = 3
    ) -> dict[str, list[dict]]:
        """Compare two resizes and temporal pooling on the same capture; no model updates."""
        indices = candidate_indices(self.labels, candidates)
        if not 1 <= top_k <= len(indices):
            raise ValueError("Invalid number of predictions")
        results = {}
        for resize in ("aspect", "square"):
            logits = self._video_logits(frames, resize)
            results[f"{resize}_mean"] = self._rank(logits.mean(dim=1), indices, top_k)
            results[f"{resize}_max"] = self._rank(logits.max(dim=1).values, indices, top_k)
        return results


def candidate_indices(labels: list[str], candidates: list[str] | None) -> list[int]:
    """Map an optional explicitly selected vocabulary to unique model classes."""
    if candidates is None:
        return list(range(len(labels)))
    indices = []
    for candidate in candidates:
        matches = [i for i, label in enumerate(labels) if label.casefold() == candidate.casefold()]
        if len(matches) != 1 or matches[0] in indices:
            raise ValueError(f"Vocabulary gloss is missing, ambiguous, or duplicated: {candidate}")
        indices.append(matches[0])
    if not indices:
        raise ValueError("Vocabulary must contain at least one class")
    return indices


def load_candidates(
    path: Path | None, labels: list[str], *, dataset: str = "wlasl"
) -> list[str] | None:
    """Load an optional WLASL vocabulary and check it against the model labels."""
    if path is None:
        return None
    from sign_language.config import load_vocabulary

    config = load_vocabulary(path)
    if config.dataset.casefold() != dataset.casefold():
        raise ValueError(f"Vocabulary dataset must be {dataset}")
    candidates = list(config.labels)
    candidate_indices(labels, candidates)
    return candidates


def preview_crop_bounds(width: int, height: int) -> tuple[int, int, int, int]:
    """Approximate the model's center crop on the original, mirrored preview."""
    if width <= 0 or height <= 0:
        raise ValueError("Preview dimensions must be positive")
    side = round(min(width, height) * 224 / 256)
    left, top = (width - side) // 2, (height - side) // 2
    return left, top, left + side, top + side


def webcam_video(
    weights: Path,
    classes: Path,
    camera: int = 0,
    seconds: float = 2.0,
    device: str = "auto",
    vocabulary: Path | None = None,
) -> None:
    """Capture an isolated sign on Space and show the top three candidate glosses."""
    import cv2

    if not np.isfinite(seconds) or seconds <= 0 or seconds > 10:
        raise ValueError("Capture duration must be greater than zero and at most 10 seconds")
    recognizer = VideoRecognizer(weights, classes, device)
    candidates = load_candidates(vocabulary, recognizer.labels)
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open webcam {camera}")
    collecting, started, frames, timestamps = False, 0.0, [], []
    header = "PILOT ONLY - guesses from selected signs" if candidates else "WLASL2000 guesses"
    messages = [header, "SPACE: capture sign | Q: quit"]
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Webcam stopped returning frames")
            if collecting:
                frames.append(frame.copy())
                now = time.monotonic()
                timestamps.append(now)
                if now - started >= seconds:
                    collecting = False
                    status = cv2.flip(frame, 1)
                    cv2.putText(
                        status,
                        "Analyzing...",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("WLASL video recognition", status)
                    cv2.waitKey(1)
                    predictions = recognizer.predict(
                        sample_live_frames(frames, timestamps),
                        top_k=min(3, len(candidates)) if candidates else 3,
                        candidates=candidates,
                    )
                    messages = [header] + [
                        f"{i}. {item['gloss']}" for i, item in enumerate(predictions, 1)
                    ]
                else:
                    messages = [header, "Recording... perform one sign"]
            display = cv2.flip(frame, 1)
            height, width = display.shape[:2]
            left, top, right, bottom = preview_crop_bounds(width, height)
            cv2.rectangle(display, (left, top), (right, bottom), (0, 190, 190), 1)
            for row, message in enumerate(messages):
                cv2.putText(
                    display,
                    message,
                    (10, 30 + row * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow("WLASL video recognition", display)
            key = cv2.waitKey(1) & 0xFF
            if (
                key in (ord("q"), 27)
                or cv2.getWindowProperty("WLASL video recognition", cv2.WND_PROP_VISIBLE) < 1
            ):
                break
            if key == ord(" ") and not collecting:
                collecting, started, frames, timestamps = True, time.monotonic(), [], []
    finally:
        capture.release()
        cv2.destroyAllWindows()
