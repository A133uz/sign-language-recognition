"""Optional ASL Citizen I3D experiment with its native class mapping and preprocessing."""

import json
from pathlib import Path

import numpy as np

from sign_language.video_recognizer import VideoRecognizer, candidate_indices

WEIGHTS_SHA256 = "3538319620331d5ba731e0f9ac79161deeb37a895b7680d215864fc00a14477d"
CLASSES_SHA256 = "cdc2b7f77b8d8797c19db24c1b7bf501618db5feb29def583c43b3da47c34910"


def load_citizen_classes(path: Path) -> list[str]:
    """Load sorted native glosses derived from the official training CSV."""
    labels = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(labels, list)
        or len(labels) != 2731
        or any(
            not isinstance(label, str) or not label or label != label.strip() for label in labels
        )
        or labels != sorted(set(labels))
    ):
        raise ValueError("Expected 2,731 unique, sorted ASL Citizen class labels")
    return labels


def citizen_frame_indices(count: int) -> np.ndarray:
    """Match the source loader's centered stride; pad deterministically with the last frame."""
    if count < 1 or count > 600:
        raise ValueError("ASL Citizen input must contain 1 to 600 frames")
    stride = 3 if count >= 160 else 2 if count >= 96 else 1
    start_limit = 160 if stride == 3 else 96 if stride == 2 else 64
    start = int(np.clip((count - 64 * stride) // 2, 0, start_limit))
    indexes = np.arange(start, min(start + 64 * stride, count), stride)
    return np.pad(indexes, (0, 64 - len(indexes)), mode="edge")


def prepare_citizen_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Reproduce source BGR resize/crop/normalization; use explicit last-frame padding."""
    indexes = citizen_frame_indices(len(frames))
    processed = {}
    for index in np.unique(indexes):
        processed[index] = crop_citizen_frame(frames[index])
    array = (np.stack([processed[index] for index in indexes]) / 255.0 * 2 - 1).astype(np.float32)
    return np.ascontiguousarray(array.transpose(3, 0, 1, 2)[None])


def crop_citizen_frame(frame: np.ndarray) -> np.ndarray:
    """Share the source spatial preprocessing between inference and the live preview."""
    import cv2

    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
        raise ValueError("Frames must be nonempty BGR color images")
    original_height, original_width = frame.shape[:2]
    if min(original_height, original_width) < 226:
        scale = 226 / min(original_height, original_width)
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
    if max(original_height, original_width) > 256:
        frame = cv2.resize(frame, (256, 256))
    height, width = frame.shape[:2]
    top, left = int(np.round((height - 224) / 2)), int(np.round((width - 224) / 2))
    return frame[top : top + 224, left : left + 224]


def citizen_crop_bounds(width: int, height: int) -> tuple[int, int, int, int]:
    """Approximate the square-resized camera crop on the original preview."""
    if width < 1 or height < 1:
        raise ValueError("Preview dimensions must be positive")
    if max(width, height) > 256:
        crop_width, crop_height = round(width * 224 / 256), round(height * 224 / 256)
    else:
        scale = max(1, 226 / min(width, height))
        crop_width = crop_height = round(224 / scale)
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    return left, top, left + crop_width, top + crop_height


class CitizenVideoRecognizer(VideoRecognizer):
    """Keep ASL Citizen labels separate; do not silently map WLASL classes or variants."""

    weights_sha256 = WEIGHTS_SHA256
    classes_sha256 = CLASSES_SHA256
    class_count = 2731
    dataset = "asl-citizen"
    comparison_variants = ("citizen",)
    read_classes = staticmethod(load_citizen_classes)
    crop_bounds = staticmethod(citizen_crop_bounds)
    frame_crop = staticmethod(crop_citizen_frame)
    inference_description = (
        "source resize/crop and centered temporal stride to 64 frames; "
        "deterministic last-frame padding; max of linearly upsampled temporal logits"
    )

    def predict(
        self, frames: list[np.ndarray], top_k: int = 5, candidates: list[str] | None = None
    ) -> list[dict]:
        import torch
        from torch.nn import functional as functional

        indices = candidate_indices(self.labels, candidates)
        if not 1 <= top_k <= len(indices):
            raise ValueError("Invalid number of predictions")
        inputs = torch.from_numpy(prepare_citizen_frames(frames)).to(self.device)
        with torch.inference_mode():
            logits = (
                functional.interpolate(
                    self.model(inputs), size=64, mode="linear", align_corners=False
                )
                .max(dim=2)
                .values[0]
                .cpu()
            )
        return self._rank(logits, indices, top_k)

    def predict_comparison(
        self, frames: list[np.ndarray], candidates: list[str], top_k: int = 3
    ) -> dict[str, list[dict]]:
        return {"citizen": self.predict(frames, top_k, candidates)}
