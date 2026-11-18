"""Webcam inference for Slovo checkpoints using their hands-only input schema."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
import torch

from sign_language.gru_model import SignGRU
from sign_language.hands import HandTracker
from sign_language.slovo_dataset import INPUT_SIZE, MAX_FRAMES, SCHEMA, normalize_clip


class SlovoRecognizer:
    def __init__(self, checkpoint: Path, models: Path, device="cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; choose --device cpu explicitly")
        self.device = torch.device(device)
        self.checkpoint = Path(checkpoint)
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.config = state["config"]
        adjacent_config = json.loads(self.checkpoint.with_name("config.json").read_text(encoding="utf-8"))
        adjacent_labels = json.loads(self.checkpoint.with_name("labels.json").read_text(encoding="utf-8"))
        if self.config != adjacent_config or state["labels"] != adjacent_labels:
            raise ValueError("Slovo checkpoint and run metadata disagree")
        if (self.config.get("dataset") != "slovo" or self.config.get("preprocessing") != SCHEMA or
                self.config["model"].get("input_size") != INPUT_SIZE or
                self.config.get("max_frames") != MAX_FRAMES):
            raise ValueError("Checkpoint does not use the current Slovo feature schema")
        for name in ("slovo_dataset.py", "gru_model.py"):
            source = Path(__file__).with_name(name).read_bytes()
            digest = hashlib.sha256(source).hexdigest()
            if self.config["source_sha256"].get(name) != digest:
                raise ValueError(f"Trained source differs: {name}")
        self.labels = state["labels"]
        self.names = [name for name, _ in sorted(self.labels.items(), key=lambda item: item[1])]
        self.epoch = state["epoch"]
        torch.backends.cudnn.enabled = False
        self.model = SignGRU(**self.config["model"]).to(self.device).eval()
        self.model.load_state_dict(state["model"], strict=True)
        self.hand_model = Path(models) / "hand_landmarker.task"
        if not self.hand_model.is_file():
            raise FileNotFoundError(self.hand_model)

    def observe_frames(self, frames, timestamps):
        if len(frames) < 8 or len(frames) != len(timestamps):
            raise ValueError("Capture at least eight frames with matching timestamps")
        shape = frames[0].shape
        observations = []
        hands_per_frame = []
        started = time.perf_counter()
        with ExitStack() as resources:
            tracker = resources.enter_context(HandTracker(self.hand_model))
            previous = -1
            for frame, timestamp in zip(frames, timestamps, strict=True):
                if frame.shape != shape or timestamp <= previous:
                    raise ValueError("Frame dimensions changed or timestamps did not increase")
                previous = timestamp
                result = tracker.detect(frame, timestamp)
                groups = {}
                for index, points in enumerate(result.hand_landmarks, 1):
                    if len(points) != 21:
                        raise ValueError("Expected 21 MediaPipe hand points")
                    groups[f"hand {index}"] = [
                        {"x": point.x, "y": point.y, "z": point.z} for point in points
                    ]
                observations.append(groups)
                hands_per_frame.append(len(groups))
        features = normalize_clip(observations, shape[1], shape[0], "webcam-capture")
        metadata = {
            "captured_frames": len(frames),
            "capture_seconds": (timestamps[-1] - timestamps[0]) / 1000,
            "landmark_seconds": time.perf_counter() - started,
            "hands_per_frame": hands_per_frame,
        }
        return features, metadata

    def predict_observations(self, features, hands_per_frame):
        if not any(hands_per_frame):
            raise ValueError("No hands detected; keep at least one signing hand visible")
        features = features.astype(np.float16).astype(np.float32)
        padded = np.zeros((MAX_FRAMES, INPUT_SIZE), dtype=np.float32)
        padded[:len(features)] = features
        batch = torch.from_numpy(padded).unsqueeze(0).to(self.device)
        lengths = torch.tensor([len(features)], dtype=torch.long)
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = self.model(batch, lengths).float()
            else:
                logits = self.model(batch, lengths).float()
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite Slovo classifier output")
        scores, ids = logits.softmax(-1).topk(min(5, len(self.names)), dim=-1)
        return {
            "top5": [{"label": self.names[index], "score": score}
                     for index, score in zip(ids[0].tolist(), scores[0].tolist(), strict=True)],
            "quality": {
                "frames_with_hands": sum(count > 0 for count in hands_per_frame),
                "frames_with_two_hands": sum(count == 2 for count in hands_per_frame),
                "frames_with_body_anchor": 0,
            },
            "input_frames": len(features),
        }

    def predict_frames(self, frames, timestamps):
        started = time.perf_counter()
        features, metadata = self.observe_frames(frames, timestamps)
        result = self.predict_observations(features, metadata.pop("hands_per_frame"))
        return {**result, **metadata, "processing_seconds": time.perf_counter() - started}
