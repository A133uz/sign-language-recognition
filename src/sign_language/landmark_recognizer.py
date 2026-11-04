"""Capture-first landmark inference shared by guided tests and saved-clip evaluation."""

from pathlib import Path

import numpy as np
import torch

from sign_language.landmarks import (
    FEATURE_SCHEMAS,
    WORLD_FEATURE_SCHEMA,
    CaptureRejected,
    HandExtractor,
    sample_sequence,
)
from sign_language.recognizer import SignModel, file_sha256, svm_vote_scores
from sign_language.video_recognizer import candidate_indices
from sign_language.view_quality import extract_capture_features


class LandmarkRecognizer:
    comparison_variants = ("landmarks",)
    inference_description = (
        "32 sampled full-frame RGB MediaPipe inputs at max dimension480; "
        "checkpoint landmark schema and representation; classifier scores are uncalibrated"
    )
    score_description = (
        "Uncalibrated classifier scores; SVM values are vote fractions across checkpoint classes"
    )

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "auto",
        *,
        hand_model: Path = Path("assets/models/hand_landmarker.task"),
        pose_model: Path = Path("assets/models/pose_landmarker_lite.task"),
    ):
        self.checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.labels = self.checkpoint["labels"]
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("Invalid checkpoint labels")
        self.schema = self.checkpoint.get("feature_schema")
        if self.schema not in FEATURE_SCHEMAS:
            raise ValueError("Unsupported landmark schema")
        self.dataset = self.checkpoint.get("dataset", "wlasl")
        self.hand_model, self.pose_model = hand_model, pose_model
        self.is_svm = self.checkpoint.get("kind") in ("linear_svm", "rbf_svm")
        if device not in ("auto", "cpu", "cuda") or self.is_svm and device == "cuda":
            raise ValueError("SVM inference uses CPU; device must be auto or cpu")
        self.device = "cpu" if self.is_svm or device == "auto" else device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self.model = None
        if not self.is_svm:
            self.model = SignModel(len(self.labels), FEATURE_SCHEMAS[self.schema]).to(self.device)
            self.model.load_state_dict(self.checkpoint["model"])
            self.model.eval()
        self.capture_metadata = {}

    def predict_features(self, features: np.ndarray) -> torch.Tensor:
        features = np.asarray(features, dtype=np.float32)
        if (
            features.ndim != 3
            or features.shape[1:] != (32, FEATURE_SCHEMAS[self.schema])
            or not np.isfinite(features).all()
        ):
            raise ValueError("Features do not match the checkpoint schema")
        inputs = torch.from_numpy(features).to(self.device)
        with torch.inference_mode():
            scores = (
                svm_vote_scores(inputs, self.checkpoint)
                if self.is_svm
                else self.model(inputs).softmax(1)
            )
        return scores.cpu()

    def predict(
        self, frames: list[np.ndarray], top_k: int = 3, candidates: list[str] | None = None
    ) -> list[dict]:
        indices = candidate_indices(self.labels, candidates)
        if not 1 <= top_k <= len(indices):
            raise ValueError("Invalid number of predictions")
        sequence = self._capture_sequence(frames)
        return self._rank(self.predict_features(sequence[None])[0], indices, top_k)

    def _capture_sequence(self, frames: list[np.ndarray]) -> np.ndarray:
        """Extract once so compatible checkpoints can compare the identical input."""
        self.capture_metadata = {}
        frames = list({id(frame): frame for frame in frames}.values())
        if len(frames) < 4:
            raise CaptureRejected("Too few distinct camera frames")
        if file_sha256(self.hand_model) != self.checkpoint["hand_model_sha256"]:
            raise ValueError("Hand model differs from training")
        pose_model = self.pose_model if self.schema != "hands-image-xyz-xsorted-v1" else None
        if pose_model and file_sha256(pose_model) != self.checkpoint["pose_model_sha256"]:
            raise ValueError("Pose model differs from training")
        with HandExtractor(
            self.hand_model, pose_model, include_world=self.schema == WORLD_FEATURE_SCHEMA
        ) as extractor:
            raw = extract_capture_features(frames, extractor, unique=True)
        hand_frames = int(np.count_nonzero(raw[:, 63] + raw[:, 127]))
        self.capture_metadata = {
            "sampled_frames": len(raw),
            "frames_with_hands": hand_frames,
            "score_kind": "vote_fraction" if self.is_svm else "uncalibrated_softmax",
        }
        if hand_frames < 4:
            raise CaptureRejected("Hands tracked in fewer than four sampled frames")
        if self.schema == WORLD_FEATURE_SCHEMA and np.count_nonzero(raw[:, 223] + raw[:, 287]) < 4:
            raise CaptureRejected("3D hands tracked in fewer than four sampled frames")
        return sample_sequence(raw)

    def _rank(self, scores: torch.Tensor, indices: list[int], top_k: int) -> list[dict]:
        ranked = sorted(indices, key=lambda index: float(scores[index]), reverse=True)[:top_k]
        return [{"gloss": self.labels[index], "score": float(scores[index])} for index in ranked]

    def predict_comparison(
        self, frames: list[np.ndarray], candidates: list[str], top_k: int = 3
    ) -> dict:
        return {"landmarks": self.predict(frames, top_k, candidates)}

    @staticmethod
    def crop_bounds(width: int, height: int) -> tuple[int, int, int, int]:
        return 0, 0, width - 1, height - 1


class LandmarkComparison:
    """Compare compatible models on one in-memory extraction; never save features."""

    comparison_variants = ("landmarks", "comparison_landmarks")
    inference_description = LandmarkRecognizer.inference_description + "; one shared extraction"
    score_description = LandmarkRecognizer.score_description
    crop_bounds = staticmethod(LandmarkRecognizer.crop_bounds)

    def __init__(self, baseline: Path, comparison: Path, device: str = "auto"):
        self.baseline = LandmarkRecognizer(baseline, device)
        self.comparison = LandmarkRecognizer(comparison, device)
        for field in ("labels", "schema", "dataset", "device"):
            if getattr(self.baseline, field) != getattr(self.comparison, field):
                raise ValueError(
                    "Compared checkpoints must share labels, schema, dataset and device"
                )
            setattr(self, field, getattr(self.baseline, field))
        for key in ("hand_model_sha256", "pose_model_sha256"):
            if self.baseline.checkpoint.get(key) != self.comparison.checkpoint.get(key):
                raise ValueError("Compared checkpoints must use identical landmark extractors")

    @property
    def capture_metadata(self) -> dict:
        return self.baseline.capture_metadata

    def predict_comparison(
        self, frames: list[np.ndarray], candidates: list[str], top_k: int = 3
    ) -> dict:
        indices = candidate_indices(self.labels, candidates)
        if not 1 <= top_k <= len(indices):
            raise ValueError("Invalid number of predictions")
        sequence = self.baseline._capture_sequence(frames)
        return {
            variant: model._rank(model.predict_features(sequence[None])[0], indices, top_k)
            for variant, model in zip(
                self.comparison_variants, (self.baseline, self.comparison), strict=True
            )
        }


def evaluate_landmark_checkpoint(features_path: Path, checkpoint_path: Path, split: str) -> dict:
    """Score a frozen checkpoint on one explicitly selected partition."""
    if split not in ("train", "val", "test"):
        raise ValueError("Unknown split")
    recognizer = LandmarkRecognizer(checkpoint_path, "cpu")
    with np.load(features_path, allow_pickle=False) as data:
        import json

        features, labels, splits = data["features"], data["labels"], data["splits"]
        report = json.loads(str(data["report"]))
    if report["label_order"] != recognizer.labels or report["feature_schema"] != recognizer.schema:
        raise ValueError("Feature class order or schema differs from checkpoint")
    mask = splits == split
    if not mask.any():
        raise ValueError("Selected partition is empty")
    pred = recognizer.predict_features(features[mask]).argmax(1).numpy()
    truth = labels[mask]
    records = [
        dict(record, predicted_gloss=recognizer.labels[int(guess)], correct=bool(guess == target))
        for record, guess, target in zip(
            [r for r in report["records"] if r["assigned_split"] == split], pred, truth, strict=True
        )
    ]
    return {
        "status": "source_split_evaluation_not_webcam_validation",
        "split": split,
        "features_sha256": file_sha256(features_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "training_archive_matches": file_sha256(features_path)
        == recognizer.checkpoint["features_sha256"],
        "correct": int((pred == truth).sum()),
        "total": len(truth),
        "per_class": {
            label: {
                "correct": int(((truth == i) & (pred == i)).sum()),
                "total": int((truth == i).sum()),
            }
            for i, label in enumerate(recognizer.labels)
        },
        "records": records,
    }
