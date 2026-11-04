"""Train a small hand-sequence classifier and use it with a webcam."""

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sign_language.landmarks import (
    FEATURE_SCHEMAS,
    FEATURE_SIZE,
    POSE_FEATURE_SCHEMA,
    SEQUENCE_LENGTH,
    WORLD_FEATURE_SCHEMA,
    HandExtractor,
    sample_sequence,
)


class SignModel(nn.Module):
    """A GRU consumes a sampled landmark sequence and predicts one gloss."""

    def __init__(self, classes: int, feature_size: int = FEATURE_SIZE):
        super().__init__()
        self.gru = nn.GRU(feature_size, 128, batch_first=True)
        self.classifier = nn.Linear(128, classes)

    def forward(self, sequences):
        _, hidden = self.gru(sequences)
        return self.classifier(hidden[-1])


def svm_vote_scores(features: torch.Tensor, checkpoint: dict) -> torch.Tensor:
    """Return one-vs-one vote fractions for exported linear SVM parameters."""
    flat = svm_inputs(features, checkpoint.get("representation", "raw"))
    x = (flat - checkpoint["mean"]) / checkpoint["scale"]
    if checkpoint["kind"] == "linear_svm":
        decisions = x @ checkpoint["coefficients"].T + checkpoint["intercepts"]
    elif checkpoint["kind"] == "rbf_svm":
        support = checkpoint["support_vectors"]
        distance = (
            x.square().sum(1, keepdim=True) + support.square().sum(1)[None] - 2 * x @ support.T
        ).clamp_min(0)
        kernel = (-checkpoint["gamma"] * distance).exp()
        counts = checkpoint["support_counts"]
        offsets = np.cumsum([0, *counts]).tolist()
        pairs = []
        for left in range(len(counts)):
            for right in range(left + 1, len(counts)):
                a, b = (
                    slice(offsets[left], offsets[left + 1]),
                    slice(offsets[right], offsets[right + 1]),
                )
                pairs.append(
                    kernel[:, a] @ checkpoint["dual_coefficients"][right - 1, a]
                    + kernel[:, b] @ checkpoint["dual_coefficients"][left, b]
                )
        decisions = torch.stack(pairs, dim=1) + checkpoint["intercepts"]
    else:
        raise ValueError("Unsupported SVM checkpoint kind")
    classes = len(checkpoint["labels"])
    votes = torch.zeros((len(x), classes), dtype=torch.float32, device=x.device)
    pair = 0
    for left in range(classes):
        for right in range(left + 1, classes):
            positive = decisions[:, pair] > 0
            votes[:, left] += positive
            votes[:, right] += ~positive
            pair += 1
    return votes / (classes - 1)


def svm_inputs(features: torch.Tensor, representation: str) -> torch.Tensor:
    """Share the exact training/live transform; summarize 3D hand shape and wrist/body location."""
    if representation == "raw":
        return features.flatten(start_dim=1)
    if representation == "body_geometry":
        from sign_language.feature_geometry import body_geometry_inputs

        return body_geometry_inputs(features)
    if representation != "world_summary" or features.shape[1:] != (32, 288):
        raise ValueError("world_summary requires 32-frame world-hand sequences")
    frames = torch.cat(
        (
            features[:, :, 160:288],
            features[:, :, 156:160],
            features[:, :, 63:64],
            features[:, :, 127:128],
        ),
        dim=2,
    )
    return torch.cat(
        [frames[:, start : start + 8].mean(1) for start in (0, 8, 16, 24)]
        + [frames.std(1, correction=0)],
        dim=1,
    )


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def augment_timing(features: np.ndarray) -> np.ndarray:
    """Repeat endpoints to vary gesture timing while preserving temporal direction.

    The caller must pass training examples only. Labels repeat in five complete
    blocks: original, delayed, advanced, slower start and slower end.
    """
    if features.ndim != 3 or features.shape[1:] != (32, 288):
        raise ValueError("Timing augmentation requires 32-frame world-hand sequences")
    indices = (
        np.arange(32),
        np.clip(np.arange(32) - 6, 0, 31),
        np.clip(np.arange(32) + 6, 0, 31),
        np.linspace(0, 24, 32).round().astype(int),
        np.linspace(7, 31, 32).round().astype(int),
    )
    return np.concatenate([features[:, index] for index in indices])


def train_svm(
    features_path: Path,
    output: Path,
    seed: int = 42,
    scale_floor: float = 0.001,
    *,
    representation: str = "raw",
    kernel: str = "linear",
    evaluate_test: bool = True,
    temporal_augmentation: bool = False,
) -> dict:
    """Train an alternative small-data linear SVM; select experiments on validation only."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if output.exists():
        raise FileExistsError(f"Checkpoint already exists: {output}")
    if not np.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("SVM scale floor must be positive and finite")
    with np.load(features_path, allow_pickle=False) as data:
        features, labels, splits = data["features"], data["labels"], data["splits"]
        report = json.loads(str(data["report"]))
    names = report["label_order"]
    feature_size = FEATURE_SCHEMAS.get(report.get("feature_schema"))
    if feature_size is None or features.shape[1:] != (SEQUENCE_LENGTH, feature_size):
        raise ValueError("Feature schema or shape does not match this recognizer")
    if not 3 <= len(names) <= 100:
        raise ValueError("The experimental SVM requires 3 to 100 classes")
    if any(not np.any(splits == split) for split in ("train", "val", "test")):
        raise ValueError("Training, validation and test splits must each have samples")
    if set(labels[splits == "train"].tolist()) != set(range(len(names))):
        raise ValueError("Every class needs at least one training sample")
    if kernel not in ("linear", "rbf"):
        raise ValueError("SVM kernel must be linear or rbf")
    if temporal_augmentation and representation not in ("world_summary", "body_geometry"):
        raise ValueError(
            "Timing augmentation requires world_summary or body_geometry representation"
        )
    flat = svm_inputs(torch.from_numpy(features.astype(np.float32)), representation).numpy()
    training = splits == "train"
    training_flat, training_labels = flat[training], labels[training]
    if temporal_augmentation:
        augmented = augment_timing(features[training].astype(np.float32))
        training_flat = svm_inputs(torch.from_numpy(augmented), representation).numpy()
        training_labels = np.tile(training_labels, 5)
    scaler = StandardScaler().fit(training_flat)
    scale = np.maximum(scaler.scale_, scale_floor)
    classifier = SVC(kernel=kernel, class_weight="balanced", random_state=seed)
    classifier.fit((training_flat - scaler.mean_) / scale, training_labels)
    checkpoint = {
        "kind": "linear_svm" if kernel == "linear" else "rbf_svm",
        "representation": representation,
        "dataset": report.get("dataset", "wlasl"),
        "labels": names,
        "feature_schema": report["feature_schema"],
        "hand_model_sha256": report["input_sha256"]["hand_model"],
        "pose_model_sha256": report["input_sha256"].get("pose_model"),
        "mean": torch.tensor(scaler.mean_, dtype=torch.float32),
        "scale": torch.tensor(scale, dtype=torch.float32),
        "scale_floor": scale_floor,
        "intercepts": torch.tensor(classifier.intercept_, dtype=torch.float32),
        "seed": seed,
        "features_sha256": file_sha256(features_path),
        "test_evaluated": evaluate_test,
        "training_augmentation": "timing-endpoints-v1" if temporal_augmentation else "none",
        "training_examples_after_augmentation": len(training_labels),
    }
    if kernel == "linear":
        checkpoint["coefficients"] = torch.tensor(classifier.coef_, dtype=torch.float32)
    else:
        checkpoint.update(
            {
                "support_vectors": torch.tensor(classifier.support_vectors_, dtype=torch.float32),
                "dual_coefficients": torch.tensor(classifier.dual_coef_, dtype=torch.float32),
                "support_counts": classifier.n_support_.tolist(),
                "gamma": float(classifier._gamma),
            }
        )
    scoring = np.ones(len(splits), dtype=bool) if evaluate_test else splits != "test"
    predictions = np.full(len(splits), -1, dtype=np.int64)
    with torch.no_grad():
        predictions[scoring] = (
            svm_vote_scores(torch.from_numpy(features[scoring].astype(np.float32)), checkpoint)
            .argmax(dim=1)
            .numpy()
        )
    metrics = {}
    for split in ("train", "val", "test") if evaluate_test else ("train", "val"):
        mask = splits == split
        metrics[split] = {
            "correct": int(np.count_nonzero(predictions[mask] == labels[mask])),
            "total": int(mask.sum()),
            "per_class": {
                name: {
                    "correct": int(np.count_nonzero(mask & (labels == i) & (predictions == i))),
                    "total": int(np.count_nonzero(mask & (labels == i))),
                }
                for i, name in enumerate(names)
            },
        }
    checkpoint["metrics"] = metrics
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        torch.save(checkpoint, destination)
    return {
        "device": "cpu",
        "method": checkpoint["kind"],
        "representation": representation,
        "test_evaluated": evaluate_test,
        "training_augmentation": checkpoint["training_augmentation"],
        "training_examples_after_augmentation": len(training_labels),
        "metrics": metrics,
        "missing_glosses": report["missing_glosses"],
        "output": str(output),
    }


def train(
    features_path: Path,
    output: Path,
    epochs: int = 100,
    seed: int = 42,
    device: str = "auto",
    *,
    evaluate_test: bool = True,
) -> dict:
    """Train on prepared data and report held-out metrics without claiming generalization."""
    if output.exists():
        raise FileExistsError(f"Checkpoint already exists: {output}")
    if epochs < 1:
        raise ValueError("Epochs must be positive")
    with np.load(features_path, allow_pickle=False) as data:
        features, labels, splits = data["features"], data["labels"], data["splits"]
        report = json.loads(str(data["report"]))
    names = report["label_order"]
    feature_size = FEATURE_SCHEMAS.get(report.get("feature_schema"))
    if feature_size is None or features.shape[1:] != (SEQUENCE_LENGTH, feature_size):
        raise ValueError("Feature schema or shape does not match this recognizer")
    if any(not np.any(splits == split) for split in ("train", "val", "test")):
        raise ValueError("Training, validation and test splits must each have samples")
    if set(labels[splits == "train"].tolist()) != set(range(len(names))):
        raise ValueError("Every class needs at least one training sample")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected_device == "auto":
        selected_device = "cpu"
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    model = SignModel(len(names), feature_size).to(selected_device)
    x = torch.from_numpy(features.astype(np.float32)).to(selected_device)
    y = torch.from_numpy(labels.astype(np.int64)).to(selected_device)
    train_idx = torch.from_numpy(np.flatnonzero(splits == "train")).to(selected_device)
    val_idx = torch.from_numpy(np.flatnonzero(splits == "val")).to(selected_device)
    test_idx = torch.from_numpy(np.flatnonzero(splits == "test")).to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss = float("inf")
    best_state, best_epoch = None, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x[train_idx]), y[train_idx])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.cross_entropy(model(x[val_idx]), y[val_idx]).item()
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    model.eval()
    metrics = {}
    with torch.no_grad():
        partitions = [("train", train_idx), ("val", val_idx)]
        if evaluate_test:
            partitions.append(("test", test_idx))
        for name, idx in partitions:
            pred = model(x[idx]).argmax(dim=1)
            truth = y[idx]
            per_class = {}
            for class_id, gloss in enumerate(names):
                selected = truth == class_id
                per_class[gloss] = {
                    "correct": int(((pred == truth) & selected).sum().item()),
                    "total": int(selected.sum().item()),
                }
            metrics[name] = {
                "correct": int((pred == truth).sum().item()),
                "total": len(idx),
                "per_class": per_class,
            }
    checkpoint = {
        "model": best_state,
        "dataset": report.get("dataset", "wlasl"),
        "labels": names,
        "feature_schema": report["feature_schema"],
        "feature_size": feature_size,
        "hand_model_sha256": report["input_sha256"]["hand_model"],
        "pose_model_sha256": report["input_sha256"].get("pose_model"),
        "seed": seed,
        "best_epoch": best_epoch,
        "features_sha256": file_sha256(features_path),
        "metrics": metrics,
        "test_evaluated": evaluate_test,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        torch.save(checkpoint, destination)
    return {
        "device": selected_device,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "missing_glosses": report["missing_glosses"],
        "output": str(output),
    }


def prediction_message(
    probabilities, labels: list[str], min_score: float, score_name: str = "score"
) -> str:
    """Display uncertain when the top score falls below an experimental threshold."""
    if not 0 <= min_score <= 1:
        raise ValueError("Minimum score must be between 0 and 1")
    index = int(probabilities.argmax().item())
    score = probabilities[index].item()
    if score_name == "vote":
        if score < min_score:
            return "Uncertain | SPACE: retry"
        return f"{labels[index]} (experimental guess) | SPACE: retry"
    if score < min_score:
        return f"Uncertain (top {score_name} {score:.0%}) | SPACE: retry"
    return f"{labels[index]} ({score:.0%} {score_name}) | SPACE: retry"


def webcam(
    checkpoint_path: Path,
    model_path: Path,
    camera: int = 0,
    seconds: float = 2.0,
    min_score: float = 0.5,
    pose_model_path: Path | None = None,
) -> None:
    """Press space to capture an isolated sign; display the most likely learned label."""
    import cv2

    if seconds <= 0:
        raise ValueError("Capture duration must be positive")
    if not 0 <= min_score <= 1:
        raise ValueError("Minimum score must be between 0 and 1")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    schema = checkpoint.get("feature_schema")
    if schema not in FEATURE_SCHEMAS:
        raise ValueError("Checkpoint uses a different landmark schema")
    with model_path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != checkpoint["hand_model_sha256"]:
            raise ValueError("Landmark model differs from the one used during training")
    if schema in (POSE_FEATURE_SCHEMA, WORLD_FEATURE_SCHEMA):
        if pose_model_path is None:
            raise ValueError("This checkpoint requires --pose-model PATH")
        with pose_model_path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != checkpoint["pose_model_sha256"]:
                raise ValueError("Pose model differs from the one used during training")
    else:
        pose_model_path = None
    feature_size = FEATURE_SCHEMAS[schema]
    if checkpoint.get("kind") in ("linear_svm", "rbf_svm"):
        model = None
    else:
        model = SignModel(len(checkpoint["labels"]), feature_size)
        model.load_state_dict(checkpoint["model"])
        model.eval()
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open webcam {camera}")
    try:
        with HandExtractor(
            model_path, pose_model_path, include_world=schema == WORLD_FEATURE_SCHEMA
        ) as extractor:
            collecting, start, collected, message = False, 0.0, [], "SPACE: capture sign | Q: quit"
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError("Webcam stopped returning frames")
                current_hands = None
                if collecting:
                    current_hands = extractor.extract(frame)
                    collected.append(current_hands)
                    if time.monotonic() - start >= seconds:
                        collecting = False
                        raw_sequence = np.stack(collected)
                        if np.count_nonzero(raw_sequence[:, 63] + raw_sequence[:, 127]) < 4:
                            message = "No hands detected. Try again."
                        else:
                            sequence = sample_sequence(raw_sequence)
                            with torch.no_grad():
                                inputs = torch.from_numpy(sequence[None])
                                if model is None:
                                    probabilities = svm_vote_scores(inputs, checkpoint)[0]
                                else:
                                    probabilities = model(inputs).softmax(dim=1)[0]
                            message = prediction_message(
                                probabilities,
                                checkpoint["labels"],
                                min_score,
                                "vote" if model is None else "score",
                            )
                    else:
                        message = "Recording... perform one sign"
                display = cv2.flip(frame, 1)
                if current_hands is not None:
                    height, width = display.shape[:2]
                    for slot in range(2):
                        offset = slot * 64
                        if current_hands[offset + 63]:
                            for point in range(21):
                                x = 1.0 - current_hands[offset + point * 3]
                                y = current_hands[offset + point * 3 + 1]
                                if 0 <= x <= 1 and 0 <= y <= 1:
                                    cv2.circle(
                                        display,
                                        (int(x * (width - 1)), int(y * (height - 1))),
                                        2,
                                        (0, 255, 255),
                                        -1,
                                    )
                cv2.putText(
                    display, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                cv2.imshow("Sign recognition", display)
                key = cv2.waitKey(1) & 0xFF
                if (
                    key in (ord("q"), 27)
                    or cv2.getWindowProperty("Sign recognition", cv2.WND_PROP_VISIBLE) < 1
                ):
                    break
                if key == ord(" ") and not collecting:
                    collecting, start, collected = True, time.monotonic(), []
    finally:
        capture.release()
        cv2.destroyAllWindows()
