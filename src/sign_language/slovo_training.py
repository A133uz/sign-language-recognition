"""Evaluation helpers for trained Slovo GRU checkpoints."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sign_language.dataset import write_json
from sign_language.gru_model import SignGRU
from sign_language.slovo_dataset import INPUT_SIZE, MAX_FRAMES, SCHEMA


def _cache_identity(cache):
    cache = Path(cache)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    status = json.loads((cache / "status.json").read_text(encoding="utf-8"))
    labels = json.loads((cache / "labels.json").read_text(encoding="utf-8"))
    if (manifest.get("dataset") != "Slovo Russian Sign Language" or
            status.get("status") != "complete" or status.get("cache_id") != manifest.get("cache_id") or
            manifest.get("identity", {}).get("feature_schema") != SCHEMA or
            manifest.get("input_size") != INPUT_SIZE or manifest.get("max_frames") != MAX_FRAMES or
            labels != manifest.get("labels")):
        raise ValueError("Incomplete or incompatible Slovo feature cache")
    return manifest, labels


def _metrics(model, feature_array, length_array, target_array, indices, device, batch_size):
    if len(indices) == 0:
        raise ValueError("Cannot evaluate an empty Slovo test subset")
    classes = model.classifier.out_features
    confusion = np.zeros((classes, classes), dtype=np.int64)
    loss_sum = correct = correct5 = processed = 0
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(indices), batch_size):
            chosen = indices[offset:offset + batch_size]
            frames = torch.from_numpy(np.asarray(feature_array[chosen], dtype=np.float32)).to(device)
            lengths = torch.from_numpy(np.asarray(length_array[chosen], dtype=np.int64))
            targets = torch.from_numpy(np.asarray(target_array[chosen], dtype=np.int64)).to(device)
            logits = model(frames, lengths)
            loss_sum += float(nn.functional.cross_entropy(logits, targets, reduction="sum"))
            top_ids = logits.topk(min(5, classes), dim=1).indices
            predicted = logits.argmax(1)
            correct += int((predicted == targets).sum())
            correct5 += int((top_ids == targets[:, None]).any(1).sum())
            actual_np = targets.cpu().numpy()
            predicted_np = predicted.cpu().numpy()
            np.add.at(confusion, (actual_np, predicted_np), 1)
            processed += len(chosen)
    support = confusion.sum(1)
    true_positive = np.diag(confusion).astype(np.float64)
    predicted_count = confusion.sum(0).astype(np.float64)
    precision = np.divide(true_positive, predicted_count, out=np.zeros_like(true_positive), where=predicted_count > 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    supported = support > 0
    return {
        "samples": processed, "top1": correct / processed, "top5": correct5 / processed,
        "loss": loss_sum / processed, "supported_classes": int(supported.sum()),
        "total_classes": classes,
        "macro_f1_supported_classes": float(f1[supported].mean()) if supported.any() else None,
        "macro_f1_all_classes": float(f1.mean()),
        "missing_test_classes": int((~supported).sum()),
    }


def evaluate_slovo(checkpoint_path, cache, output, *, device="cuda", batch_size=128):
    """Evaluate all official test clips and the unseen-signer subset separately."""
    checkpoint_path, cache, output = map(Path, (checkpoint_path, cache, output))
    if output.exists():
        raise ValueError(f"Evaluation output already exists: {output}")
    if batch_size < 1 or device not in ("cuda", "cpu"):
        raise ValueError("Invalid evaluation device or batch size")
    manifest, labels = _cache_identity(cache)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if (config.get("dataset") != "slovo" or config.get("cache_id") != manifest["cache_id"] or
            config.get("preprocessing") != SCHEMA or checkpoint["labels"] != labels or
            json.loads((checkpoint_path.parent / "labels.json").read_text(encoding="utf-8")) != labels):
        raise ValueError("Checkpoint, label map, and Slovo cache identity differ")
    for name in ("slovo_dataset.py", "gru_model.py"):
        digest = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        if config.get("source_sha256", {}).get(name) != digest:
            raise ValueError(f"Trained source differs: {name}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu explicitly")
    torch.backends.cudnn.enabled = False
    model = SignGRU(**config["model"]).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    test_features = np.load(cache / "test-features.npy", mmap_mode="r")
    test_lengths = np.load(cache / "test-lengths.npy", mmap_mode="r")
    test_targets = np.load(cache / "test-targets.npy", mmap_mode="r")
    test_index = json.loads((cache / "test-index.json").read_text(encoding="utf-8"))
    unseen_indices = json.loads((cache / "test-unseen-indices.json").read_text(encoding="utf-8"))
    all_indices = np.flatnonzero(test_lengths > 0).astype(np.int64)
    unseen_indices = np.asarray(unseen_indices, dtype=np.int64)
    unseen_indices = unseen_indices[test_lengths[unseen_indices] > 0]
    result = {
        "status": "complete", "dataset": "Slovo Russian Sign Language",
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint["epoch"], "cache_id": manifest["cache_id"],
        "classes": len(labels), "training_performed": False, "augmentation": False,
        "official_flag_test": _metrics(model, test_features, test_lengths, test_targets,
                                        all_indices, device, batch_size),
        "unseen_signer_test": _metrics(model, test_features, test_lengths, test_targets,
                                        unseen_indices, device, batch_size),
        "split_metadata": json.loads((cache / "split.json").read_text(encoding="utf-8")),
        "test_signers_all": len({row["user_id"] for row in test_index}),
        "test_signers_unseen": len({test_index[int(i)]["user_id"] for i in unseen_indices}),
        "test_empty_landmark_rows_excluded": int((test_lengths == 0).sum()),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result
