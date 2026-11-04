"""Verify fixed framing geometry, safe failure, and training-only reference fitting."""

import hashlib
import json

import numpy as np
import pytest

from sign_language.body_alignment import align_frames, build_reference, estimate_alignment
from sign_language.landmarks import POSE_FEATURE_SCHEMA


def features():
    values = np.zeros((32, 160), dtype=np.float32)
    pose = values[:, 128:156].reshape(32, 7, 4)
    pose[:, :3, 3] = 1
    pose[:, 0, :2] = [0.5, 0.43]
    pose[:, 1, :2], pose[:, 2, :2] = [0.3, 0.7], [0.7, 0.7]
    return values


def test_alignment_maps_nose_and_span_without_rotating_or_changing_motion():
    reference = {"nose_xy_px": [116, 57], "shoulder_span_px": 103}
    matrix, stats = estimate_alignment(features(), 640, 480, reference)
    nose = matrix @ [320, np.float32(0.43) * 480, 1]
    assert nose == pytest.approx([132, 73], abs=0.0001)
    assert matrix[0, 0] * (np.float32(0.7) - np.float32(0.3)) * 640 == pytest.approx(103)
    assert matrix[0, 1] == matrix[1, 0] == 0
    assert matrix[0, 0] == matrix[1, 1]
    assert stats["valid_pose_samples"] == 32
    pytest.importorskip("cv2")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    aligned, padding = align_frames([frame, frame.copy()], matrix)
    assert np.array_equal(aligned[0], aligned[1])
    assert aligned[0].shape == (256, 256, 3) and padding < 0.35


def test_alignment_rejects_missing_pose_and_excessive_padding():
    reference = {"nose_xy_px": [112, 57], "shoulder_span_px": 103}
    with pytest.raises(ValueError, match="eight"):
        estimate_alignment(np.zeros((32, 160)), 640, 480, reference)
    pytest.importorskip("cv2")
    with pytest.raises(ValueError, match="35%"):
        align_frames(
            [np.zeros((256, 256, 3), dtype=np.uint8)], np.array([[1, 0, 500], [0, 1, 500]])
        )


def test_reference_fits_training_only_and_rejects_changed_pixels(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "1.mp4"
    video.write_bytes(b"verified source")

    class Capture:
        def read(self):
            return True, np.zeros((256, 256, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: Capture())
    archive = tmp_path / "features.npz"
    report = {
        "feature_schema": POSE_FEATURE_SCHEMA,
        "records": [
            {
                "video_id": "1",
                "assigned_split": "train",
                "content_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            },
            {"video_id": "missingVal", "assigned_split": "val"},
            {"video_id": "missingTest", "assigned_split": "test"},
        ],
    }
    np.savez_compressed(
        archive,
        features=np.stack([features()] * 3),
        splits=["train", "val", "test"],
        report=np.array(json.dumps(report)),
    )
    result = build_reference(archive, tmp_path, tmp_path / "reference.json")
    assert result["training_clips"] == 1 and result["training_video_ids"] == ["1"]
    assert result["nose_xy_px"] == pytest.approx([112, 94.08], abs=0.0001)
    video.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="changed"):
        build_reference(archive, tmp_path, tmp_path / "changed_reference.json")
