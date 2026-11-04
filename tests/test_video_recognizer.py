"""Validate I3D preprocessing and label ordering without external weights."""

import json

import numpy as np
import pytest

pytest.importorskip("cv2")

from sign_language.video_recognizer import (  # noqa: E402
    VideoRecognizer,
    candidate_indices,
    load_classes,
    prepare_frames,
    preview_crop_bounds,
    read_video_clip,
    sample_live_frames,
)


def test_prepare_video_preserves_bgr_and_repeats_short_sequences():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :, 0] = 255
    result = prepare_frames([frame])
    assert result.shape == (1, 3, 16, 224, 224)
    assert result[0, 0].min() == 1
    assert result[0, 1].max() == -1
    with pytest.raises(ValueError):
        prepare_frames([])


def test_square_comparison_preserves_wider_horizontal_field():
    frame = np.zeros((256, 512, 3), dtype=np.uint8)
    frame[:, 50:100, 0] = 255
    assert prepare_frames([frame])[0, 0].max() == -1
    assert prepare_frames([frame], resize="square")[0, 0].max() == 1
    with pytest.raises(ValueError, match="Resize mode"):
        prepare_frames([frame], resize="unknown")


def test_comparison_pools_time_before_restricted_ranking():
    torch = pytest.importorskip("torch")
    recognizer = object.__new__(VideoRecognizer)
    recognizer.labels = ["hello", "no", "excluded"]
    calls = []

    def logits(frames, resize="aspect"):
        calls.append(resize)
        return torch.tensor([[0.0, 8.0], [5.0, 5.0], [100.0, 100.0]])

    recognizer._video_logits = logits
    predictions = recognizer.predict_comparison([], ["hello", "no"], top_k=2)
    assert calls == ["aspect", "square"]
    assert predictions["aspect_mean"][0]["gloss"] == "no"
    assert predictions["aspect_max"][0]["gloss"] == "hello"
    assert recognizer.predict([], 2, ["hello", "no"])[0]["gloss"] == "no"


def test_class_mapping_preserves_spaces_and_checks_ids(tmp_path):
    path = tmp_path / "classes.txt"
    path.write_text("\n".join(f"{i}\tword {i}" for i in range(2000)), encoding="utf-8")
    assert load_classes(path)[10] == "word 10"
    path.write_text("0 hello\n0 goodbye\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_classes(path)


def test_pretrained_assets_must_match_known_hashes(tmp_path):
    pytest.importorskip("torch")
    weights = tmp_path / "wrong.pt"
    weights.write_bytes(b"wrong file")
    with pytest.raises(ValueError, match="checksum"):
        VideoRecognizer(weights, tmp_path / "classes.txt", "cpu")


def test_restricted_vocabulary_maps_labels_and_rejects_unknown_or_duplicate():
    labels = ["hello", "thank you", "goodbye"]
    assert candidate_indices(labels, None) == [0, 1, 2]
    assert candidate_indices(labels, ["Thank You", "hello"]) == [1, 0]
    with pytest.raises(ValueError, match="missing"):
        candidate_indices(labels, ["unknown"])
    with pytest.raises(ValueError, match="duplicated"):
        candidate_indices(labels, ["hello", "HELLO"])


def test_preview_guide_matches_center_crop():
    assert preview_crop_bounds(640, 480) == (110, 30, 530, 450)
    assert preview_crop_bounds(480, 640) == (30, 110, 450, 530)
    with pytest.raises(ValueError):
        preview_crop_bounds(0, 480)


def test_live_frames_sample_by_time_and_reject_invalid_timestamps():
    frames = [np.full((1, 1, 3), i, dtype=np.uint8) for i in range(3)]
    sampled = sample_live_frames(frames, [0.0, 0.1, 1.0])
    assert len(sampled) == 16
    assert sampled[0][0, 0, 0] == 0
    assert sampled[-1][0, 0, 0] == 2
    assert sum(int(frame[0, 0, 0]) == 2 for frame in sampled) > 4
    with pytest.raises(ValueError, match="timestamps"):
        sample_live_frames(frames, [0.0, 0.4, 0.2])


def test_local_clip_predict_cli_and_frame_limit(tmp_path, monkeypatch, capsys):
    import cv2

    from sign_language.cli import main

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for value in (30, 60, 90):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()
    frames = read_video_clip(path)
    assert len(frames) == 3
    assert frames[0].shape[:2] == (256, 341)
    with pytest.raises(ValueError, match="frame limit"):
        read_video_clip(path, max_frames=2)

    class StubRecognizer:
        def __init__(self, weights, classes, device):
            self.labels = ["hello"]
            self.device = device

        def predict(self, frames, top_k, candidates):
            assert len(frames) == 3
            return [{"gloss": "hello", "score": 1.0}]

    monkeypatch.setattr("sign_language.video_recognizer.VideoRecognizer", StubRecognizer)
    assert main(["predict-i3d", str(path), "--device", "cpu"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["decoded_frames"] == 3
    assert result["scores"] == "uncalibrated"
    assert result["predictions"][0]["gloss"] == "hello"
