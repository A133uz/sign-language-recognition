"""Check capture-first feature parity, asset checks and tracking rejection for local classifiers."""

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sign_language.landmark_recognizer import LandmarkRecognizer  # noqa: E402
from sign_language.landmarks import WORLD_FEATURE_SCHEMA, CaptureRejected  # noqa: E402


@pytest.mark.parametrize("hands_present", [True, False])
def test_capture_uses_distinct_samples_and_closes_extractor(tmp_path, monkeypatch, hands_present):
    import sign_language.landmark_recognizer as module

    model_files = []
    for name in ("hand.task", "pose.task"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        model_files.append(path)
    hand_model, pose_model = model_files
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "kind": "linear_svm",
            "representation": "world_summary",
            "dataset": "asl-citizen",
            "feature_schema": WORLD_FEATURE_SCHEMA,
            "labels": ["YES", "NO", "ME"],
            "hand_model_sha256": hashlib.sha256(hand_model.read_bytes()).hexdigest(),
            "pose_model_sha256": hashlib.sha256(pose_model.read_bytes()).hexdigest(),
            "mean": torch.zeros(670),
            "scale": torch.ones(670),
            "coefficients": torch.zeros((3, 670)),
            "intercepts": torch.ones(3),
        },
        checkpoint,
    )

    class Extractor:
        calls = 0
        closed = False

        def __init__(self, *args, include_world):
            assert include_world

        def __enter__(self):
            return self

        def __exit__(self, *args):
            Extractor.closed = True

        def extract(self, frame):
            assert frame.shape == (360, 480, 3)
            Extractor.calls += 1
            encoded = np.zeros(288, dtype=np.float32)
            encoded[63] = encoded[223] = int(hands_present)
            return encoded

    monkeypatch.setattr(module, "HandExtractor", Extractor)
    recognizer = LandmarkRecognizer(checkpoint, hand_model=hand_model, pose_model=pose_model)
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
    if hands_present:
        result = recognizer.predict(frames)
        assert result[0]["gloss"] == "YES"
    else:
        with pytest.raises(CaptureRejected, match="Hands tracked"):
            recognizer.predict(frames)
    assert Extractor.calls == 5 and Extractor.closed
    assert recognizer.capture_metadata["frames_with_hands"] == (5 if hands_present else 0)
    with pytest.raises(CaptureRejected, match="distinct"):
        recognizer.predict([frames[0]] * 10)
    assert recognizer.capture_metadata == {}
    hand_model.write_bytes(b"changed model")
    with pytest.raises(ValueError, match="differs"):
        recognizer.predict(frames)


def test_comparison_shares_one_extraction_and_rejects_incompatible_models(monkeypatch):
    import sign_language.landmark_recognizer as module

    sequence = np.zeros((32, 288), dtype=np.float32)
    calls = []

    class Recognizer:
        labels = ["YES", "NO", "ME"]
        schema = WORLD_FEATURE_SCHEMA
        dataset = "asl-citizen"
        device = "cpu"
        checkpoint = {"hand_model_sha256": "hand", "pose_model_sha256": "pose"}
        capture_metadata = {"sampled_frames": 32}
        _rank = module.LandmarkRecognizer._rank

        def __init__(self, path, device):
            self.path = path

        def _capture_sequence(self, frames):
            calls.append(("extract", self.path))
            return sequence

        def predict_features(self, inputs):
            assert np.shares_memory(inputs, sequence)
            calls.append(("score", self.path))
            return torch.tensor([[0.1, 0.2, 0.9] if self.path == "new" else [0.8, 0.1, 0.2]])

    monkeypatch.setattr(module, "LandmarkRecognizer", Recognizer)
    comparison = module.LandmarkComparison("old", "new", "cpu")
    predictions = comparison.predict_comparison([], ["YES", "NO", "ME"])
    assert predictions["landmarks"][0]["gloss"] == "YES"
    assert predictions["comparison_landmarks"][0]["gloss"] == "ME"
    assert calls == [("extract", "old"), ("score", "old"), ("score", "new")]
    assert comparison.capture_metadata == {"sampled_frames": 32}

    class Incompatible(Recognizer):
        def __init__(self, path, device):
            super().__init__(path, device)
            if path == "new":
                self.labels = ["NO", "YES", "ME"]

    monkeypatch.setattr(module, "LandmarkRecognizer", Incompatible)
    with pytest.raises(ValueError, match="share labels"):
        module.LandmarkComparison("old", "new", "cpu")
