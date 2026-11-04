"""Check ASL Citizen class identity, source-derived temporal sampling and score reduction."""

import json
from pathlib import Path

import numpy as np
import pytest

from sign_language.citizen_recognizer import (
    CitizenVideoRecognizer,
    citizen_crop_bounds,
    citizen_frame_indices,
    load_citizen_classes,
    prepare_citizen_frames,
)
from sign_language.video_recognizer import load_candidates


def test_native_vocabulary_keeps_yes_and_no_as_strings():
    from sign_language.config import load_vocabulary

    vocabulary = load_vocabulary(
        Path(__file__).resolve().parents[1] / "configs/vocabulary_aslcitizen.yaml"
    )
    assert "YES" in vocabulary.labels and "NO" in vocabulary.labels
    assert "FINE1" in vocabulary.labels and "FINE2" in vocabulary.labels
    assert vocabulary.dataset == "asl-citizen"


def test_citizen_temporal_stride_centering_and_last_frame_padding():
    assert citizen_frame_indices(1).tolist() == [0] * 64
    assert citizen_frame_indices(68).tolist() == list(range(2, 66))
    assert citizen_frame_indices(96).tolist() == list(range(0, 96, 2)) + [94] * 16
    assert citizen_frame_indices(194).tolist() == list(range(1, 193, 3))
    for count in (0, 601):
        with pytest.raises(ValueError, match="1 to 600"):
            citizen_frame_indices(count)


def test_citizen_input_preserves_bgr_shape_and_wider_camera_crop():
    pytest.importorskip("cv2")
    frame = np.zeros((256, 512, 3), dtype=np.uint8)
    frame[:, 50:100, 0] = 255
    tensor = prepare_citizen_frames([frame])
    assert tensor.shape == (1, 3, 64, 224, 224)
    assert tensor[0, 0].max() == 1
    assert tensor[0, 1].max() == -1
    assert citizen_crop_bounds(640, 480) == (40, 30, 600, 450)


def test_citizen_classes_are_sorted_and_variants_remain_distinct(tmp_path):
    path = tmp_path / "classes.json"
    labels = sorted(["FINE1", "FINE2", "ME"] + [f"SIGN{i:04}" for i in range(2728)])
    path.write_text(json.dumps(labels), encoding="utf-8")
    assert load_citizen_classes(path) == labels
    path.write_text(json.dumps(list(reversed(labels))), encoding="utf-8")
    with pytest.raises(ValueError, match="sorted"):
        load_citizen_classes(path)
    vocabulary = tmp_path / "vocabulary.yaml"
    vocabulary.write_text(
        "schema_version: 1\nlanguage: asl\ndataset: wlasl\nlabels: [me]\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="asl-citizen"):
        load_candidates(vocabulary, labels, dataset="asl-citizen")


def test_citizen_uses_max_upsampled_logits_and_its_own_candidate_order():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    recognizer = object.__new__(CitizenVideoRecognizer)
    recognizer.labels, recognizer.device = ["ME", "YES", "excluded"], "cpu"

    def model(inputs):
        assert inputs.shape == (1, 3, 64, 224, 224)
        return torch.tensor([[[0.0, 8.0], [5.0, 5.0], [100.0, 100.0]]])

    recognizer.model = model
    predictions = recognizer.predict_comparison(
        [np.zeros((256, 256, 3), dtype=np.uint8)], ["YES", "ME"], top_k=2
    )
    assert set(predictions) == {"citizen"}
    assert [guess["gloss"] for guess in predictions["citizen"]] == ["ME", "YES"]
