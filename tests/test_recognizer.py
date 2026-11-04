"""Exercise training and saved checkpoint on a small synthetic feature set."""

import json

import numpy as np
import pytest

from sign_language.landmarks import FEATURE_SCHEMA, FEATURE_SIZE, SEQUENCE_LENGTH


def test_train_saves_loadable_checkpoint_and_metrics(tmp_path):
    torch = pytest.importorskip("torch")
    from sign_language.recognizer import SignModel, train

    x = np.zeros((6, SEQUENCE_LENGTH, FEATURE_SIZE), dtype=np.float32)
    x[1::2, :, 0] = 1
    report = {
        "feature_schema": FEATURE_SCHEMA,
        "label_order": ["one", "two"],
        "input_sha256": {"hand_model": "test-model-hash"},
        "missing_glosses": {"train": [], "val": [], "test": []},
    }
    features = tmp_path / "features.npz"
    np.savez_compressed(
        features,
        features=x,
        labels=np.array([0, 1, 0, 1, 0, 1]),
        splits=np.array(["train", "train", "val", "val", "test", "test"]),
        report=np.array(json.dumps(report)),
    )
    output = tmp_path / "model.pt"
    result = train(features, output, epochs=2, device="cpu")
    assert result["metrics"]["test"]["total"] == 2
    checkpoint = torch.load(output, weights_only=True)
    model = SignModel(2)
    model.load_state_dict(checkpoint["model"])
    assert model(torch.from_numpy(x[:1])).shape == (1, 2)
    with pytest.raises(FileExistsError):
        train(features, output, epochs=1, device="cpu")


def test_low_score_is_reported_as_uncertain():
    torch = pytest.importorskip("torch")
    from sign_language.recognizer import prediction_message

    labels = ["hello", "sorry"]
    assert prediction_message(torch.tensor([0.45, 0.55]), labels, 0.6).startswith("Uncertain")
    assert prediction_message(torch.tensor([0.45, 0.55]), labels, 0.5).startswith("sorry")
    assert "experimental guess" in prediction_message(torch.tensor([0.0, 1.0]), labels, 0.5, "vote")
    with pytest.raises(ValueError):
        prediction_message(torch.tensor([0.45, 0.55]), labels, 2.0)


def test_exported_svm_votes_match_sklearn(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    from sign_language.recognizer import svm_vote_scores, train_svm

    rng = np.random.default_rng(7)
    x = rng.normal(size=(12, SEQUENCE_LENGTH, FEATURE_SIZE)).astype(np.float32)
    y = np.array([0, 1, 2] * 4)
    splits = np.array(["train"] * 6 + ["val"] * 3 + ["test"] * 3)
    report = {
        "feature_schema": FEATURE_SCHEMA,
        "label_order": ["a", "b", "c"],
        "input_sha256": {"hand_model": "example"},
        "missing_glosses": {"train": [], "val": [], "test": []},
    }
    features, output = tmp_path / "features.npz", tmp_path / "svm.pt"
    np.savez_compressed(
        features, features=x, labels=y, splits=splits, report=np.array(json.dumps(report))
    )
    train_svm(features, output)
    checkpoint = torch.load(output, weights_only=True)
    scaler = StandardScaler().fit(x[:6].reshape(6, -1))
    classifier = SVC(kernel="linear", class_weight="balanced", random_state=42)
    scale = np.maximum(scaler.scale_, 0.001)
    classifier.fit((x[:6].reshape(6, -1) - scaler.mean_) / scale, y[:6])
    expected = classifier.predict((x.reshape(12, -1) - scaler.mean_) / scale)
    actual = svm_vote_scores(torch.from_numpy(x), checkpoint).argmax(dim=1).numpy()
    assert np.array_equal(actual, expected)


def test_world_summary_keeps_motion_bins_and_ignores_absolute_image_position():
    torch = pytest.importorskip("torch")
    from sign_language.recognizer import svm_inputs

    x = torch.zeros((1, 32, 288))
    x[:, :, 63] = x[:, :, 127] = x[:, :, 223] = x[:, :, 287] = 1
    x[0, :, 163] = torch.arange(32)
    encoded = svm_inputs(x, "world_summary")
    assert encoded.shape == (1, 670)
    assert encoded[0, [3, 137, 271, 405]].tolist() == [3.5, 11.5, 19.5, 27.5]
    shifted = x.clone()
    shifted[:, :, :63] += 0.2
    shifted[:, :, 64:127] += 0.2
    assert torch.equal(svm_inputs(shifted, "world_summary"), encoded)


def test_timing_augmentation_preserves_direction_and_example_blocks():
    pytest.importorskip("torch")
    from sign_language.recognizer import augment_timing

    x = np.zeros((2, 32, 288), dtype=np.float32)
    x[0, :, 0] = np.arange(32)
    x[1, :, 0] = 100 + np.arange(32)
    before = x.copy()
    result = augment_timing(x)
    assert result.shape == (10, 32, 288)
    assert np.array_equal(x, before)
    assert np.array_equal(result[:2], x)
    assert np.all(np.diff(result[:, :, 0], axis=1) >= 0)
    assert np.all(result[::2, :, 0] <= 31)
    assert np.all(result[1::2, :, 0] >= 100)
    assert result[2, 6, 0] == 0 and result[2, 7, 0] == 1
    assert result[4, -1, 0] == 31
    with pytest.raises(ValueError, match="32-frame world-hand"):
        augment_timing(x[:, :16])


def test_augmented_svm_fits_only_training_and_skip_test_never_scores_test(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from sklearn.preprocessing import StandardScaler

    import sign_language.recognizer as recognition
    from sign_language.landmarks import WORLD_FEATURE_SCHEMA

    rng = np.random.default_rng(13)
    x = rng.normal(size=(18, 32, 288)).astype(np.float32)
    x[12:] += 1000  # A held-out distribution must not change the fitted scaler.
    truth = np.array([0, 1, 2] * 6)
    splits = np.array(["train"] * 12 + ["val"] * 3 + ["test"] * 3)
    report = {
        "feature_schema": WORLD_FEATURE_SCHEMA,
        "label_order": ["a", "b", "c"],
        "input_sha256": {"hand_model": "test", "pose_model": "test"},
        "missing_glosses": {s: [] for s in ("train", "val", "test")},
    }
    features, output = tmp_path / "features.npz", tmp_path / "timing.pt"
    np.savez_compressed(
        features, features=x, labels=truth, splits=splits, report=np.array(json.dumps(report))
    )
    scored = []
    original = recognition.svm_vote_scores

    def record_scores(inputs, checkpoint):
        scored.append(inputs.numpy().copy())
        return original(inputs, checkpoint)

    monkeypatch.setattr(recognition, "svm_vote_scores", record_scores)
    result = recognition.train_svm(
        features,
        output,
        representation="world_summary",
        evaluate_test=False,
        temporal_augmentation=True,
    )
    checkpoint = torch.load(output, weights_only=True)
    flat = recognition.svm_inputs(
        torch.from_numpy(recognition.augment_timing(x[:12])), "world_summary"
    ).numpy()
    expected = StandardScaler().fit(flat)
    assert np.allclose(checkpoint["mean"].numpy(), expected.mean_, atol=1e-6)
    assert checkpoint["training_augmentation"] == "timing-endpoints-v1"
    assert checkpoint["training_examples_after_augmentation"] == 60
    assert result["metrics"]["train"]["total"] == 12
    assert "test" not in result["metrics"]
    assert len(scored) == 1 and np.array_equal(scored[0], x[:15])
    with pytest.raises(ValueError, match="world_summary"):
        recognition.train_svm(features, tmp_path / "invalid.pt", temporal_augmentation=True)


def test_exported_rbf_world_classifier_matches_sklearn_without_test_scoring(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    from sign_language.landmarks import WORLD_FEATURE_SCHEMA
    from sign_language.recognizer import svm_inputs, svm_vote_scores, train_svm

    rng = np.random.default_rng(8)
    x = rng.normal(size=(24, 32, 288)).astype(np.float32)
    truth = np.array([0, 1, 2, 3] * 6)
    splits = np.array(["train"] * 16 + ["val"] * 4 + ["test"] * 4)
    report = {
        "feature_schema": WORLD_FEATURE_SCHEMA,
        "label_order": ["one", "two", "three", "four"],
        "input_sha256": {"hand_model": "test", "pose_model": "test"},
        "missing_glosses": {s: [] for s in ("train", "val", "test")},
    }
    features = tmp_path / "world.npz"
    np.savez_compressed(
        features, features=x, labels=truth, splits=splits, report=np.array(json.dumps(report))
    )
    result = train_svm(
        features,
        tmp_path / "rbf.pt",
        representation="world_summary",
        kernel="rbf",
        evaluate_test=False,
    )
    assert "test" not in result["metrics"] and result["test_evaluated"] is False
    saved = torch.load(tmp_path / "rbf.pt", weights_only=True)
    flat = svm_inputs(torch.from_numpy(x), "world_summary").numpy()
    scaler = StandardScaler().fit(flat[:16])
    transformed = (flat - scaler.mean_) / np.maximum(scaler.scale_, 0.001)
    expected = (
        SVC(kernel="rbf", class_weight="balanced", random_state=42)
        .fit(transformed[:16], truth[:16])
        .predict(transformed)
    )
    actual = svm_vote_scores(torch.from_numpy(x), saved).argmax(1).numpy()
    assert np.array_equal(actual, expected)
