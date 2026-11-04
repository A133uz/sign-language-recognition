"""Validate native label identity and source signer/media boundaries."""

import csv

import numpy as np
import pytest

from sign_language.citizen_data import load_citizen_records, prepare_citizen


def write_splits(root, rows):
    for split, row in zip(("train", "val", "test"), rows, strict=True):
        with (root / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Participant ID", "Video file", "Gloss", "ASL-LEX Code"])
            writer.writerow(row)


def test_source_splits_keep_native_labels_and_reject_overlap(tmp_path):
    rows = [[f"P{i}", f"{i}-HELLO.mp4", "HELLO", "D_02_055"] for i in (1, 2, 3)]
    write_splits(tmp_path, rows)
    result = load_citizen_records(tmp_path, ("HELLO",))
    assert [row["assigned_split"] for row in result] == ["train", "val", "test"]
    assert all(row["class_id"] == 0 for row in result)
    rows[1][0] = "P1"
    write_splits(tmp_path, rows)
    with pytest.raises(ValueError, match="Signer overlap"):
        load_citizen_records(tmp_path, ("HELLO",))


def test_source_splits_reject_duplicate_media_and_unsafe_filenames(tmp_path):
    rows = [[f"P{i}", f"{i}-ME.mp4", "ME", "B_01_068"] for i in (1, 2, 3)]
    rows[1][1] = rows[0][1]
    write_splits(tmp_path, rows)
    with pytest.raises(ValueError, match="Duplicate"):
        load_citizen_records(tmp_path, ("ME",))
    rows[1][1] = "../outside.mp4"
    write_splits(tmp_path, rows)
    with pytest.raises(ValueError, match="Unsafe"):
        load_citizen_records(tmp_path, ("ME",))


def test_preparation_preserves_source_memberships_and_rejects_identical_cross_split_pixels(
    tmp_path, monkeypatch
):
    import sign_language.citizen_data as module

    rows = [[f"P{i}", f"{i}-ME.mp4", "ME", "B_01_068"] for i in (1, 2, 3)]
    write_splits(tmp_path, rows)
    for i in (1, 2, 3):
        (tmp_path / f"{i}-ME.mp4").write_bytes(str(i).encode())
    vocabulary = tmp_path / "vocab.yaml"
    vocabulary.write_text(
        "schema_version: 1\nlanguage: asl\ndataset: asl-citizen\nlabels: [ME]\n", encoding="utf-8"
    )
    model = tmp_path / "model.task"
    model.write_bytes(b"model")

    class Extractor:
        def __init__(self, *args, include_world):
            assert include_world

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract(self, frame):
            encoded = np.zeros(288, dtype=np.float32)
            encoded[63] = encoded[223] = 1
            return encoded

    monkeypatch.setattr(module, "HandExtractor", Extractor)
    monkeypatch.setattr(module, "read_video", lambda path: [np.zeros((2, 2, 3))] * 5)
    output = tmp_path / "features.npz"
    result = prepare_citizen(tmp_path, vocabulary, tmp_path, model, model, output)
    assert result["clip_counts"] == {"train": 1, "val": 1, "test": 1}
    with np.load(output, allow_pickle=False) as data:
        assert data["features"].shape == (3, 32, 288)
        assert data["splits"].tolist() == ["train", "val", "test"]
    (tmp_path / "2-ME.mp4").write_bytes(b"1")
    with pytest.raises(RuntimeError, match="Identical video"):
        prepare_citizen(tmp_path, vocabulary, tmp_path, model, model, tmp_path / "leaky.npz")
    assert not (tmp_path / "leaky.npz").exists()
