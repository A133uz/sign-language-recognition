"""Protect dataset class IDs against invalid or ambiguous vocabulary input."""

from pathlib import Path

import pytest
import yaml

from sign_language.config import load_vocabulary


def write_config(tmp_path: Path, labels: object) -> Path:
    path = tmp_path / "vocabulary.yaml"
    path.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "language": "asl", "dataset": "wlasl", "labels": labels}
        ),
        encoding="utf-8",
    )
    return path


def test_preserves_class_order_and_multiword_labels(tmp_path):
    vocabulary = load_vocabulary(write_config(tmp_path, ["thank you", "hello", "no"]))
    assert vocabulary.labels == ("thank you", "hello", "no")


@pytest.mark.parametrize("labels", [[], ["hello", "HELLO"], [" hello"], [True], "hello"])
def test_rejects_ambiguous_or_invalid_labels(tmp_path, labels):
    with pytest.raises(ValueError):
        load_vocabulary(write_config(tmp_path, labels))


def test_yaml_boolean_labels_require_quotes(tmp_path):
    path = tmp_path / "vocabulary.yaml"
    path.write_text(
        "schema_version: 1\nlanguage: asl\ndataset: wlasl\nlabels: [yes, no]\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="nonempty string"):
        load_vocabulary(path)


def test_rejects_unknown_fields(tmp_path):
    path = write_config(tmp_path, ["hello"])
    path.write_text(path.read_text(encoding="utf-8") + "labells: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        load_vocabulary(path)
