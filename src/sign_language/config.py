"""Validate the vocabulary before preparing datasets or assigning class IDs."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Vocabulary:
    """Label ordering is the future model's class-ID ordering."""

    language: str
    dataset: str
    labels: tuple[str, ...]


def load_vocabulary(path: Path) -> Vocabulary:
    """Read a versioned YAML vocabulary, rejecting ambiguous class mappings."""
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("Vocabulary must be a YAML mapping.")
    expected = {"schema_version", "language", "dataset", "labels"}
    if set(value) != expected:
        raise ValueError(f"Vocabulary must contain exactly these fields: {sorted(expected)}")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported vocabulary schema_version; expected integer 1.")
    for field in ("language", "dataset"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"{field} must be a nonempty string.")
    labels = value["labels"]
    if not isinstance(labels, list) or not labels:
        raise ValueError("labels must be a nonempty list.")
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("Each label must be a nonempty string.")
    if any(label != label.strip() for label in labels):
        raise ValueError("Labels must not have leading or trailing whitespace.")
    if len({label.casefold() for label in labels}) != len(labels):
        raise ValueError("Labels must be unique, ignoring case.")
    return Vocabulary(value["language"], value["dataset"], tuple(labels))
