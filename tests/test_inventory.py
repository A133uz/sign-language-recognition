"""Check local inventory reporting and safe output handling."""

import json

import pytest
import yaml

from sign_language.cli import main
from sign_language.inventory import inventory_videos


def signs():
    return [
        {
            "gloss": "hello",
            "instances": [
                {"video_id": "123", "signer_id": 1, "frame_start": 1, "frame_end": -1},
                {"video_id": "456", "signer_id": 2},
            ],
        }
    ]


def test_inventory_records_missing_and_decoded(tmp_path, monkeypatch):
    (tmp_path / "123.mp4").write_bytes(b"clip")
    monkeypatch.setattr(
        "sign_language.inventory.inspect_local_video",
        lambda path: {"status": "decoded", "sha256": "abc", "decoded_frames": 12},
    )
    report = inventory_videos(signs(), ("hello",), tmp_path)
    assert report["counts"] == {"decoded": 1, "missing": 1}
    assert report["records"][0]["frame_end"] == -1
    assert report["records"][1]["file"] == "456.mp4"


def test_inventory_rejects_unsafe_or_duplicate_ids(tmp_path):
    examples = signs()
    examples[0]["instances"][0]["video_id"] = "../clip"
    with pytest.raises(ValueError, match="Invalid video ID"):
        inventory_videos(examples, ("hello",), tmp_path)
    examples = signs()
    examples[0]["instances"][1]["video_id"] = "123"
    with pytest.raises(ValueError, match="Duplicate video ID"):
        inventory_videos(examples, ("hello",), tmp_path)


def test_cli_inventory_records_hashes_and_does_not_overwrite(tmp_path, capsys):
    metadata, vocab, output = (tmp_path / name for name in ("m.json", "v.yaml", "out.json"))
    metadata.write_text(json.dumps(signs()), encoding="utf-8")
    vocab.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "language": "asl", "dataset": "wlasl", "labels": ["hello"]}
        ),
        encoding="utf-8",
    )
    args = [
        "inventory-wlasl",
        "--metadata",
        str(metadata),
        "--vocabulary",
        str(vocab),
        "--videos",
        str(tmp_path),
        "--output",
        str(output),
    ]
    assert main(args) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["counts"] == {"missing": 2}
    assert len(report["metadata_sha256"]) == len(report["vocabulary_sha256"]) == 64
    original = output.read_bytes()
    capsys.readouterr()
    with pytest.raises(SystemExit):
        main(args)
    assert output.read_bytes() == original
