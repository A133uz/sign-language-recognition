"""Verify dataset splits preserve groups and expose missing coverage."""

import json

import pytest
import yaml

from sign_language.cli import main
from sign_language.splits import split_metadata, split_with_coverage


def example_signs():
    return [
        {
            "gloss": "hello",
            "instances": [
                {"signer_id": 1, "video_id": "a", "url": "https://example.com/shared"},
                {"signer_id": 2, "video_id": "b", "url": "https://example.com/shared"},
                {"signer_id": 3, "video_id": "c", "url": "https://example.com/c"},
                {"signer_id": 4, "video_id": "d", "url": "https://example.com/d"},
                {"signer_id": 5, "video_id": "e", "url": "https://example.com/e"},
            ],
        },
        {
            "gloss": "bye",
            "instances": [{"signer_id": 1, "video_id": "f", "url": "https://example.com/f"}],
        },
    ]


def test_split_is_reproducible_and_groups_shared_media():
    first = split_metadata(example_signs(), ("bye", "hello"), seed=42)
    assert first == split_metadata(example_signs(), ("bye", "hello"), seed=42)
    assert first["status"] == "provisional_metadata_only"
    assert first["label_order"] == ["bye", "hello"]
    assert sum(first["clip_counts"].values()) == 6
    assert all(record["availability"] == "not_checked" for record in first["records"])
    assignments = {record["video_id"]: record["assigned_split"] for record in first["records"]}
    assert assignments["a"] == assignments["b"] == assignments["f"]
    assert all(
        counts == {"signer_ids": 0, "exact_urls": 0, "video_ids": 0}
        for counts in first["overlap"].values()
    )
    assert first["records"][0]["class_id"] == 0
    assert sum("bye" in missing for missing in first["missing_glosses"].values()) == 2


def test_split_groups_matching_video_ids_even_with_different_urls():
    signs = example_signs()
    signs[0]["instances"][1]["url"] = "https://example.com/another-url"
    signs[0]["instances"][1]["video_id"] = "a"
    report = split_metadata(signs, ("hello", "bye"))
    records = report["records"]
    shared_video_splits = {
        record["assigned_split"] for record in records if record["video_id"] == "a"
    }
    assert len(shared_video_splits) == 1


def test_split_groups_duplicate_file_hashes():
    signs = example_signs()
    signs[0]["instances"][0]["content_sha256"] = "same-hash"
    signs[0]["instances"][2]["content_sha256"] = "same-hash"
    records = split_metadata(signs, ("hello",))["records"]
    assert (
        next(r for r in records if r["video_id"] == "a")["assigned_split"]
        == next(r for r in records if r["video_id"] == "c")["assigned_split"]
    )


def test_coverage_selection_is_reproducible_and_preserves_grouping():
    signs = example_signs()
    selected = split_with_coverage(signs, ("hello", "bye"), attempts=20)
    assert selected == split_with_coverage(signs, ("hello", "bye"), attempts=20)
    assert selected["missing_glosses"]["train"] == []
    assert selected["selection"]["attempts"] == 20
    assert all(
        all(value == 0 for value in overlap.values()) for overlap in selected["overlap"].values()
    )


@pytest.mark.parametrize("field", ["signer_id", "video_id", "url"])
def test_split_rejects_missing_required_identity(field):
    signs = example_signs()
    del signs[0]["instances"][0][field]
    with pytest.raises(ValueError, match="Missing"):
        split_metadata(signs, ("hello",))


def test_split_rejects_unknown_gloss():
    with pytest.raises(ValueError, match="not found"):
        split_metadata(example_signs(), ("unknown",))


def test_split_rejects_mixed_signer_id_types():
    signs = example_signs()
    signs[0]["instances"][1]["signer_id"] = "1"
    with pytest.raises(ValueError, match="Mixed signer ID types"):
        split_metadata(signs, ("hello",))


def test_cli_writes_new_manifest_and_will_not_overwrite(tmp_path, capsys):
    metadata = tmp_path / "metadata.json"
    vocabulary = tmp_path / "vocabulary.yaml"
    output = tmp_path / "manifest.json"
    metadata.write_text(json.dumps(example_signs()), encoding="utf-8")
    vocabulary.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "language": "asl", "dataset": "wlasl", "labels": ["hello", "bye"]}
        ),
        encoding="utf-8",
    )
    arguments = [
        "split-wlasl",
        "--metadata",
        str(metadata),
        "--vocabulary",
        str(vocabulary),
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert len(manifest["metadata_sha256"]) == 64
    assert len(manifest["vocabulary_sha256"]) == 64
    original = output.read_text(encoding="utf-8")
    capsys.readouterr()
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 1
    assert output.read_text(encoding="utf-8") == original
