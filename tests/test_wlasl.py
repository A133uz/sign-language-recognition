"""Exercise metadata reading, lookup and meaningful data errors."""

import json

import pytest

from sign_language.cli import main
from sign_language.wlasl import audit_metadata, find_sign, load_metadata, summarize_sign


def test_read_find_and_summarize_gloss(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            [
                {
                    "gloss": "hello",
                    "instances": [
                        {"signer_id": 3},
                        {"signer_id": 3},
                        {"signer_id": 7},
                        {"signer_id": None},
                        {},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    signs = load_metadata(path)
    sign = find_sign(signs, " HELLO ")
    assert sign is not None
    assert summarize_sign(sign) == {"gloss": "hello", "clips": 5, "signers": 2}
    assert find_sign(signs, "unknown") is None


@pytest.mark.parametrize(
    "bad_data",
    [{}, ["hello"], [{"gloss": "hello"}], [{"gloss": "hi", "instances": [None]}]],
)
def test_rejects_invalid_metadata_shape(tmp_path, bad_data):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(bad_data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_metadata(path)


def test_missing_gloss_exits_cleanly(tmp_path, capsys):
    path = tmp_path / "metadata.json"
    path.write_text('[{"gloss": "hello", "instances": []}]', encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["inspect-wlasl", "missing", "--metadata", str(path)])
    assert error.value.code == 1
    assert "WLASL gloss not found: missing" in capsys.readouterr().err


def test_audit_counts_overlapping_signers_and_missing_fields():
    signs = [
        {
            "gloss": "hello",
            "instances": [
                {"split": "train", "signer_id": 3, "url": "https://example.com/a"},
                {"split": "test", "signer_id": 3, "url": "https://example.com/b"},
                {"split": "test", "signer_id": None},
            ],
        },
        {"gloss": "bye", "instances": [{"split": "train", "signer_id": 4, "url": " "}]},
    ]
    report = audit_metadata(signs)
    assert report["gloss_count"] == 2
    assert report["listed_clips"] == 4
    assert report["known_signer_ids"] == 2
    assert report["clips_without_signer_id"] == 1
    assert report["clips_without_url"] == 2
    assert report["clips_by_source_split"] == {"test": 2, "train": 2}
    assert report["signer_id_overlap_by_split"] == {"test / train": 1}
    assert report["glosses"][0]["signers"] == 1


def test_audit_writes_detailed_report_and_protects_input(tmp_path, capsys):
    path = tmp_path / "metadata.json"
    path.write_text('[{"gloss": "hello", "instances": []}]', encoding="utf-8")
    output = tmp_path / "report.json"
    assert main(["audit-wlasl", "--metadata", str(path), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["glosses"][0]["gloss"] == "hello"
    assert len(report["metadata_sha256"]) == 64
    assert "glosses" not in json.loads(capsys.readouterr().out)

    with pytest.raises(SystemExit) as error:
        main(["audit-wlasl", "--metadata", str(path), "--output", str(path)])
    assert error.value.code == 1
    assert "must not overwrite" in capsys.readouterr().err
    assert json.loads(path.read_text(encoding="utf-8"))[0]["gloss"] == "hello"
