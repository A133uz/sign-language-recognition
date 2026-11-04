"""Validate reporting for present, missing, and failed WLASL examples."""

import pytest

from sign_language.evaluation import evaluate_wlasl_videos


class StubRecognizer:
    labels = ["hello", "no", "other"]

    def predict(self, frames, top_k, candidates):
        assert frames == ["frame"]
        return [{"gloss": "hello", "score": 0.8}, {"gloss": "no", "score": 0.2}]


def test_evaluation_counts_missing_and_per_class(tmp_path, monkeypatch):
    signs = [
        {
            "gloss": "hello",
            "instances": [
                {"video_id": "11", "signer_id": 1, "split": "test"},
                {"video_id": "12", "signer_id": 2, "split": "test"},
            ],
        },
        {
            "gloss": "no",
            "instances": [
                {"video_id": "13", "signer_id": 3, "split": "test"},
                {"video_id": "14", "signer_id": 4, "split": "train"},
            ],
        },
    ]
    (tmp_path / "11.mp4").write_bytes(b"clip 11")
    (tmp_path / "13.mp4").write_bytes(b"clip 13")
    monkeypatch.setattr("sign_language.evaluation.read_video_clip", lambda path: ["frame"])
    report = evaluate_wlasl_videos(
        signs, ("hello", "no"), tmp_path, StubRecognizer(), source_split="test", restricted=True
    )
    assert report["counts"] == {"predicted": 2, "missing": 1}
    assert report["top1_correct"] == 1
    assert report["top3_correct"] == 2
    assert report["per_gloss"][0] == {
        "gloss": "hello",
        "listed": 2,
        "predicted": 1,
        "top1_correct": 1,
        "top3_correct": 1,
    }
    assert len(report["records"][0]["sha256"]) == 64
    limited = evaluate_wlasl_videos(
        signs,
        ("hello", "no"),
        tmp_path,
        StubRecognizer(),
        source_split="test",
        restricted=True,
        limit=1,
    )
    assert limited["limit_reached"] and len(limited["records"]) == 1


def test_evaluation_rejects_unsafe_ids(tmp_path):
    signs = [{"gloss": "hello", "instances": [{"video_id": "../file", "split": "test"}]}]
    with pytest.raises(ValueError, match="Invalid video ID"):
        evaluate_wlasl_videos(
            signs, ("hello",), tmp_path, StubRecognizer(), source_split="test", restricted=False
        )


def test_decode_failure_is_reported_not_counted_as_wrong_prediction(tmp_path, monkeypatch):
    signs = [{"gloss": "hello", "instances": [{"video_id": "11", "split": "test"}]}]
    (tmp_path / "11.mp4").write_bytes(b"not a video")

    def fail_decode(path):
        raise ValueError("Video contains no decodable frames")

    monkeypatch.setattr("sign_language.evaluation.read_video_clip", fail_decode)
    report = evaluate_wlasl_videos(
        signs, ("hello",), tmp_path, StubRecognizer(), source_split="test", restricted=True
    )
    assert report["counts"] == {"decode_failed": 1}
    assert report["per_gloss"][0]["predicted"] == 0
    assert "top1_correct" not in report["records"][0]
