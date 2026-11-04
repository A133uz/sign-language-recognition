"""Verify prompted-trial accounting and camera cleanup without accessing a real camera."""

from pathlib import Path

import numpy as np
import pytest

from sign_language.webcam_diagnostics import build_trial_plan, summarize_trials


def test_trial_plan_is_seeded_complete_and_rejects_untrained_prompts():
    labels = ["hello", "thank you", "no"]
    plan = build_trial_plan(labels, ["NO", "hello"], 3, 42)
    assert plan.count("no") == plan.count("hello") == 3
    assert plan == build_trial_plan(labels, ["no", "hello"], 3, 42)
    with pytest.raises(ValueError, match="vocabulary"):
        build_trial_plan(labels, ["me"], 3, 42)
    with pytest.raises(ValueError, match="unique"):
        build_trial_plan(labels, ["no", "NO"], 3, 42)
    with pytest.raises(ValueError, match="repeats"):
        build_trial_plan(labels, None, 0, 42)


def test_prompt_agreement_keeps_wrong_top1_and_missing_trials_visible():
    guesses = [{"gloss": "hello", "score": 0.6}, {"gloss": "no", "score": 0.4}]
    records = [
        {
            "prompt": "no",
            "predictions": {
                variant: guesses
                for variant in ("aspect_mean", "aspect_max", "square_mean", "square_max")
            },
        }
    ]
    summary = summarize_trials(["no", "hello", "no"], records)
    assert summary["aspect_mean"]["top1_prompt_matches"] == 0
    assert summary["aspect_mean"]["top3_prompt_matches"] == 1
    assert summary["aspect_mean"]["per_sign"]["no"] == {
        "planned": 2,
        "completed": 1,
        "predicted": 1,
        "top1_prompt_matches": 0,
        "top3_prompt_matches": 1,
    }
    assert summary["aspect_mean"]["per_sign"]["hello"]["completed"] == 0


@pytest.mark.parametrize("camera_fails", [False, True])
@pytest.mark.parametrize("backend", ["wlasl", "citizen"])
@pytest.mark.parametrize("quality_enabled", [False, True])
def test_guided_capture_uses_countdown_and_saves_predictions_only(
    monkeypatch, camera_fails, backend, quality_enabled
):
    cv2 = pytest.importorskip("cv2")
    import sign_language.webcam_diagnostics as module

    class Camera:
        released = False
        reads = 0

        def isOpened(self):
            return True

        def set(self, prop, value):
            return True

        def read(self):
            self.reads += 1
            if camera_fails and self.reads > 65:
                return False, None
            return True, np.zeros((256, 256, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    class Recognizer:
        labels = ["hello"]
        device = "cpu"
        dataset = "asl-citizen" if backend == "citizen" else "wlasl"
        comparison_variants = ("citizen",) if backend == "citizen" else module.DEFAULT_VARIANTS

        def __init__(self, *args):
            pass

        @staticmethod
        def frame_crop(frame):
            return cv2.resize(frame, (224, 224))

        def predict_comparison(self, frames, candidates, top_k):
            assert len(frames) >= 16
            return {
                variant: [{"gloss": "hello", "score": 1.0}] for variant in self.comparison_variants
            }

    class QualityExtractor:
        closed = False

        def __init__(self, *args):
            pass

        def extract(self, frame):
            result = np.zeros(160, dtype=np.float32)
            result[:63] = np.tile([0.5, 0.5, 0], 21)
            result[63] = 1
            return result

        def close(self):
            QualityExtractor.closed = True

    clock = iter(np.arange(0, 20, 0.05))
    ticks = [0]

    def key(delay):
        ticks[0] += 1
        return ord(" ") if ticks[0] == 1 else ord("q") if ticks[0] > 66 else -1

    camera = Camera()
    closed = []
    monkeypatch.setattr(module, "VideoRecognizer", Recognizer)
    monkeypatch.setattr("sign_language.citizen_recognizer.CitizenVideoRecognizer", Recognizer)
    monkeypatch.setattr("sign_language.landmarks.HandExtractor", QualityExtractor)
    monkeypatch.setattr(module, "load_candidates", lambda *args, **kwargs: ["hello"])
    monkeypatch.setattr(module.time, "monotonic", lambda: float(next(clock)))
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: camera)
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", key)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: closed.append(True))
    report = module.run_guided_webcam(
        Path("w"),
        Path("c"),
        Path("v"),
        repeats=1,
        seconds=0.5,
        backend=backend,
        quality_models=(Path("hand"), Path("pose")) if quality_enabled else None,
    )
    assert report["status"] == ("camera_failed" if camera_fails else "complete")
    assert report["media_saved"] is False
    assert report["records"][0]["camera_frames"] >= 8
    variant = "citizen" if backend == "citizen" else "aspect_mean"
    assert report["records"][0]["predictions"][variant][0]["gloss"] == "hello"
    assert report["backend"] == backend
    assert report["view_quality_enabled"] == quality_enabled
    if quality_enabled:
        assert QualityExtractor.closed
        assert report["records"][0]["view_quality"]["frames_with_hands"] == 32
        assert "features" not in report["records"][0]["view_quality"]
    assert camera.released and closed


def test_guided_cli_writes_partial_report_and_refuses_overwrite(tmp_path, monkeypatch, capsys):
    import json

    from sign_language.cli import main

    for name in ("weights.pt", "classes.txt", "vocabulary.yaml", "uv.lock"):
        (tmp_path / name).write_text("fixture", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "cancelled", "records": [], "plan": ["yes"], "media_saved": False}

    monkeypatch.setattr("sign_language.webcam_diagnostics.run_guided_webcam", run)
    args = [
        "test-webcam-i3d",
        "--weights",
        "weights.pt",
        "--classes",
        "classes.txt",
        "--vocabulary",
        "vocabulary.yaml",
        "--sign",
        "yes",
        "--output",
        "report.json",
    ]
    assert main(args) == 0
    output = tmp_path / "report.json"
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "cancelled" and report["media_saved"] is False
    assert len(report["input_sha256"]["lockfile"]) == 64
    assert calls[0]["signs"] == ["yes"]
    saved = output.read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    assert len(calls) == 1 and output.read_bytes() == saved
    capsys.readouterr()


def test_non_wlasl_variant_summary_retains_uncompleted_prompts():
    records = [{"prompt": "ME", "predictions": {"citizen": [{"gloss": "ME", "score": 0.9}]}}]
    summary = summarize_trials(["ME", "NO"], records, ("citizen",))
    assert set(summary) == {"citizen"}
    assert summary["citizen"]["top1_prompt_matches"] == 1
    assert summary["citizen"]["per_sign"]["NO"]["completed"] == 0
    empty = summarize_trials(["ME"], [], ("citizen",))
    assert empty["citizen"]["completed"] == 0


def test_missing_alignment_prediction_remains_visible_in_denominators():
    records = [{"prompt": "no", "predictions": {"aspect_mean": [{"gloss": "no"}]}}]
    summary = summarize_trials(["no"], records, ("aspect_mean", "body_aligned"))
    assert summary["aspect_mean"]["predicted"] == 1
    assert summary["body_aligned"]["completed"] == 1
    assert summary["body_aligned"]["predicted"] == 0
    assert summary["body_aligned"]["top1_prompt_matches"] == 0


def test_landmark_cli_needs_no_external_class_list_and_preserves_provenance(
    tmp_path, monkeypatch, capsys
):
    from sign_language.cli import main

    monkeypatch.chdir(tmp_path)
    (tmp_path / "assets/models").mkdir(parents=True)
    for name in (
        "model.pt",
        "comparison.pt",
        "uv.lock",
        "assets/models/hand_landmarker.task",
        "assets/models/pose_landmarker_lite.task",
    ):
        (tmp_path / name).write_bytes(b"fixture")
    calls = []

    def run(weights, classes, vocabulary, **kwargs):
        assert classes is None and vocabulary is None
        calls.append(kwargs)
        return {"status": "cancelled", "records": [], "plan": ["YES"]}

    monkeypatch.setattr("sign_language.webcam_diagnostics.run_guided_webcam", run)
    assert (
        main(
            [
                "test-webcam",
                "--backend",
                "landmarks",
                "--weights",
                "model.pt",
                "--compare-weights",
                "comparison.pt",
                "--output",
                "out.json",
            ]
        )
        == 0
    )
    assert calls[0]["backend"] == "landmarks"
    assert calls[0]["comparison_weights"] == Path("comparison.pt")
    import json

    saved = json.loads((tmp_path / "out.json").read_text())
    assert "comparison_weights" in saved["input_sha256"]
    capsys.readouterr()
