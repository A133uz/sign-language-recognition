"""Prompted, in-memory webcam trials for diagnosing isolated-sign recognition."""

import math
import random
import time
from collections import Counter
from pathlib import Path

from sign_language.landmarks import CaptureRejected
from sign_language.video_recognizer import (
    VideoRecognizer,
    load_candidates,
    preview_crop_bounds,
    resize_frame,
    sample_live_frames,
)

DEFAULT_VARIANTS = ("aspect_mean", "aspect_max", "square_mean", "square_max")


def build_trial_plan(
    labels: list[str], signs: list[str] | None, repeats: int, seed: int
) -> list[str]:
    """Randomize repeated, explicit prompts without inspecting recognition results."""
    if not 1 <= repeats <= 10:
        raise ValueError("Trial repeats must be between 1 and 10")
    mapping = {label.casefold(): label for label in labels}
    selected = labels if signs is None else [mapping.get(sign.casefold(), "") for sign in signs]
    if not selected or any(not label for label in selected):
        raise ValueError("Every requested sign must be in the selected vocabulary")
    if len(set(selected)) != len(selected):
        raise ValueError("Trial signs must be unique")
    plan = list(selected) * repeats
    random.Random(seed).shuffle(plan)
    return plan


def summarize_trials(
    plan: list[str], records: list[dict], variants: tuple[str, ...] | None = None
) -> dict:
    """Measure agreement with prompts, which are not independently reviewed ground truth."""
    per_variant = {}
    if variants is None:
        variants = tuple(records[0]["predictions"]) if records else DEFAULT_VARIANTS
    for variant in variants:
        per_sign = {}
        for sign in dict.fromkeys(plan):
            subset = [record for record in records if record["prompt"] == sign]
            evaluated = [record for record in subset if variant in record["predictions"]]
            per_sign[sign] = {
                "planned": plan.count(sign),
                "completed": len(subset),
                "predicted": len(evaluated),
                "top1_prompt_matches": sum(
                    record["predictions"][variant][0]["gloss"] == sign for record in evaluated
                ),
                "top3_prompt_matches": sum(
                    sign in [guess["gloss"] for guess in record["predictions"][variant]]
                    for record in evaluated
                ),
            }
        per_variant[variant] = {
            "completed": len(records),
            "predicted": sum(item["predicted"] for item in per_sign.values()),
            "top1_prompt_matches": sum(item["top1_prompt_matches"] for item in per_sign.values()),
            "top3_prompt_matches": sum(item["top3_prompt_matches"] for item in per_sign.values()),
            "per_sign": per_sign,
        }
    return per_variant


def run_guided_webcam(
    weights: Path,
    classes: Path | None,
    vocabulary: Path | None,
    *,
    signs: list[str] | None = None,
    repeats: int = 3,
    seed: int = 42,
    seconds: float = 2.0,
    camera: int = 0,
    device: str = "auto",
    backend: str = "wlasl",
    quality_models: tuple[Path, Path] | None = None,
    alignment_reference: Path | None = None,
    comparison_weights: Path | None = None,
) -> dict:
    """Collect predictions only; raw frames remain in memory and are never saved."""
    import cv2

    if not math.isfinite(seconds) or not 0.5 <= seconds <= 5:
        raise ValueError("Trial capture duration must be between 0.5 and 5 seconds")
    reference = None
    if comparison_weights and backend != "landmarks":
        raise ValueError("Checkpoint comparison requires the landmarks backend")
    if alignment_reference:
        from sign_language.body_alignment import load_reference

        if backend != "wlasl" or not quality_models:
            raise ValueError("Body alignment requires the WLASL backend and hand/pose models")
        reference = load_reference(alignment_reference)
    if backend == "landmarks":
        from sign_language.landmark_recognizer import LandmarkComparison, LandmarkRecognizer

        if quality_models:
            raise ValueError("The pixel-crop quality option is specific to video backends")
        recognizer = (
            LandmarkComparison(weights, comparison_weights, device)
            if comparison_weights
            else LandmarkRecognizer(weights, device)
        )
    elif backend == "citizen":
        from sign_language.citizen_recognizer import CitizenVideoRecognizer

        recognizer = CitizenVideoRecognizer(weights, classes, device)
    elif backend == "wlasl":
        recognizer = VideoRecognizer(weights, classes, device)
    else:
        raise ValueError("Backend must be wlasl, citizen or landmarks")
    candidates = (
        load_candidates(
            vocabulary, recognizer.labels, dataset=getattr(recognizer, "dataset", "wlasl")
        )
        or recognizer.labels
    )
    plan = build_trial_plan(candidates, signs, repeats, seed)
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open webcam {camera}")
    title = {
        "citizen": "ASL Citizen guided webcam check",
        "wlasl": "WLASL guided webcam check",
        "landmarks": "Landmark guided webcam check",
    }[backend]
    live_limit = 600 if backend != "wlasl" else 64
    records, frames, timestamps = [], [], []
    stage, deadline, started = "ready", 0.0, 0.0
    status, detail = "cancelled", None
    messages = []
    retained_bytes = 0
    quality_extractor = None
    aligned_preview = None
    try:
        if quality_models:
            from sign_language.landmarks import HandExtractor

            quality_extractor = HandExtractor(*quality_models)
        if backend != "wlasl" or quality_models:
            # Keep native pixels for the source preprocessing while bounding capture memory.
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                status, detail = "camera_failed", "Webcam stopped returning frames"
                break
            now = time.monotonic()
            if stage == "countdown" and now >= deadline:
                stage, started = "capture", now
                frames, timestamps = [], []
                retained_bytes = 0
                aligned_preview = None
            if stage == "capture":
                # Limit retained image memory; neither pixels nor landmarks go into the report.
                retained = (
                    frame.copy() if backend != "wlasl" or quality_models else resize_frame(frame)
                )
                retained_bytes += retained.nbytes
                if retained_bytes > 256 * 1024 * 1024:
                    raise RuntimeError(
                        "Capture exceeds 256 MiB; reduce camera resolution or duration"
                    )
                frames.append(retained)
                timestamps.append(now)
                if now - started >= seconds:
                    if len(frames) < 8:
                        stage = "ready"
                        messages = ["Too few camera frames. SPACE: retry this prompt"]
                    else:
                        display = cv2.flip(frame, 1)
                        cv2.putText(
                            display,
                            "Analyzing...",
                            (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                        cv2.imshow(title, display)
                        cv2.waitKey(1)
                        began = time.perf_counter()
                        rejection = None
                        try:
                            predictions = recognizer.predict_comparison(
                                sample_live_frames(frames, timestamps, limit=live_limit),
                                candidates,
                                top_k=min(3, len(candidates)),
                            )
                        except CaptureRejected as exc:
                            predictions, rejection = {}, str(exc)
                        record = {
                            "trial": len(records) + 1,
                            "prompt": plan[len(records)],
                            "camera_frames": len(frames),
                            "camera_frame_shape": list(frame.shape),
                            "capture_span_seconds": timestamps[-1] - timestamps[0],
                            "comparison_latency_ms": (time.perf_counter() - began) * 1000,
                            "predictions": predictions,
                        }
                        if rejection:
                            record["recognition_rejected"] = rejection
                        if backend == "landmarks":
                            record["tracking"] = recognizer.capture_metadata
                        if quality_extractor:
                            from sign_language.view_quality import (
                                extract_capture_features,
                                summarize_model_view,
                            )

                            quality_started = time.perf_counter()
                            try:
                                capture_features = extract_capture_features(
                                    frames, quality_extractor
                                )
                                record["view_quality"] = summarize_model_view(
                                    capture_features,
                                    frame.shape[1],
                                    frame.shape[0],
                                    backend=backend,
                                )
                                if reference:
                                    from sign_language.body_alignment import (
                                        align_frames,
                                        estimate_alignment,
                                    )

                                    matrix, alignment = estimate_alignment(
                                        capture_features, frame.shape[1], frame.shape[0], reference
                                    )
                                    aligned, padding = align_frames(frames, matrix)
                                    aligned_started = time.perf_counter()
                                    predictions["body_aligned"] = recognizer.predict(
                                        sample_live_frames(aligned, timestamps),
                                        top_k=min(3, len(candidates)),
                                        candidates=candidates,
                                    )
                                    record["alignment"] = dict(alignment, padding_fraction=padding)
                                    record["alignment_inference_latency_ms"] = (
                                        time.perf_counter() - aligned_started
                                    ) * 1000
                                    aligned_preview = recognizer.frame_crop(aligned[-1]).copy()
                            except (OSError, ValueError, RuntimeError) as exc:
                                key = (
                                    "alignment_error"
                                    if reference and "view_quality" in record
                                    else "view_quality_error"
                                )
                                record[key] = str(exc)
                            record["view_quality_latency_ms"] = (
                                time.perf_counter() - quality_started
                            ) * 1000
                        records.append(record)
                        stage = "result"
                        messages = (
                            [f"Performed prompt: {records[-1]['prompt']}"]
                            + [
                                f"{variant}: {', '.join(guess['gloss'] for guess in guesses)}"
                                for variant, guesses in predictions.items()
                            ]
                            + ["SPACE: next trial | Q: finish and save partial report"]
                        )
                        if "alignment_error" in record:
                            messages.insert(1, "Body alignment unavailable; baseline guesses shown")
                        if rejection:
                            messages.insert(1, "Insufficient hand tracking; no guess")
                    frames, timestamps = [], []
            display = cv2.flip(frame, 1)
            height, width = display.shape[:2]
            crop_bounds = getattr(recognizer, "crop_bounds", preview_crop_bounds)
            left, top, right, bottom = crop_bounds(width, height)
            cv2.rectangle(display, (left, top), (right, bottom), (0, 190, 190), 1)
            if stage == "ready":
                lines = [
                    f"Trial {len(records) + 1}/{len(plan)}: {plan[len(records)]}",
                    "SPACE: get ready | Q: quit",
                    *messages,
                ]
            elif stage == "countdown":
                lines = [f"Get ready: {plan[len(records)]}", str(math.ceil(deadline - now))]
            elif stage == "capture":
                lines = [
                    f"SIGN NOW: {plan[len(records)]}",
                    f"{max(0, seconds - (now - started)):.1f}s",
                ]
            else:
                lines = messages
            lines = ["Closed-set diagnostic - no video saved", *lines]
            for row, line in enumerate(lines):
                cv2.putText(
                    display,
                    line,
                    (10, 30 + row * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            if quality_extractor:
                side = min(224, max(32, height - 70))
                display = cv2.copyMakeBorder(display, 0, 0, 0, side + 20, cv2.BORDER_CONSTANT)
                model_view = cv2.flip(recognizer.frame_crop(frame), 1)
                preview_label = "Model crop"
                if aligned_preview is not None and stage == "result":
                    model_view = cv2.flip(aligned_preview, 1)
                    preview_label = "Aligned capture"
                display[50 : 50 + side, width + 10 : width + 10 + side] = cv2.resize(
                    model_view, (side, side)
                )
                cv2.putText(
                    display,
                    preview_label,
                    (width + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow(title, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                status = "complete" if len(records) == len(plan) else "cancelled"
                break
            if key == ord(" ") and stage in ("ready", "result"):
                if len(records) == len(plan):
                    status = "complete"
                    break
                stage, deadline = "countdown", time.monotonic() + 2.0
                messages = []
    except RuntimeError as exc:
        status, detail = "runtime_failed", str(exc)
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if quality_extractor:
            quality_extractor.close()
    return {
        "status": status,
        "detail": detail,
        "evidence": "prompt_agreement_not_independently_reviewed",
        "limitations": [
            "Prompts describe intended signs; the performed signs are not independently reviewed.",
            "All variants choose from the configured vocabulary and cannot reject unknown signs.",
            "Same-capture comparisons are diagnostic; selecting a variant requires fresh trials.",
        ],
        "media_saved": False,
        "view_quality_enabled": bool(quality_models),
        "body_alignment_enabled": bool(reference),
        "checkpoint_comparison_enabled": bool(comparison_weights),
        "device": recognizer.device,
        "backend": backend,
        "camera": camera,
        "seconds": seconds,
        "countdown_seconds": 2.0,
        "preprocessing": {
            "retained_frames": (
                "native BGR, camera requested at 640x480, buffer limited to 256 MiB"
                if backend != "wlasl" or quality_models
                else "BGR, aspect-preserving 256-pixel short side"
            ),
            "sampling": (
                f"nearest timestamp to at most {live_limit} uniform points; repeat to at least 16"
            ),
            "comparison": getattr(recognizer, "inference_description", "test double"),
            "scores": getattr(
                recognizer,
                "score_description",
                "uncalibrated and conditional on configured vocabulary",
            ),
        },
        "seed": seed,
        "vocabulary": candidates,
        "plan": plan,
        "planned_counts": dict(Counter(plan)),
        "summary": summarize_trials(
            plan,
            records,
            (
                (*getattr(recognizer, "comparison_variants", DEFAULT_VARIANTS), "body_aligned")
                if reference
                else getattr(recognizer, "comparison_variants", None)
            ),
        ),
        "records": records,
    }
