"""Setup diagnostics and metadata inspection commands."""

import argparse
import hashlib
import importlib
import json
import platform
from pathlib import Path

import yaml

from sign_language import __version__
from sign_language.config import load_vocabulary
from sign_language.inventory import inventory_videos
from sign_language.splits import split_metadata
from sign_language.video_probe import probe_gloss
from sign_language.wlasl import audit_metadata, find_sign, load_metadata, summarize_sign


def doctor(*, vision: bool, require_cuda: bool, require_torch: bool = False) -> tuple[dict, bool]:
    """Probe requested capabilities without opening the webcam or downloading files."""
    report: dict = {"python": platform.python_version(), "platform": platform.system()}
    errors = []
    modules = ["numpy", "yaml"]
    if vision:
        modules.extend(["cv2", "mediapipe"])
    if require_cuda or require_torch:
        modules.append("torch")
    for name in modules:
        try:
            module = importlib.import_module(name)
            report[name] = getattr(module, "__version__", "imported")
            if name == "mediapipe":
                from mediapipe.tasks.python import vision as mp_vision

                report["hand_landmarker_api"] = hasattr(mp_vision, "HandLandmarker")
                if not report["hand_landmarker_api"]:
                    errors.append("MediaPipe HandLandmarker API is unavailable.")
            if name == "torch":
                cpu_result = (module.ones(4) * 2).sum().item()
                report["cpu_calculation_passed"] = cpu_result == 8
                if cpu_result != 8:
                    errors.append("CPU calculation returned an unexpected result.")
                if not require_cuda:
                    continue
                available = module.cuda.is_available()
                report["cuda_available"] = available
                if not available:
                    errors.append("CUDA is required but unavailable to PyTorch.")
                else:
                    result = (module.ones(4, device="cuda") * 2).sum().item()
                    report["gpu"] = module.cuda.get_device_name(0)
                    report["torch_cuda_runtime"] = module.version.cuda
                    report["gpu_calculation_passed"] = result == 8
                    if result != 8:
                        errors.append("GPU calculation returned an unexpected result.")
        except Exception as exc:  # Diagnostics must also report native DLL/runtime failures.
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    report["errors"] = errors
    return report, not errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ASL project configuration and setup checks")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("check-config", help="Validate a vocabulary YAML file")
    config.add_argument("path", type=Path, help="Path to the vocabulary YAML file")
    diagnostics = commands.add_parser("doctor", help="Check installed runtime capabilities")
    diagnostics.add_argument("--vision", action="store_true", help="Require OpenCV and MediaPipe")
    diagnostics.add_argument("--torch", action="store_true", help="Run a PyTorch CPU calculation")
    diagnostics.add_argument("--require-cuda", action="store_true", help="Run a GPU calculation")
    inspect = commands.add_parser("inspect-wlasl", help="Inspect one gloss in WLASL metadata")
    inspect.add_argument("gloss", help="Gloss to look up, for example 'thank you'")
    inspect.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/raw/WLASL_v0.3.json"),
        help="Path to the WLASL JSON file (default: data/raw/WLASL_v0.3.json)",
    )
    audit = commands.add_parser("audit-wlasl", help="Summarize the full WLASL metadata")
    audit.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/raw/WLASL_v0.3.json"),
        help="Path to the WLASL JSON file (default: data/raw/WLASL_v0.3.json)",
    )
    audit.add_argument("--output", type=Path, help="Write the complete report as JSON to this file")
    probe = commands.add_parser("probe-wlasl", help="Check a few direct MP4 clips for one gloss")
    probe.add_argument("gloss", help="Gloss to inspect, for example 'hello'")
    probe.add_argument(
        "--metadata", type=Path, default=Path("data/raw/WLASL_v0.3.json"), help="WLASL JSON path"
    )
    probe.add_argument("--limit", type=int, default=3, help="Maximum direct links to test (1-10)")
    split = commands.add_parser("split-wlasl", help="Create a provisional signer-ID split")
    split.add_argument("--metadata", type=Path, default=Path("data/raw/WLASL_v0.3.json"))
    split.add_argument("--vocabulary", type=Path, default=Path("configs/vocabulary.yaml"))
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--output", type=Path, required=True, help="New output JSON path")
    inventory = commands.add_parser("inventory-wlasl", help="Inspect local WLASL MP4 files")
    inventory.add_argument("--metadata", type=Path, default=Path("data/raw/WLASL_v0.3.json"))
    inventory.add_argument("--vocabulary", type=Path, default=Path("configs/vocabulary.yaml"))
    inventory.add_argument("--videos", type=Path, default=Path("data/raw/videos"))
    inventory.add_argument("--output", type=Path, required=True, help="New output JSON path")
    prepare = commands.add_parser(
        "prepare-wlasl", help="Extract and split local WLASL clip landmarks"
    )
    prepare.add_argument("--metadata", type=Path, default=Path("data/raw/WLASL_v0.3.json"))
    prepare.add_argument("--vocabulary", type=Path, default=Path("configs/vocabulary.yaml"))
    prepare.add_argument("--videos", type=Path, default=Path("data/raw/videos"))
    prepare.add_argument("--model", type=Path, default=Path("assets/models/hand_landmarker.task"))
    prepare.add_argument("--pose-model", type=Path, help="Optional MediaPipe pose model")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--output", type=Path, required=True)
    citizen_prepare = commands.add_parser(
        "prepare-citizen", help="Extract world-hand features using native ASL Citizen splits"
    )
    citizen_prepare.add_argument("--splits", type=Path, default=Path("data/raw/aslcitizen/splits"))
    citizen_prepare.add_argument("--videos", type=Path, default=Path("data/raw/aslcitizen/videos"))
    citizen_prepare.add_argument(
        "--vocabulary", type=Path, default=Path("configs/vocabulary_aslcitizen.yaml")
    )
    citizen_prepare.add_argument(
        "--model", type=Path, default=Path("assets/models/hand_landmarker.task")
    )
    citizen_prepare.add_argument(
        "--pose-model", type=Path, default=Path("assets/models/pose_landmarker_lite.task")
    )
    citizen_prepare.add_argument("--output", type=Path, required=True)
    trainer = commands.add_parser(
        "train", help="Train an experimental landmark sequence classifier"
    )
    trainer.add_argument("features", type=Path)
    trainer.add_argument("--output", type=Path, required=True)
    trainer.add_argument("--epochs", type=int, default=100)
    trainer.add_argument("--seed", type=int, default=42)
    trainer.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    trainer.add_argument("--method", choices=("gru", "svm"), default="gru")
    trainer.add_argument("--svm-scale-floor", type=float, default=0.001)
    trainer.add_argument("--svm-kernel", choices=("linear", "rbf"), default="linear")
    trainer.add_argument(
        "--representation", choices=("raw", "world_summary", "body_geometry"), default="raw"
    )
    trainer.add_argument(
        "--temporal-augmentation",
        action="store_true",
        help="Vary training timing; requires SVM world_summary or body_geometry representation",
    )
    trainer.add_argument(
        "--skip-test", action="store_true", help="Do not score held-out test clips"
    )
    live = commands.add_parser("webcam", help="Recognize a captured isolated sign from a webcam")
    live.add_argument("checkpoint", type=Path)
    live.add_argument("--model", type=Path, default=Path("assets/models/hand_landmarker.task"))
    live.add_argument("--pose-model", type=Path, help="Pose model used by the checkpoint")
    live.add_argument("--camera", type=int, default=0)
    live.add_argument("--seconds", type=float, default=2.0)
    live.add_argument(
        "--min-score", type=float, default=0.5, help="Experimental uncertainty threshold (0-1)"
    )
    video_live = commands.add_parser(
        "webcam-i3d", help="Experimental pretrained WLASL2000 webcam model"
    )
    video_live.add_argument("--weights", type=Path, default=Path("assets/models/wlasl2000_i3d.pt"))
    video_live.add_argument(
        "--classes", type=Path, default=Path("assets/models/wlasl2000_classes.txt")
    )
    video_live.add_argument("--camera", type=int, default=0)
    video_live.add_argument("--seconds", type=float, default=2.0)
    video_live.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    video_live.add_argument(
        "--vocabulary", type=Path, help="Restrict guesses to these WLASL labels"
    )
    video_predict = commands.add_parser("predict-i3d", help="Predict signs in one saved video clip")
    video_predict.add_argument("video", type=Path)
    video_predict.add_argument(
        "--weights", type=Path, default=Path("assets/models/wlasl2000_i3d.pt")
    )
    video_predict.add_argument(
        "--classes", type=Path, default=Path("assets/models/wlasl2000_classes.txt")
    )
    video_predict.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    video_predict.add_argument("--vocabulary", type=Path, help="Restrict guesses to these labels")
    video_predict.add_argument("--top-k", type=int, default=3)
    video_evaluate = commands.add_parser(
        "evaluate-i3d", help="Explore pretrained predictions on local WLASL clips"
    )
    video_evaluate.add_argument("--metadata", type=Path, default=Path("data/raw/WLASL_v0.3.json"))
    video_evaluate.add_argument("--vocabulary", type=Path, default=Path("configs/vocabulary.yaml"))
    video_evaluate.add_argument("--videos", type=Path, default=Path("data/raw/videos"))
    video_evaluate.add_argument(
        "--weights", type=Path, default=Path("assets/models/wlasl2000_i3d.pt")
    )
    video_evaluate.add_argument(
        "--classes", type=Path, default=Path("assets/models/wlasl2000_classes.txt")
    )
    video_evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    video_evaluate.add_argument("--source-split", choices=("train", "val", "test"), default="test")
    video_evaluate.add_argument(
        "--restricted", action="store_true", help="Choose only from vocabulary labels"
    )
    video_evaluate.add_argument("--limit", type=int, help="Maximum listed records to inspect")
    video_evaluate.add_argument("--output", type=Path, required=True, help="New output JSON path")
    guided = commands.add_parser(
        "test-webcam",
        aliases=["test-webcam-i3d"],
        help="Guided webcam prompts with a predictions-only diagnostic report",
    )
    guided.add_argument("--backend", choices=("wlasl", "citizen", "landmarks"), default="wlasl")
    guided.add_argument("--weights", type=Path, help="Checkpoint; defaults depend on backend")
    guided.add_argument(
        "--compare-weights",
        type=Path,
        help="Compare a compatible landmark checkpoint on the same extracted capture",
    )
    guided.add_argument("--classes", type=Path, help="Class mapping; defaults depend on backend")
    guided.add_argument("--vocabulary", type=Path, help="Vocabulary; defaults depend on backend")
    guided.add_argument(
        "--sign", action="append", help="Prompt only this sign; repeat for more signs"
    )
    guided.add_argument("--repeats", type=int, default=3)
    guided.add_argument("--seed", type=int, default=42)
    guided.add_argument("--seconds", type=float, default=2.0)
    guided.add_argument("--camera", type=int, default=0)
    guided.add_argument(
        "--quality",
        action="store_true",
        help="Show model crop and save aggregate hand/pose geometry; no landmarks or video",
    )
    guided.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    guided.add_argument(
        "--align-body",
        type=Path,
        help="Compare experimental body alignment using this local framing reference (WLASL only)",
    )
    guided.add_argument("--output", type=Path, required=True, help="New predictions-only JSON path")
    framing = commands.add_parser(
        "build-framing-reference",
        help="Fit an experimental body-framing reference on training clips",
    )
    framing.add_argument("features", type=Path)
    framing.add_argument("--videos", type=Path, default=Path("data/raw/videos"))
    framing.add_argument("--output", type=Path, required=True)
    landmark_eval = commands.add_parser(
        "evaluate-landmarks", help="Score one frozen landmark checkpoint partition"
    )
    landmark_eval.add_argument("features", type=Path)
    landmark_eval.add_argument("checkpoint", type=Path)
    landmark_eval.add_argument("--split", choices=("train", "val", "test"), default="test")
    landmark_eval.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "evaluate-landmarks":
        try:
            from sign_language.landmark_recognizer import evaluate_landmark_checkpoint

            if args.output.exists():
                raise FileExistsError(f"Output already exists: {args.output}")
            report = evaluate_landmark_checkpoint(args.features, args.checkpoint, args.split)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            parser.exit(1, f"Cannot evaluate landmark checkpoint: {exc}\n")
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("records", "per_class")}, indent=2
            )
        )
        return 0
    if args.command == "build-framing-reference":
        try:
            from sign_language.body_alignment import build_reference

            reference = build_reference(args.features, args.videos, args.output)
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            parser.exit(1, f"Cannot build framing reference: {exc}\n")
        print(
            json.dumps(
                {
                    "training_clips": reference["training_clips"],
                    "nose_xy_px": reference["nose_xy_px"],
                    "shoulder_span_px": reference["shoulder_span_px"],
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0
    if args.command in ("test-webcam", "test-webcam-i3d"):
        try:
            from sign_language.webcam_diagnostics import run_guided_webcam

            if args.output.exists():
                raise FileExistsError(f"Output already exists: {args.output}")
            citizen = args.backend == "citizen"
            landmark_backend = args.backend == "landmarks"
            if args.compare_weights and not landmark_backend:
                raise ValueError("--compare-weights requires --backend landmarks")
            if args.backend != "wlasl" and args.align_body:
                raise ValueError("Body alignment is currently a WLASL-only experiment")
            if landmark_backend and (args.quality or args.classes or not args.weights):
                raise ValueError(
                    "Landmark backend needs --weights CHECKPOINT; omit --classes and --quality"
                )
            args.weights = args.weights or Path(
                "assets/models/aslcitizen_i3d.pt" if citizen else "assets/models/wlasl2000_i3d.pt"
            )
            args.classes = (
                None
                if landmark_backend
                else args.classes
                or Path(
                    "assets/models/aslcitizen_classes.json"
                    if citizen
                    else "assets/models/wlasl2000_classes.txt"
                )
            )
            args.vocabulary = (
                args.vocabulary
                if landmark_backend
                else args.vocabulary
                or Path(
                    "configs/vocabulary_aslcitizen.yaml" if citizen else "configs/vocabulary.yaml"
                )
            )
            inputs = {}
            input_paths = [
                ("weights", args.weights),
                ("classes", args.classes),
                ("vocabulary", args.vocabulary),
                ("lockfile", Path("uv.lock")),
                ("comparison_weights", args.compare_weights),
            ]
            quality_models = None
            if args.quality or args.align_body:
                quality_models = (
                    Path("assets/models/hand_landmarker.task"),
                    Path("assets/models/pose_landmarker_lite.task"),
                )
                input_paths.extend(zip(("hand_model", "pose_model"), quality_models, strict=True))
            if args.align_body:
                input_paths.append(("framing_reference", args.align_body))
            if landmark_backend:
                input_paths.extend(
                    [
                        ("hand_model", Path("assets/models/hand_landmarker.task")),
                        ("pose_model", Path("assets/models/pose_landmarker_lite.task")),
                    ]
                )
            for name, path in input_paths:
                if path is None:
                    continue
                with path.open("rb") as source:
                    inputs[name] = hashlib.file_digest(source, "sha256").hexdigest()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            report = run_guided_webcam(
                args.weights,
                args.classes,
                args.vocabulary,
                signs=args.sign,
                repeats=args.repeats,
                seed=args.seed,
                seconds=args.seconds,
                camera=args.camera,
                device=args.device,
                backend=args.backend,
                quality_models=quality_models,
                alignment_reference=args.align_body,
                comparison_weights=args.compare_weights,
            )
            report["input_sha256"] = inputs
            with args.output.open("x", encoding="utf-8") as output:
                json.dump(report, output, indent=2, allow_nan=False)
                output.write("\n")
        except (ImportError, OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
            parser.exit(1, f"Cannot run guided webcam check: {exc}\n")
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "completed": len(report["records"]),
                    "planned": len(report["plan"]),
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 1 if report["status"] in ("camera_failed", "runtime_failed") else 0
    if args.command == "evaluate-i3d":
        try:
            from sign_language.evaluation import evaluate_wlasl_videos
            from sign_language.video_recognizer import VideoRecognizer

            if args.output.exists():
                raise FileExistsError(f"Output already exists: {args.output}")
            vocabulary = load_vocabulary(args.vocabulary)
            if vocabulary.dataset.casefold() != "wlasl":
                raise ValueError("Vocabulary dataset must be WLASL")
            recognizer = VideoRecognizer(args.weights, args.classes, args.device)
            report = evaluate_wlasl_videos(
                load_metadata(args.metadata),
                vocabulary.labels,
                args.videos,
                recognizer,
                source_split=args.source_split,
                restricted=args.restricted,
                limit=args.limit,
            )
            report["device"] = recognizer.device
            report["input_sha256"] = {}
            for name, path in (
                ("metadata", args.metadata),
                ("vocabulary", args.vocabulary),
                ("weights", args.weights),
                ("classes", args.classes),
            ):
                with path.open("rb") as source:
                    report["input_sha256"][name] = hashlib.file_digest(source, "sha256").hexdigest()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                json.dump(report, output, indent=2, allow_nan=False)
                output.write("\n")
        except (ImportError, OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
            parser.exit(1, f"Cannot evaluate WLASL videos: {exc}\n")
        summary = {k: v for k, v in report.items() if k not in ("records", "per_gloss")}
        summary["output"] = str(args.output)
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "predict-i3d":
        try:
            from sign_language.video_recognizer import (
                VideoRecognizer,
                load_candidates,
                read_video_clip,
            )

            frames = read_video_clip(args.video)
            recognizer = VideoRecognizer(args.weights, args.classes, args.device)
            candidates = load_candidates(args.vocabulary, recognizer.labels)
            results = recognizer.predict(frames, args.top_k, candidates)
        except (ImportError, OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
            parser.exit(1, f"Cannot predict WLASL video: {exc}\n")
        print(
            json.dumps(
                {
                    "video": str(args.video),
                    "decoded_frames": len(frames),
                    "device": recognizer.device,
                    "vocabulary": str(args.vocabulary) if candidates else "WLASL2000",
                    "scores": "uncalibrated",
                    "predictions": results,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "webcam-i3d":
        try:
            from sign_language.video_recognizer import webcam_video

            webcam_video(
                args.weights, args.classes, args.camera, args.seconds, args.device, args.vocabulary
            )
        except (ImportError, OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
            parser.exit(1, f"Cannot run WLASL video recognizer: {exc}\n")
        return 0
    if args.command == "prepare-wlasl":
        from sign_language.prepare import prepare_wlasl

        try:
            report = prepare_wlasl(
                args.metadata,
                args.vocabulary,
                args.videos,
                args.model,
                args.output,
                args.seed,
                args.pose_model,
            )
        except (OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
            parser.exit(1, f"Cannot prepare WLASL clips: {exc}\n")
        print(
            json.dumps(
                {
                    "clips": report["clip_counts"],
                    "missing_glosses": report["missing_glosses"],
                    "skipped": report["skipped_counts"],
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "prepare-citizen":
        try:
            from sign_language.citizen_data import prepare_citizen

            report = prepare_citizen(
                args.splits, args.vocabulary, args.videos, args.model, args.pose_model, args.output
            )
        except (ImportError, OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
            parser.exit(1, f"Cannot prepare ASL Citizen: {exc}\n")
        print(
            json.dumps(
                {
                    "clips": report["clip_counts"],
                    "signers": report["signer_counts"],
                    "skipped": report["skipped_counts"],
                    "missing_glosses": report["missing_glosses"],
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "train":
        try:
            from sign_language.recognizer import train, train_svm

            if args.method == "svm":
                if args.device == "cuda":
                    raise ValueError(
                        "The SVM training method uses CPU; choose --device cpu or auto"
                    )
                report = train_svm(
                    args.features,
                    args.output,
                    args.seed,
                    args.svm_scale_floor,
                    representation=args.representation,
                    kernel=args.svm_kernel,
                    evaluate_test=not args.skip_test,
                    temporal_augmentation=args.temporal_augmentation,
                )
            else:
                if (
                    args.representation != "raw"
                    or args.svm_kernel != "linear"
                    or args.temporal_augmentation
                ):
                    raise ValueError("Representation and SVM kernel options require --method svm")
                report = train(
                    args.features,
                    args.output,
                    args.epochs,
                    args.seed,
                    args.device,
                    evaluate_test=not args.skip_test,
                )
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            parser.exit(1, f"Cannot train classifier: {exc}\n")
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "webcam":
        try:
            from sign_language.recognizer import webcam

            webcam(
                args.checkpoint,
                args.model,
                args.camera,
                args.seconds,
                args.min_score,
                args.pose_model,
            )
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            parser.exit(1, f"Cannot run webcam recognizer: {exc}\n")
        return 0
    if args.command == "inventory-wlasl":
        try:
            vocabulary = load_vocabulary(args.vocabulary)
            if vocabulary.dataset.casefold() != "wlasl":
                raise ValueError("Vocabulary dataset must be WLASL.")
            report = inventory_videos(load_metadata(args.metadata), vocabulary.labels, args.videos)
            for name, path in (("metadata", args.metadata), ("vocabulary", args.vocabulary)):
                with path.open("rb") as source:
                    report[f"{name}_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                json.dump(report, output, indent=2, allow_nan=False)
                output.write("\n")
        except (OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
            parser.exit(1, f"Cannot inventory WLASL videos: {exc}\n")
        print(json.dumps({"counts": report["counts"], "output": str(args.output)}, indent=2))
        return 0
    if args.command == "doctor":
        report, success = doctor(
            vision=args.vision, require_cuda=args.require_cuda, require_torch=args.torch
        )
        print(json.dumps(report, indent=2))
        return 0 if success else 1
    if args.command == "inspect-wlasl":
        try:
            signs = load_metadata(args.metadata)
        except (OSError, ValueError) as exc:
            parser.exit(1, f"Cannot read WLASL metadata: {exc}\n")
        sign = find_sign(signs, args.gloss)
        if sign is None:
            parser.exit(1, f"WLASL gloss not found: {args.gloss}\n")
        print(json.dumps(summarize_sign(sign), indent=2))
        return 0
    if args.command == "audit-wlasl":
        try:
            if args.output and args.output.resolve() == args.metadata.resolve():
                raise ValueError("Report output must not overwrite the metadata file.")
            signs = load_metadata(args.metadata)
            with args.metadata.open("rb") as source:
                source_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
            report = {"metadata_sha256": source_sha256, **audit_metadata(signs)}
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except (OSError, ValueError) as exc:
            parser.exit(1, f"Cannot audit WLASL metadata: {exc}\n")
        summary = {key: value for key, value in report.items() if key != "glosses"}
        if args.output:
            summary["report"] = str(args.output)
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "probe-wlasl":
        if not 1 <= args.limit <= 10:
            parser.exit(1, "Probe limit must be between 1 and 10.\n")
        try:
            sign = find_sign(load_metadata(args.metadata), args.gloss)
            if sign is None:
                parser.exit(1, f"WLASL gloss not found: {args.gloss}\n")
            report = probe_gloss(sign, args.limit)
        except (OSError, ValueError, RuntimeError) as exc:
            parser.exit(1, f"Cannot probe WLASL videos: {exc}\n")
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "split-wlasl":
        try:
            signs = load_metadata(args.metadata)
            vocabulary = load_vocabulary(args.vocabulary)
            if vocabulary.dataset.casefold() != "wlasl":
                raise ValueError("Vocabulary dataset must be WLASL.")
            report = split_metadata(signs, vocabulary.labels, seed=args.seed)
            with args.metadata.open("rb") as source:
                report["metadata_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
            with args.vocabulary.open("rb") as source:
                report["vocabulary_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                json.dump(report, output, indent=2)
                output.write("\n")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            parser.exit(1, f"Cannot create WLASL split: {exc}\n")
        summary = {
            key: value for key, value in report.items() if key not in {"records", "coverage"}
        }
        summary["output"] = str(args.output)
        print(json.dumps(summary, indent=2))
        return 0
    try:
        vocabulary = load_vocabulary(args.path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(1, f"Invalid vocabulary: {exc}\n")
    print(
        json.dumps(
            {
                "language": vocabulary.language,
                "dataset": vocabulary.dataset,
                "class_count": len(vocabulary.labels),
                "labels": vocabulary.labels,
            },
            indent=2,
        )
    )
    return 0
