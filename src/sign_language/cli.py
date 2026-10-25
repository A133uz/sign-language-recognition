"""Setup diagnostics; recognition and training are separate future work."""

import argparse
import importlib
import json
import platform
from pathlib import Path

import yaml

from sign_language import __version__
from sign_language.config import load_vocabulary


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
    args = parser.parse_args(argv)
    if args.command == "doctor":
        report, success = doctor(
            vision=args.vision, require_cuda=args.require_cuda, require_torch=args.torch
        )
        print(json.dumps(report, indent=2))
        return 0 if success else 1
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
