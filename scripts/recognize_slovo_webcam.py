"""Run a Slovo GRU checkpoint on manually bounded webcam signs."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "slovo-gru-v1" / "best.pt")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument("--report", type=Path,
                        default=ROOT / ".cache" / f"slovo-webcam-{datetime.now():%Y%m%d-%H%M%S}.json")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report exists; choose a new path")
    import torch
    torch.set_num_threads(2)
    from sign_language.slovo_recognition import SlovoRecognizer
    from sign_language.recognition_webcam import run_recognition
    recognizer = SlovoRecognizer(args.checkpoint, args.models, args.device)
    print(f"MODEL_READY | dataset=Slovo | model=GRU | epoch={recognizer.epoch} | "
          f"classes={len(recognizer.names)} | device={args.device}", flush=True)
    report = run_recognition(recognizer, args.report, args.camera, args.max_seconds)
    print(f"FINISHED | {len(report['trials'])} trials | {args.report}", flush=True)


if __name__ == "__main__":
    main()
