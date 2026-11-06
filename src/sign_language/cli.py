"""Preparation and hand visualization commands. No training command is exposed."""

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ASL Citizen setup and MediaPipe hand skeletons")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup-hands", help="Download Google's pretrained landmark detector")
    download = commands.add_parser("download-dataset", help="Resume the complete official ASL Citizen ZIP")
    download.add_argument("--workers", type=int, default=6)
    commands.add_parser("extract-dataset", help="Extract, CRC-check and audit the official splits")
    commands.add_parser("audit-dataset", help="Check counts, labels, file presence and signer separation")
    metadata = commands.add_parser("audit-metadata", help="Check CSVs without claiming videos are present")
    metadata.add_argument("splits", type=Path)
    webcam = commands.add_parser("webcam", help="Show live hand skeletons; Q closes the window")
    webcam.add_argument("--camera", type=int, default=0)
    webcam.add_argument("--duration", type=float, default=0)
    webcam.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.command == "webcam" and args.report and args.report.exists():
        parser.error("Report already exists; choose a new filename")
    if args.command == "setup-hands":
        from sign_language.assets import setup_hands
        report = setup_hands()
    elif args.command == "download-dataset":
        from sign_language.dataset import download_dataset
        report = download_dataset(args.workers)
    elif args.command == "extract-dataset":
        from sign_language.dataset import extract_dataset
        report = extract_dataset()
    elif args.command == "audit-dataset":
        from sign_language.dataset import audit_dataset
        report = audit_dataset()
    elif args.command == "audit-metadata":
        from sign_language.dataset import audit_metadata
        report = audit_metadata(args.splits)
    else:
        from sign_language.webcam import run_webcam
        report = run_webcam(args.camera, args.duration)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return int(report.get("status") == "failed")
