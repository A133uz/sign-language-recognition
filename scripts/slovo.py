"""Audit, prepare, or evaluate the additive Slovo RSL pipeline."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def concise(value):
    if isinstance(value, dict):
        return {key: concise(item) for key, item in value.items()}
    if isinstance(value, list) and len(value) > 20:
        return {"count": len(value), "examples": value[:8]}
    if isinstance(value, list):
        return [concise(item) for item in value]
    return value


def main():
    defaults = {
        "annotations": ROOT / "SlovoDataset" / "annotations.csv",
        "landmarks": ROOT / "SlovoDataset" / "slovo_mediapipe.json",
        "cache": ROOT / "data" / "training" / "slovo-hands-128-v1",
        "evaluation": ROOT / "data" / "evaluations" / "slovo-gru-v1-report.json",
    }
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)

    audit = commands.add_parser("audit", help="Check CSV/landmark joins and split integrity")
    audit.add_argument("--annotations", type=Path, default=defaults["annotations"])
    audit.add_argument("--landmarks", type=Path, default=defaults["landmarks"])
    audit.add_argument("--report", type=Path, help="Optional JSON report destination")

    prepare = commands.add_parser("prepare", help="Stream landmarks into a signer-aware GRU cache")
    prepare.add_argument("--annotations", type=Path, default=defaults["annotations"])
    prepare.add_argument("--landmarks", type=Path, default=defaults["landmarks"])
    prepare.add_argument("--output", type=Path, default=defaults["cache"])
    prepare.add_argument("--validation-fraction", type=float, default=0.2)
    prepare.add_argument("--seed", type=int, default=42)

    evaluate = commands.add_parser("evaluate", help="Evaluate official and unseen-signer test protocols")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--cache", type=Path, default=defaults["cache"])
    evaluate.add_argument("--output", type=Path, default=defaults["evaluation"])
    evaluate.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    evaluate.add_argument("--batch-size", type=int, default=128)

    args = parser.parse_args()
    if args.action == "audit":
        from sign_language.slovo_dataset import audit_slovo, write_json
        result = audit_slovo(args.annotations, args.landmarks)
        if args.report:
            write_json(args.report, result)
    elif args.action == "prepare":
        from sign_language.slovo_dataset import prepare_cache
        result = prepare_cache(args.annotations, args.landmarks, args.output,
                               validation_fraction=args.validation_fraction, seed=args.seed)
    else:
        from sign_language.slovo_training import evaluate_slovo
        result = evaluate_slovo(args.checkpoint, args.cache, args.output,
                                device=args.device, batch_size=args.batch_size)
    # Windows PowerShell terminals may use a legacy code page; escaped JSON
    # remains readable there and Unicode labels stay intact in report files.
    print(json.dumps(concise(result), ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
