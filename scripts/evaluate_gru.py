"""Evaluate the validation-selected regularized GRU on the official held-out test set."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT/'models/asl-gru-128-cv-regularized-v1/official-refit/best.pt')
    parser.add_argument('--cache', type=Path, default=ROOT/'data/training/asl-gru-128-v1')
    parser.add_argument('--output', type=Path, default=ROOT/'data/evaluations/asl-gru-regularized-test-v1')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    from sign_language.evaluation import evaluate
    report = evaluate(args.checkpoint, args.cache, ROOT/'data/asl_citizen/ASL_Citizen', args.output,
                      batch_size=args.batch_size, workers=args.workers)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
