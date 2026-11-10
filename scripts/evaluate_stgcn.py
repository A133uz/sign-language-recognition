"""Evaluate a validation-selected ST-GCN checkpoint once on the official test split."""

import argparse
import json
from pathlib import Path
import sys

from train_stgcn import ROOT, frozen_source, DEFAULT_OUTPUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_OUTPUT/'best.pt')
    parser.add_argument('--cache', type=Path, default=ROOT/'data/training/asl-gru-128-v1')
    parser.add_argument('--output', type=Path, default=ROOT/'data/evaluations/asl-stgcn27-test-v1')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    sys.path.insert(0, str(frozen_source(args.checkpoint.parent)))
    from sign_language.evaluation import evaluate
    result = evaluate(args.checkpoint, args.cache, ROOT/'data/asl_citizen/ASL_Citizen', args.output,
                      batch_size=args.batch_size, workers=args.workers)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
