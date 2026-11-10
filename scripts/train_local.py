"""Prepare cached landmarks or train the ASL GRU on a verified CUDA device."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_directory():
    """Keep existing CV jobs/resumes on their immutable pre-ablation engine."""
    study = ROOT/'models/asl-gru-128-cv-regularized-v1'
    snapshot = study/'engine_snapshot'
    inherited = os.environ.get('SIGN_LANGUAGE_ENGINE_SNAPSHOT')
    if inherited:
        if Path(inherited).resolve() != snapshot.resolve():
            raise ValueError('Unknown training-engine snapshot')
        return snapshot/'src'
    if '--output' in sys.argv:
        index = sys.argv.index('--output')+1
        if index < len(sys.argv) and Path(sys.argv[index]).resolve().is_relative_to(study.resolve()):
            if (snapshot/'manifest.json').exists():
                os.environ['SIGN_LANGUAGE_ENGINE_SNAPSHOT'] = str(snapshot.resolve())
                return snapshot/'src'
    return ROOT/'src'


sys.path.insert(0, str(source_directory()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'train', 'cuda-check'))
    parser.add_argument('--cache', type=Path, default=ROOT/'data/training/asl-gru-128-v1')
    parser.add_argument('--output', type=Path, default=ROOT/'models/asl-gru-128-baseline')
    parser.add_argument('--workers', type=int, choices=(0, 1, 2, 3, 4), default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--profile', choices=('baseline', 'regularized-v1'), default='baseline')
    parser.add_argument('--fold-file', type=Path)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--hidden-size', type=int, choices=(64, 128, 256), default=256)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error('Epochs, batch size and patience must be positive')
    if args.action == 'prepare':
        if args.workers == 0:
            parser.error('Preparation requires at least one worker')
        from sign_language.training_cache import prepare_cache
        result = prepare_cache(ROOT/'data/landmarks/asl-citizen-raw-v1-cpu-hands-pose',
                               ROOT/'data/asl_citizen/ASL_Citizen', args.cache, args.workers)
    else:
        from sign_language.training import cuda_check, train
        if args.action == 'cuda-check':
            result = cuda_check()
        else:
            if args.smoke and args.output == ROOT/'models/asl-gru-128-baseline':
                args.output = ROOT/'models/asl-gru-128-smoke'
            size_option = {'hidden_size': args.hidden_size} if args.hidden_size != 256 else {}
            result = train(args.cache, args.output, epochs=1 if args.smoke else args.epochs,
                           batch_size=args.batch_size, workers=args.workers, patience=args.patience,
                           resume=args.resume, smoke=args.smoke, profile=args.profile,
                           fold_file=args.fold_file, fold=args.fold, **size_option)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
