"""Run a selected frozen ASL GRU or ST-GCN on manually bounded webcam signs."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT/'models/asl-gru-five-signs-early-v2/regularized/best.pt')
    parser.add_argument('--models', type=Path, default=ROOT/'models')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--max-seconds', type=float, default=8.)
    parser.add_argument('--capture-fps', type=float, default=30.,
                        help='Capture FPS limit (1-60), default 30; slower cameras are not upsampled')
    parser.add_argument('--expected-sign',type=str.upper,
                        help='Optional diagnostic prompt label, saved in the log; never used for prediction')
    parser.add_argument('--report', type=Path,
                        default=ROOT/f'.cache/webcam-recognition-{datetime.now():%Y%m%d-%H%M%S}.json')
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Report exists; choose a new path')
    if not 1 <= args.capture_fps <= 60:
        parser.error('Capture FPS must be between 1 and 60')
    import torch
    torch.set_num_threads(2)
    from sign_language.recognition import SignRecognizer
    from sign_language.recognition_webcam import run_recognition
    recognizer = SignRecognizer(args.checkpoint, args.models, args.device)
    if args.expected_sign is not None and args.expected_sign not in recognizer.names:
        parser.error('Expected sign must be one of the selected checkpoint labels')
    print(f'MODEL_READY | model={type(recognizer.model).__name__} | epoch={recognizer.epoch} | classes={len(recognizer.names)} | device={args.device}', flush=True)
    report = run_recognition(recognizer, args.report, args.camera, args.max_seconds,
                             capture_fps=args.capture_fps,expected_sign=args.expected_sign)
    print(f'FINISHED | {len(report["trials"])} trials | {args.report}', flush=True)


if __name__ == '__main__':
    main()
