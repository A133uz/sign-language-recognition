"""User-approved two-still review; no video/coordinate files or training writes."""
from pathlib import Path
import argparse
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sign_language.recognition import SignRecognizer
from sign_language.recognition_webcam import run_recognition
from sign_language.landmark_still_review import StillReviewRecognizer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--camera',type=int,default=0)
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to((ROOT/'.cache').resolve()):parser.error('Review output must stay inside project .cache')
    torch.set_num_threads(2)
    recognizer=SignRecognizer(ROOT/'models/asl-gru-five-signs-early-v2/regularized/best.pt',ROOT/'models')
    reviewer=StillReviewRecognizer(recognizer,output)
    run_recognition(reviewer,output/f'{output.name}.json',args.camera,max_seconds=4.,
                    capture_fps=30.,expected_sign='HELLO',auto_finish=True)


if __name__=='__main__':main()
