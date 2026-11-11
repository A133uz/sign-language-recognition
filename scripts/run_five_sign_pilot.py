"""Train a separate five-class baseline (<=10 epochs) and regularized GRU (<=28)."""
from pathlib import Path
import sys

# Torch must initialize before OpenCV/MediaPipe native libraries on this machine.
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


if __name__=='__main__':
    from sign_language.five_sign_training import run
    run(ROOT)
