"""Train ST-GCN on the fixed five-sign subset, up to 33 epochs, on the local GPU."""
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


if __name__=='__main__':
    from sign_language.five_sign_stgcn import run
    run(ROOT)
