"""Run the 50-epoch face-aware 15-sign GRU with one subtitle-style guess."""
from datetime import datetime
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from sign_language.face_gru_recognition import FaceAwareRecognizer
from webcam_700_subtitles import run


def main():
    torch.set_num_threads(2)
    checkpoint=ROOT/'models/asl-gru-fifteen-signs-face-50epochs-v1/model.pt'
    recognizer=FaceAwareRecognizer(checkpoint,ROOT/'models',device='cuda')
    if len(recognizer.names)!=15 or recognizer.epoch!=36:raise ValueError('Unexpected face-aware checkpoint')
    report=ROOT/f'.cache/webcam-15-face-50epochs-{datetime.now():%Y%m%d-%H%M%S}.json'
    run(recognizer,report,title='ASL 15 - FACE GRU - 50 epochs, selected 36')

if __name__=='__main__':main()
