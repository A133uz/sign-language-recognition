"""Run the retained 15-label trajectory GRU with one subtitle-style guess."""
from datetime import datetime
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from sign_language.recognition import SignRecognizer
from webcam_700_subtitles import run


class SubtitleRecognizer(SignRecognizer):
    """Expose only the top-ranked sign to the subtitle window and log."""
    def predict_frames(self,frames,timestamps):
        result=super().predict_frames(frames,timestamps)
        guesses=result.pop('top5')
        result['top1']=guesses[0]
        return result


def main():
    torch.set_num_threads(2)
    checkpoint=ROOT/'models/asl-gru-fifteen-signs-trajectory-readout-c01/best.pt'
    recognizer=SubtitleRecognizer(checkpoint,ROOT/'models',device='cuda')
    if len(recognizer.names)!=15 or recognizer.checkpoint_sha256!='34b766c393cf20a77b001004376195e60ffb58943d7b1f9661bd23edbab81c91':
        raise ValueError('Unexpected 15-sign checkpoint')
    report=ROOT/f'.cache/webcam-15-trajectory-subtitles-{datetime.now():%Y%m%d-%H%M%S}.json'
    run(recognizer,report,title='ASL 15 - GRU trajectory readout - epoch 13')


if __name__=='__main__':main()
