"""Controlled dataset-only letterboxing probe; original checkpoint stays frozen."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import cv2
import numpy as np
import torch

from sign_language.dataset import write_json
from sign_language.recognition import SignRecognizer
from sign_language.recognition_diagnostics import observation_statistics, score_observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output file')
    audit = json.loads(args.audit.read_text())
    torch.set_num_threads(2)
    recognizer = SignRecognizer(ROOT/'models/asl-gru-128-cv-regularized-v1/official-refit/best.pt', ROOT/'models')
    rows = []
    for item in audit['results']['raw_cache_and_fresh_video_checks']:
        capture = cv2.VideoCapture(str(ROOT/'data/asl_citizen/ASL_Citizen/videos'/item['video_file']))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            smaller = cv2.resize(frame, (round(w*.6), round(h*.6)), interpolation=cv2.INTER_AREA)
            canvas = np.zeros_like(frame)
            top, left = (h-smaller.shape[0])//2, (w-smaller.shape[1])//2
            canvas[top:top+smaller.shape[0], left:left+smaller.shape[1]] = smaller
            frames.append(canvas)
        capture.release()
        arrays, shape, metadata = recognizer.observe_frames(frames, [round(i*1000/fps) for i in range(len(frames))])
        result = score_observations(recognizer, arrays, shape, item['gloss'])
        rows.append({'gloss': item['gloss'], 'split': item['split'], 'video_file': item['video_file'],
                     'scale': .6, 'native_prediction': item['cached_raw_prediction'],
                     'resized_prediction': result, 'statistics': observation_statistics(arrays, shape)})
        print(item['split'], item['gloss'], 'native', item['cached_raw_prediction'][0]['label'],
              'scaled', result['top5'][0]['label'], 'rank', result['expected_rank_min'], flush=True)
        write_json(args.output, {'status': 'running', 'training_performed': False, 'records': rows})
    write_json(args.output, {'status': 'complete', 'training_performed': False,
                            'note': '60% image-scale letterbox probe; borders are an additional image change; not a webcam accuracy estimate',
                            'records': rows})


if __name__ == '__main__':
    main()
