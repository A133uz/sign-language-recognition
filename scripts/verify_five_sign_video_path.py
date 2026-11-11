"""Check four public HELLO validation videos through the webcam prediction path."""
from pathlib import Path
import sys
import json
import math

import torch
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sign_language.recognition import SignRecognizer
from sign_language.five_sign_pilot import sha256, write_json


def main():
    output=ROOT/'docs/assets/five-sign-gru-video-path-2026-11-11.json'
    if output.exists():raise FileExistsError('Verification evidence already exists')
    torch.set_num_threads(2);cv2.setNumThreads(1)
    checkpoint=ROOT/'models/asl-gru-five-signs-early-v2/regularized/best.pt'
    recognizer=SignRecognizer(checkpoint,ROOT/'models');before=sha256(checkpoint)
    cache=ROOT/'data/training/asl-five-signs-128-v1'
    index=json.loads((cache/'val-index.json').read_text())
    features=np.load(cache/'val-features.npy',mmap_mode='r');lengths=np.load(cache/'val-lengths.npy',mmap_mode='r')
    results=[]
    for i,row in enumerate(index):
        if row['gloss']!='HELLO':continue
        video=ROOT/'data/asl_citizen/ASL_Citizen/videos'/row['video_file']
        capture=cv2.VideoCapture(str(video));frames=[];timestamps=[]
        fps=float(capture.get(cv2.CAP_PROP_FPS));declared=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            if not capture.isOpened() or not math.isfinite(fps) or fps<=0:raise ValueError('Invalid video')
            while True:
                ok,frame=capture.read()
                if not ok:break
                timestamps.append(max(timestamps[-1]+1 if timestamps else 0,round(len(frames)*1000/fps)))
                frames.append(frame)
        finally:capture.release()
        if len(frames)!=declared:raise ValueError('Incomplete public-video decode')
        actual=recognizer.predict_frames(frames,timestamps)
        with torch.inference_mode(),torch.autocast('cuda'):
            logits=recognizer.model(torch.from_numpy(np.array(features[i],dtype=np.float32))[None].cuda(),torch.tensor([int(lengths[i])])).float()
        values,ids=logits.softmax(-1).topk(5)
        expected_names=[recognizer.names[k] for k in ids[0].tolist()]
        difference=max(abs(item['score']-value) for item,value in zip(actual['top5'],values[0].tolist(),strict=True))
        results.append({'video_file':row['video_file'],'participant_id':row['participant_id'],
                        'expected':'HELLO','source_fps':fps,'frame_shape':list(frames[0].shape),
                        'cached_top1':expected_names[0],'same_label_ranking':[r['label'] for r in actual['top5']]==expected_names,
                        'max_score_difference':difference,'frame_to_landmark_to_gru':actual})
        print(row['video_file'],actual['top5'][0],'score delta',difference,flush=True)
    report={'clips':len(results),'split':'validation','checkpoint_sha256':before,
            'checkpoint_unchanged':before==sha256(checkpoint),'webcam_opened':False,
            'training_performed':False,'all_rankings_equal':all(r['same_label_ranking'] for r in results),
            'maximum_score_difference':max(r['max_score_difference'] for r in results),'results':results}
    write_json(output,report)
    print('Public video verification saved; no camera opened.',flush=True)


if __name__=='__main__':main()
