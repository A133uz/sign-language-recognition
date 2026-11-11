"""Public validation-video controls: scale and resolution, fresh MediaPipe each time."""
import argparse
import json
from pathlib import Path
import sys

import torch
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.recognition import SignRecognizer
from sign_language.recognition_diagnostics import observation_statistics
from sign_language.preprocessing import normalize_sequence
from sign_language.recognition_forensics import infer_clips,scores
from sign_language.five_sign_pilot import write_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    r=SignRecognizer(ROOT/'models/asl-gru-five-signs-early-v2/regularized/best.pt',ROOT/'models')
    rows=json.loads((ROOT/'data/training/asl-five-signs-128-v1/val-index.json').read_text())
    chosen=[row for name in ('HELLO','THANKYOU','NO') for row in [x for x in rows if x['gloss']==name][:2]]
    report={'status':'running','selection':'First two validation-index rows of each HELLO/THANKYOU/NO, regardless of prediction',
            'checkpoint_sha256':r.checkpoint_sha256,'test_loaded':False,'webcam_opened':False,
            'media_saved':False,'production_model_trained':False,'results':[]}
    for row in chosen:
        cap=cv2.VideoCapture(str(ROOT/'data/asl_citizen/ASL_Citizen/videos'/row['video_file']))
        frames=[];timestamps=[]
        try:
            if not cap.isOpened():raise ValueError('Video not opened')
            fps=float(cap.get(cv2.CAP_PROP_FPS));declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not np.isfinite(fps) or fps<=0:raise ValueError('Invalid source FPS')
            while True:
                ok,f=cap.read()
                if not ok:break
                timestamps.append(max(timestamps[-1]+1 if timestamps else 0,round(len(frames)*1000/fps)));frames.append(f)
        finally:cap.release()
        if len(frames)!=declared:raise ValueError('Incomplete decode')
        outcomes={}
        for name in ('original','person_scale_065','half_resolution'):
            altered=[]
            for frame in frames:
                h,w=frame.shape[:2]
                if name=='original':out=frame
                elif name=='half_resolution':
                    out=cv2.resize(cv2.resize(frame,(w//2,h//2),interpolation=cv2.INTER_AREA),(w,h),interpolation=cv2.INTER_LINEAR)
                else:
                    small=cv2.resize(frame,(round(w*.65),round(h*.65)),interpolation=cv2.INTER_AREA)
                    out=np.zeros_like(frame);dh=(h-small.shape[0])//2;dw=(w-small.shape[1])//2
                    out[dh:dh+small.shape[0],dw:dw+small.shape[1]]=small
                altered.append(out)
            arrays,shape,metadata=r.observe_frames(altered,timestamps)
            features,_=normalize_sequence(arrays,shape);features=features.astype(np.float16).astype(np.float32)
            z,_=infer_clips(r,[features])
            outcomes[name]={**scores(z[0],r.names,r.labels[row['gloss']]),
                            'statistics':observation_statistics(arrays,shape),**metadata}
            print(row['gloss'],row['video_file'],name,outcomes[name]['top5'][0],flush=True)
        report['results'].append({**row,'variants':outcomes});write_json(a.output,report)
    report.update(status='complete',checkpoint_unchanged=sha256(r.checkpoint)==r.checkpoint_sha256)
    write_json(a.output,report)


if __name__=='__main__':main()
