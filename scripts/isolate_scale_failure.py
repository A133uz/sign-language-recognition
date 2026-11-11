"""Explain one observed public-video scale failure, not a new benchmark."""
import argparse
import json
from pathlib import Path
import sys

import torch
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.recognition import SignRecognizer
from sign_language.preprocessing import normalize_sequence
from sign_language.recognition_diagnostics import observation_statistics
from sign_language.recognition_forensics import infer_clips,scores
from sign_language.five_sign_pilot import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--camera-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    report=json.loads(a.camera_report.read_text())
    candidates=[x for x in report['results'] if x['variants']['original']['correct'] and not x['variants']['person_scale_065']['correct']]
    if not candidates:raise ValueError('No scale failure to isolate')
    row=candidates[0];torch.set_num_threads(2);cv2.setNumThreads(1)
    r=SignRecognizer(ROOT/'models/asl-gru-five-signs-early-v2/regularized/best.pt',ROOT/'models')
    cap=cv2.VideoCapture(str(ROOT/'data/asl_citizen/ASL_Citizen/videos'/row['video_file']));frames=[];timestamps=[]
    try:
        fps=cap.get(cv2.CAP_PROP_FPS);declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or fps<=0:raise ValueError('Video invalid')
        while True:
            ok,f=cap.read()
            if not ok:break
            timestamps.append(max(timestamps[-1]+1 if timestamps else 0,round(len(frames)*1000/fps)));frames.append(f)
    finally:cap.release()
    if len(frames)!=declared:raise ValueError('Incomplete decode')
    h,w=frames[0].shape[:2];sw,sh=round(w*.65),round(h*.65);dw,dh=(w-sw)//2,(h-sh)//2
    scaled=[]
    for frame in frames:
        out=np.zeros_like(frame);out[dh:dh+sh,dw:dw+sw]=cv2.resize(frame,(sw,sh),interpolation=cv2.INTER_AREA);scaled.append(out)
    arrays,shape,_=r.observe_frames(frames,timestamps);small,_,_=r.observe_frames(scaled,timestamps)
    original,_=normalize_sequence(arrays,shape);altered,_=normalize_sequence(small,shape)
    # Exact same estimated landmarks under a known image-coordinate transform.
    theoretical={k:v.copy() for k,v in arrays.items()}
    for key in ('hands_xyz','pose_xyz'):
        theoretical[key][...,0]=theoretical[key][...,0]*(sw/w)+dw/w
        theoretical[key][...,1]=theoretical[key][...,1]*(sh/h)+dh/h
        theoretical[key][...,2]*=sw/w
    ideal,_=normalize_sequence(theoretical,shape)
    clips={'original':original,'scaled_fresh_mediapipe':altered,'scaled_exact_landmarks':ideal}
    for name,channels in {
        'restore_pose':list(range(39))+[253],
        'restore_pose_coordinates':list(range(26)),
        'restore_pose_masks':list(range(26,39))+[253],
        'restore_face':list(range(14))+list(range(26,33)),
        'restore_shoulders':list(range(14,18))+[33,34,253],
        'restore_pose_arms':list(range(18,26))+list(range(35,39)),
        'restore_body_hands':list(range(39,123)),
        'restore_local_xy':[i for i in range(123,249) if (i-123)%3!=2],
        'restore_local_depth':list(range(125,249,3)),
        'restore_all_local':list(range(123,249))+[251,252],
        'restore_all_hands':list(range(39,253)),
        'restore_masks':list(range(26,39))+list(range(249,254)),
    }.items():
        x=altered.copy();x[:,channels]=original[:,channels];clips[name]=x
    logits,_=infer_clips(r,[x.astype(np.float16).astype(np.float32) for x in clips.values()])
    result={'video_file':row['video_file'],'expected':row['gloss'],
        'selection':'First original-correct / reduced-scale-wrong row in prespecified six-video control. Explanatory selection, not a new evaluation set.',
        'checkpoint_sha256':r.checkpoint_sha256,'coordinates_saved':False,'media_saved':False,
        'exact_coordinate_transform_max_feature_difference':float(np.max(np.abs(ideal-original))),
        'source_statistics':observation_statistics(arrays,shape),'scaled_statistics':observation_statistics(small,shape),
        'pose_validity_changes_by_group':{name:int((arrays['pose_valid_mask'][:,ids]!=small['pose_valid_mask'][:,ids]).sum())
               for name,ids in [('face',[0,2,5,7,8,9,10]),('shoulders',[11,12]),('arms',[13,14,15,16])]},
        'feature_group_rms_difference':{name:float(np.sqrt(np.mean((original[:,start:end]-altered[:,start:end])**2)))
             for name,start,end in [('pose',0,26),('body_hands',39,123),('local_hands',123,249),('hand_masks',249,253)]},
        'variants':{name:scores(z,r.names,r.labels[row['gloss']]) for name,z in zip(clips,logits)},
        'limits':'Restoring groups is a controlled transplant between two renderings of the same public video; neither detector output is ground-truth joints.'}
    write_json(a.output,result)
    print(json.dumps({k:result[k] for k in ('video_file','exact_coordinate_transform_max_feature_difference','feature_group_rms_difference','variants')},indent=2),flush=True)


if __name__=='__main__':main()
