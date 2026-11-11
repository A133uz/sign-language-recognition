"""Additional target-blind phase controls, train/validation only."""
import argparse
import json
from pathlib import Path
import pickle
import sys

import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.recognition import SignRecognizer
from sign_language.recognition_forensics import phase_lengths,match_training_phases,infer_clips,metrics,scores
from sign_language.five_sign_pilot import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    with a.bank.open('rb') as f:bank=pickle.load(f)
    torch.set_num_threads(2);r=SignRecognizer(ROOT/'models/asl-gru-five-signs-early-v2/regularized/best.pt',ROOT/'models')
    desired=np.median([phase_lengths(x) for x in bank['train_clips'] if phase_lengths(x) is not None],axis=0).astype(int)
    report={'source':'target-blind medians of training-only pre/hand-span/post frame counts',
            'desired_phase_lengths':desired.tolist(),'splits':{},'test_loaded':False,'production_model_trained':False}
    cache=ROOT/'data/training/asl-five-signs-128-v1'
    for split in ('train','val'):
        f=np.load(cache/f'{split}-features.npy').astype(np.float32);n=np.load(cache/f'{split}-lengths.npy')
        y=np.load(cache/f'{split}-targets.npy');rows=json.loads((cache/f'{split}-index.json').read_text())
        report['splits'][split]={}
        for name,motion_only in [('match_training_phases',False),('match_training_motion_span',True)]:
            pairs=[match_training_phases(x[:int(length)],desired,motion_only) for x,length in zip(f,n)]
            z,_=infer_clips(r,[x for x,applied in pairs])
            result={**metrics(z,y,r.names),'applied_clips':sum(applied for x,applied in pairs),
                    'predictions':[{'video_file':row['video_file'],'expected':row['gloss'],**scores(logits,r.names,int(target))}
                                   for row,logits,target in zip(rows,z,y)]}
            report['splits'][split][name]=result;print(split,name,result['correct'],'/',len(y),flush=True)
    write_json(a.output,report)


if __name__=='__main__':main()
