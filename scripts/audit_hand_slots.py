"""Flag nearby single-hand slot changes in existing prepared features.

Heuristic only: zero modifications or inference. A change can also be caused by
alternate detections of two hands, so source-video review is required.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'data/training/asl-gru-128-v1'


def slot_changes(features, lengths):
    present=features[:,:,249:251]>0
    single=present.sum(axis=-1)==1
    slot=present.argmax(axis=-1)
    anchor=features[:,:,253]>0
    wrist=np.where(slot[:,:,None]==0,features[:,:,39:41],features[:,:,81:83]).astype(np.float32)
    nearby=np.linalg.norm(wrist[:,1:]-wrist[:,:-1],axis=-1)<.5
    valid=np.arange(features.shape[1]-1)[None,:]<(lengths[:,None]-1)
    eligible=single[:,1:]&single[:,:-1]&anchor[:,1:]&anchor[:,:-1]&valid
    changes=eligible&(slot[:,1:]!=slot[:,:-1])
    return changes, changes&nearby, eligible


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    results=[];summary={}
    for split in ('train','val','test'):
        index=json.loads((CACHE/f'{split}-index.json').read_text())
        features=np.load(CACHE/f'{split}-features.npy',mmap_mode='r')
        lengths=np.load(CACHE/f'{split}-lengths.npy',mmap_mode='r')
        totals={'clips':len(index),'clips_with_nearby_slot_changes':0,'nearby_slot_changes':0,'eligible_adjacent_pairs':0}
        for start in range(0,len(index),512):
            end=min(start+512,len(index));changes,nearby,eligible=slot_changes(features[start:end],lengths[start:end])
            totals['clips_with_nearby_slot_changes']+=int(nearby.any(axis=1).sum())
            totals['nearby_slot_changes']+=int(nearby.sum());totals['eligible_adjacent_pairs']+=int(eligible.sum())
            for offset in np.flatnonzero(nearby.any(axis=1)):
                i=start+int(offset)
                results.append({**index[i],'index':i,'slot_changes':int(changes[offset].sum()),
                                'nearby_slot_changes':int(nearby[offset].sum()),
                                'sampled_frame_indices':(np.flatnonzero(nearby[offset])+1).tolist()})
        summary[split]=totals
        print(split,totals,flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'slot-change-candidates.json').write_text(json.dumps(results,indent=2)+'\n')
    (args.output/'slot-change-summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
