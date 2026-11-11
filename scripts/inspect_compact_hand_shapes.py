"""Check existing NO/YES/HELLO features for preserved finger separation and motion.

No test data, webcam, training or source edits. These are estimates, not joint GT.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def quantiles(values):
    return {str(q):float(np.quantile(values,q)) for q in (.1,.5,.9)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    cache=ROOT/'data/training/asl-five-signs-128-v1';records=[];summary={}
    for split in ('train','val'):
        features=np.load(cache/f'{split}-features.npy',mmap_mode='r')
        lengths=np.load(cache/f'{split}-lengths.npy');rows=json.loads((cache/f'{split}-index.json').read_text())
        for i,row in enumerate(rows):
            if row['gloss'] not in ('HELLO','NO','YES'):continue
            clip=np.asarray(features[i,:int(lengths[i])],np.float32)
            present=(clip[:,249:251]>0)&(clip[:,251:253]>0)
            slot=int(present.sum(0).argmax());ids=np.flatnonzero(present[:,slot])
            if len(ids)<2:
                records.append({**row,'usable_frames':len(ids),'insufficient':True});continue
            points=clip[:,123:249].reshape(-1,2,21,3)[ids,slot]
            vectors={}
            for name,tip in [('thumb_index',8),('thumb_middle',12)]:
                delta=points[:,tip]-points[:,4]
                vectors[name+'_xy']=np.linalg.norm(delta[:,:2],axis=1)
                vectors[name+'_xyz']=np.linalg.norm(delta,axis=1)
            vectors['index_tip_to_wrist']=np.linalg.norm(points[:,8],axis=1)
            vectors['middle_tip_to_wrist']=np.linalg.norm(points[:,12],axis=1)
            record={**row,'dominant_slot':slot,'usable_frames':len(ids),
                    'metrics':{name:quantiles(values) for name,values in vectors.items()},
                    'thumb_index_xy_p90_minus_p10':float(np.quantile(vectors['thumb_index_xy'],.9)-np.quantile(vectors['thumb_index_xy'],.1)),
                    'xyz_minus_xy_thumb_index_median':float(np.median(vectors['thumb_index_xyz']-vectors['thumb_index_xy']))}
            records.append(record)
        summary[split]={}
        for gloss in ('HELLO','NO','YES'):
            chosen=[r for r in records if r['split']==split and r['gloss']==gloss and not r.get('insufficient')]
            summary[split][gloss]={'clips':len(chosen),
                'clip_median_thumb_index_xy':quantiles([r['metrics']['thumb_index_xy']['0.5'] for r in chosen]),
                'clip_median_thumb_index_xyz':quantiles([r['metrics']['thumb_index_xyz']['0.5'] for r in chosen]),
                'clip_thumb_index_xy_range':quantiles([r['thumb_index_xy_p90_minus_p10'] for r in chosen]),
                'clip_median_index_tip_to_wrist':quantiles([r['metrics']['index_tip_to_wrist']['0.5'] for r in chosen])}
    result={'scope':'Existing training/validation features only; no live coordinates available',
        'units':'Wrist-relative coordinates divided by per-clip median palm size, not pixels or metres',
        'limits':'Dominant canonical slot; includes detected rest; finite sampling, occlusion and slot errors can affect measures. No ground-truth joint accuracy or recognition-cause claim.',
        'summary':summary,'records':records}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
