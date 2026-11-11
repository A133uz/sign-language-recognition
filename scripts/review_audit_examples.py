"""Create exact public-dataset video/landmark boards; no webcam or training data writes."""
import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sign_language.preprocessing import normalize_sequence

HANDS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
         (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
         (13,17),(0,17),(17,18),(18,19),(19,20)]
POSE = [(11,12),(11,13),(13,15),(12,14),(14,16)]
RAW = ROOT/'data/landmarks/asl-citizen-raw-v1-cpu-hands-pose'
CACHE = ROOT/'data/training/asl-gru-128-v1'
VIDEOS = ROOT/'data/asl_citizen/ASL_Citizen/videos'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    grouped={g:[] for g in ('HELLO','THANKYOU','NO','ME','YES')}
    for split in ('train','val'):
        index=json.loads((CACHE/f'{split}-index.json').read_text())
        shards={}
        for i,row in enumerate(index):
            if row['gloss'] in grouped: shards.setdefault(i//100,[]).append((i%100,row))
        for shard,selected in shards.items():
            name=f'{split}-{shard:05d}'
            with zipfile.ZipFile(RAW/f'{name}.zip') as z:
                records=json.loads(z.read('records.json'))['records']
                for offset,row in selected:
                    rec=records[offset]
                    with np.load(io.BytesIO(z.read(rec['member'])),allow_pickle=False) as a:
                        hp=a['hand_present'].any(axis=1)
                        _,q=normalize_sequence(a,rec['frame_shape'])
                    grouped[row['gloss']].append({**rec,'shard':name,'offset':offset,
                        'hand_coverage':float(hp.mean()),'anchor_coverage':q['frames_with_body_anchor']/len(hp)})
    (args.output/'focused-train-val.json').write_text(json.dumps(grouped,indent=2)+'\n')
    selected=[]
    for gloss,rows in grouped.items():
        ordered=sorted(rows,key=lambda r:(r['hand_coverage'],r['video_file']))
        if not ordered: continue
        for reason,rec in [('lowest coverage',ordered[0]),('median coverage',ordered[len(ordered)//2])]:
            selected.append({**rec,'selection':reason})
    for gloss in grouped:
        rows=[r for r in selected if r['gloss']==gloss]
        board=np.zeros((len(rows)*286,5*320,3),np.uint8)
        for rowno,rec in enumerate(rows):
            with zipfile.ZipFile(RAW/f"{rec['shard']}.zip") as z:
                with np.load(io.BytesIO(z.read(rec['member'])),allow_pickle=False) as loaded:
                    a={k:loaded[k] for k in loaded.files}
            ids=np.linspace(0,rec['frames']-1,5).round().astype(int)
            capture=cv2.VideoCapture(str(VIDEOS/rec['video_file']))
            for col,frame_id in enumerate(ids):
                capture.set(cv2.CAP_PROP_POS_FRAMES,int(frame_id));ok,frame=capture.read()
                if not ok: continue
                h,w=frame.shape[:2]
                for left,right in POSE:
                    if a['pose_valid_mask'][frame_id,[left,right]].all():
                        p,q=(a['pose_xyz'][frame_id,[left,right],:2]*[w,h]).astype(int)
                        cv2.line(frame,tuple(p),tuple(q),(255,180,0),2)
                for slot in (0,1):
                    if not a['hand_present'][frame_id,slot]: continue
                    points=(a['hands_xyz'][frame_id,slot,:,:2]*[w,h]).astype(int)
                    for p,q in HANDS: cv2.line(frame,tuple(points[p]),tuple(points[q]),(0,230,0),2)
                    for point in points: cv2.circle(frame,tuple(point),2,(0,255,255),-1)
                thumb=cv2.resize(frame,(320,240))
                x=col*320;y=rowno*286
                board[y+46:y+286,x:x+320]=thumb
                text=f"{rec['split']} {rec['participant_id']} frame {frame_id} hands={int(a['hand_present'][frame_id].sum())}"
                cv2.putText(board,text,(x+4,y+40),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)
            capture.release()
            text=f"{gloss}: {rec['selection']}; hands {rec['hand_coverage']:.1%}; {rec['video_file']}"
            cv2.putText(board,text,(4,rowno*286+16),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
        cv2.imwrite(str(args.output/f'{gloss.lower()}-review.png'),board)
    (args.output/'selected-examples.json').write_text(json.dumps(selected,indent=2)+'\n')
    print(json.dumps({g:{'train_val_clips':len(rs),'zero_hand_clips':sum(r['hand_coverage']==0 for r in rs),
                        'median_hand_coverage':float(np.median([r['hand_coverage'] for r in rs])) if rs else None}
                      for g,rs in grouped.items()},indent=2),flush=True)


if __name__=='__main__': main()
