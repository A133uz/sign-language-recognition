"""Compare the selected pilot's live-input path with its cached validation inputs."""
from pathlib import Path
import sys
import io
import json
import zipfile
import argparse

import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sign_language.recognition import SignRecognizer
from sign_language.five_sign_pilot import sha256, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'models/asl-gru-five-signs-early-v2/regularized/best.pt')
    args=parser.parse_args()
    torch.set_num_threads(2)
    output=args.checkpoint.resolve().parent
    cache=ROOT/'data/training/asl-five-signs-128-v1'
    recognizer=SignRecognizer(args.checkpoint,ROOT/'models')
    if recognizer.config['cache_id']!=json.loads((cache/'manifest.json').read_text())['cache_id'] or len(recognizer.names)!=5:
        raise ValueError('Checkpoint does not match the five-sign cache')
    index=json.loads((cache/'val-index.json').read_text())
    features=np.load(cache/'val-features.npy',mmap_mode='r')
    lengths=np.load(cache/'val-lengths.npy',mmap_mode='r')
    checks=[]
    before=sha256(args.checkpoint)
    for i,row in enumerate(index):
        shard,offset=divmod(row['parent_index'],100)
        with zipfile.ZipFile(ROOT/f'data/landmarks/asl-citizen-raw-v1-cpu-hands-pose/val-{shard:05d}.zip') as z:
            record=json.loads(z.read('records.json'))['records'][offset]
            if record['video_file']!=row['video_file']:raise ValueError('Row identity mismatch')
            with np.load(io.BytesIO(z.read(record['member'])),allow_pickle=False) as loaded:
                arrays={k:loaded[k] for k in loaded.files}
        actual=recognizer.predict_observations(arrays,record['frame_shape'])['top5']
        with torch.inference_mode(),torch.autocast('cuda'):
            logits=recognizer.model(torch.from_numpy(np.array(features[i],dtype=np.float32))[None].cuda(),torch.tensor([int(lengths[i])]))
        scores,ids=logits.float().softmax(-1).topk(5,dim=-1)
        names=[recognizer.names[k] for k in ids[0].tolist()]
        difference=max(abs(item['score']-score) for item,score in zip(actual,scores[0].tolist(),strict=True))
        checks.append({'video_file':row['video_file'],'labels_equal':[r['label'] for r in actual]==names,
                       'max_score_difference':difference,'top1':actual[0]['label'],'target':row['gloss']})
    result={'clips':len(checks),'all_labels_equal':all(r['labels_equal'] for r in checks),
            'maximum_score_difference':max(r['max_score_difference'] for r in checks),
            'checkpoint_unchanged':before==sha256(args.checkpoint),'epoch':recognizer.epoch,
            'model':type(recognizer.model).__name__,
            'classes':recognizer.names,'checks':checks,'webcam_opened':False}
    write_json(output/'runtime-consistency.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
    if not result['all_labels_equal'] or result['maximum_score_difference']>1e-6 or not result['checkpoint_unchanged']:
        raise AssertionError('Live-input and cache paths disagree')


if __name__=='__main__':main()
