"""Fixed 15-class scalar readout controls; original versus added trajectory features."""
import json
from pathlib import Path
import sys

import torch
import numpy as np
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.trajectory_geometry import trajectory_batch
from sign_language.training import LandmarkDataset
from sign_language.yes_no_geometry import make_probe
from sign_language.five_sign_pilot import write_json


def main():
    torch.set_num_threads(2)
    cache=ROOT/'data/training/asl-fifteen-signs-128-v1'
    labels=list(json.loads((cache/'manifest.json').read_text())['labels'])
    data={s:(np.concatenate([trajectory_batch(f,l) for f,l,y in DataLoader(LandmarkDataset(cache,s),batch_size=16)]),
             np.load(cache/f'{s}-targets.npy')) for s in ('train','val')}
    x,y=data['train'];validation,truth=data['val'];results={}
    for width in (17,23):
        model=make_probe().fit(x[:,:width],y)
        probabilities=model.predict_proba(validation[:,:width]);predicted=probabilities.argmax(1)
        results[str(width)]={'correct':int((predicted==truth).sum()),'samples':len(truth),
            'per_class':{g:{'correct':int(((predicted==truth)&(truth==i)).sum()),'support':int((truth==i).sum())} for i,g in enumerate(labels)},
            'probabilities':probabilities.tolist(),'predicted':predicted.tolist()}
    report={'protocol':'Fixed C=1 train-only logistic regression; 17 versus 23 scalar features; no search, test or deployment',
            'results':results}
    output=ROOT/'docs/assets/trajectory-readout-check-2026-11-12.json'
    if output.exists():raise FileExistsError('Preserve previous diagnostic')
    write_json(output,report)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
