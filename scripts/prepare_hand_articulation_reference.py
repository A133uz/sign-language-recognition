"""Reference scalar hand articulation from the existing train/val cache only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.hand_articulation import summarize_articulation
from sign_language.five_sign_pilot import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    cache=ROOT/'data/training/asl-five-signs-128-v1';rows=[]
    manifest=json.loads((cache/'manifest.json').read_text())
    for split in ('train','val'):
        f=np.load(cache/f'{split}-features.npy',mmap_mode='r');n=np.load(cache/f'{split}-lengths.npy')
        index=json.loads((cache/f'{split}-index.json').read_text())
        for i,row in enumerate(index):
            if row['gloss'] in ('NO','YES','HELLO'):
                rows.append({**row,'articulation':summarize_articulation(f[i,:int(n[i])])})
    write_json(a.output,{'cache_id':manifest['cache_id'],'scope':'45 training and 12 validation clips; test not read',
                         'coordinates_saved':False,'units':'palm/shoulder normalized scalar distances; estimated orientation in degrees',
                         'records':rows})
    print('ARTICULATION_REFERENCE_READY',len(rows),flush=True)


if __name__=='__main__':main()
