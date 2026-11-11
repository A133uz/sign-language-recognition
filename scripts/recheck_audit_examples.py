"""Re-extract two public diagnostic clips in memory to check cache reproducibility."""
import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sign_language.raw_landmarks import extract_clip
from sign_language.preprocessing import normalize_sequence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    cv2.setNumThreads(1)
    directory=args.output/'visual-review'
    groups=json.loads((directory/'focused-train-val.json').read_text())
    requested={'0035172227152260316-ME.mp4','9009854854743657-YES.mp4'}
    rows=[r for group in groups.values() for r in group if r['video_file'] in requested]
    if len(rows)!=2:raise ValueError('Expected two audit examples')
    results=[]
    for r in rows:
        data,metadata=extract_clip(ROOT/'data/asl_citizen/ASL_Citizen/videos'/r['video_file'],
                                  ROOT/'models/hand_landmarker.task',ROOT/'models/pose_landmarker_lite.task')
        with np.load(io.BytesIO(data),allow_pickle=False) as loaded:
            fresh={k:loaded[k] for k in loaded.files}
        with zipfile.ZipFile(ROOT/'data/landmarks/asl-citizen-raw-v1-cpu-hands-pose'/f"{r['shard']}.zip") as z:
            with np.load(io.BytesIO(z.read(r['member'])),allow_pickle=False) as loaded:
                saved={k:loaded[k] for k in loaded.files}
        comparison={k:{'exact':bool(np.array_equal(saved[k],fresh[k])),
                       'max_absolute_difference':float(np.max(np.abs(saved[k].astype(np.float64)-fresh[k].astype(np.float64))))}
                    for k in saved}
        old,_=normalize_sequence(saved,r['frame_shape']);new,_=normalize_sequence(fresh,r['frame_shape'])
        results.append({'video_file':r['video_file'],'frames':metadata['frames'],
                        'metadata_equal':all(metadata[k]==r[k] for k in metadata),
                        'arrays':comparison,'float16_prepared_equal':bool(np.array_equal(old.astype(np.float16),new.astype(np.float16))),
                        'fresh_frames_with_hands':metadata['frames_with_hands']})
    path=args.output/'fresh-extraction-check.json';path.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
