"""Descriptive public YOU/YES hand-shape probes; no fitting or model intervention."""
import io
import json
from pathlib import Path
import sys
import zipfile

import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.pronoun_landmark_review import canonical,hand_shape_summary
from sign_language.five_sign_pilot import sha256


def main():
    output=ROOT/'docs/assets/you-yes-hand-shape-probes-2026-11-14.json'
    if output.exists():raise FileExistsError('Preserve the previous probe report')
    cache=ROOT/'data/training/asl-fifteen-signs-128-v1';raw=ROOT/'data/landmarks/asl-citizen-raw-v1-cpu-hands-pose'
    manifest=json.loads((cache/'manifest.json').read_text());protected={str(cache/n):h for n,h in manifest['files_sha256'].items()}
    checkpoint=ROOT/'models/asl-gru-fifteen-signs-trajectory-readout-c01/best.pt';protected[str(checkpoint)]=sha256(checkpoint)
    records=[]
    for split in ('train','val'):
        rows=json.loads((cache/f'{split}-index.json').read_text())
        for row in rows:
            if row['gloss'] not in ('YOU','YES'):continue
            parent=row['parent_index'];path=raw/f'{split}-{parent//100:05d}.zip'
            if str(path) not in protected:
                marker=json.loads(path.with_suffix('.complete.json').read_text())
                assert path.stat().st_size==marker['bytes'] and sha256(path)==marker['sha256']
                protected[str(path)]=marker['sha256']
            with zipfile.ZipFile(path) as z:
                record=json.loads(z.read('records.json'))['records'][parent%100]
                assert all(record[k]==row[k] for k in ('video_file','gloss','participant_id','split'))
                with np.load(io.BytesIO(z.read(record['member'])),allow_pickle=False) as source:
                    arrays={k:source[k] for k in source.files}
            hands,present=canonical(arrays,record['frame_shape']);ids=np.flatnonzero(present.any(1))
            index=int(ids[len(ids)//2]) if len(ids) else None
            metrics=hand_shape_summary(hands,present,index,record['frame_shape'][1]) if index is not None else []
            # Select dominant slot by complete-clip presence, independent of gloss and geometry values.
            dominant=int(present.sum(0).argmax())
            metric=next((m for m in metrics if m['slot']==dominant),None)
            records.append({**row,'probe_frame_index':index,'dominant_slot':dominant,'metrics':metric})
    names=('projected_tip_to_mcp_over_palm','projected_tip_to_wrist_over_palm',
           'projected_index_chord_over_bone_path','estimated_local_index_z_span_over_palm')
    stats={}
    for gloss in ('YOU','YES'):
        chosen=[r['metrics'] for r in records if r['gloss']==gloss and r['metrics']]
        stats[gloss]={'probes':len(chosen),'features':{}}
        for name in names:
            a=np.array([r[name] for r in chosen if r[name] is not None])
            stats[gloss]['features'][name]={'min':float(a.min()),'p10':float(np.quantile(a,.1)),
                'median':float(np.median(a)),'p90':float(np.quantile(a,.9)),'max':float(a.max())}
    assert all(sha256(Path(p))==h for p,h in protected.items())
    report={'status':'complete','records':records,'statistics':stats,
            'selection':'Middle tracked frame, then dominant slot from whole-clip hand presence; fixed before viewing metrics. Train and val only.',
            'limits':['A tracked frame can be rest or transition, not verified sign posture.',
                      'Projected hand ratios are view-dependent; local hand z is a model estimate, not measured physical depth.',
                      'Descriptor overlap is not a classifier or a proof of cause.'],
            'geometry_disabled':False,'model_inference_performed':False,'training':False,'test_read':False,
            'protected_sha256':{str(Path(p).relative_to(ROOT)):h for p,h in protected.items()},'protected_unchanged':True}
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
