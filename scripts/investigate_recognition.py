"""Prespecified frozen-GRU experiments on train/validation only; separate scratch heads."""
import argparse
import io
import json
import pickle
from pathlib import Path
import sys
from zipfile import ZipFile

import torch
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.recognition import SignRecognizer
from sign_language.recognition_diagnostics import observation_statistics
from sign_language.recognition_forensics import VARIANTS,make_variant,summary_vector,feature_summary,infer_clips,metrics,scores
from sign_language.preprocessing import normalize_sequence
from sign_language.five_sign_pilot import write_json,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    cache=ROOT/'data/training/asl-five-signs-128-v1'
    checkpoint=ROOT/'models/asl-gru-five-signs-early-v2/regularized/best.pt'
    recognizer=SignRecognizer(checkpoint,ROOT/'models');before=sha256(checkpoint)
    manifest=json.loads((cache/'manifest.json').read_text());payload={f:sha256(cache/f) for f in manifest['files_sha256']}
    if payload!=manifest['files_sha256']:raise ValueError('Cache manifest mismatch')
    datasets={}
    for split in ('train','val'):
        f=np.load(cache/f'{split}-features.npy').astype(np.float32);lengths=np.load(cache/f'{split}-lengths.npy')
        datasets[split]={'clips':[f[i,:int(n)].copy() for i,n in enumerate(lengths)],
                         'targets':np.load(cache/f'{split}-targets.npy'),
                         'rows':json.loads((cache/f'{split}-index.json').read_text())}
    # Deterministic target-blind medoid of final no-hand training frames.
    ends=np.asarray([x[-1] for x in datasets['train']['clips'] if not x[-1,249:251].any()])
    center=np.median(ends,axis=0);dist=np.mean((ends-center)**2,axis=1);rest=ends[int(dist.argmin())].copy()
    report={'status':'running','scope':'75 train / 19 val; test never loaded; no production training or data edits',
            'checkpoint_sha256':before,'variants_are_diagnostic_not_validated_augmentations':True,
            'rest_source':'training no-hand final-frame medoid, target-blind',
            'probe_specification':{'linear':'StandardScaler + multinomial LogisticRegression(C=1,max_iter=2000)',
                                   'mlp':'StandardScaler + MLP(64 ReLU, alpha=.01, max_iter=500, seed=42)',
                                   'selection':'Fixed diagnostics, no hyperparameter search or production replacement'},
            'splits':{},'heads':{},'raw_records':[]}
    bank={'rest':rest,'names':recognizer.names,'checkpoint_sha256':before,'probes':{}}
    for split,data in datasets.items():
        records={};base_hidden=None
        for name in VARIANTS:
            clips=[make_variant(x,name,rest) for x in data['clips']]
            logits,hidden=infer_clips(recognizer,clips)
            records[name]={**metrics(logits,data['targets'],recognizer.names),
                           'predictions':[{'video_file':row['video_file'],'expected':row['gloss'],
                                           **scores(z,recognizer.names,int(y))} for row,z,y in zip(data['rows'],logits,data['targets'])]}
            if name=='original':base_hidden=hidden
            print(split,name,records[name]['correct'],'/',len(clips),flush=True)
        data['hidden']=base_hidden
        report['splits'][split]={'variants':records,'features':[{'video_file':r['video_file'],'gloss':r['gloss'],
                                  **feature_summary(x)} for r,x in zip(data['rows'],data['clips'])]}
    train=datasets['train'];val=datasets['val']
    for kind in ('hidden','timing','pose','hands'):
        x=train['hidden'] if kind=='hidden' else np.stack([summary_vector(x,kind) for x in train['clips']])
        v=val['hidden'] if kind=='hidden' else np.stack([summary_vector(x,kind) for x in val['clips']])
        models={'linear':make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=42))}
        if kind=='hidden':models['mlp']=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(64,),alpha=.01,max_iter=500,random_state=42))
        for name,model in models.items():
            model.fit(x,train['targets']);key=f'{kind}_{name}';bank['probes'][key]=model
            # Store only class decisions/aggregate metrics, not embeddings, in tracked evidence.
            report['heads'][key]={s:metrics(model.predict_proba(z),data['targets'],recognizer.names)
                                  for s,z,data in [('train',x,train),('val',v,val)]}
            print('PROBE',key,report['heads'][key],flush=True)
    bank.update(train_hidden=train['hidden'],train_targets=train['targets'],train_rows=train['rows'],
                train_clips=train['clips'])
    # Audit all 94 selected raw examples, not the stopped whole-corpus cleaning job.
    raw=ROOT/'data/landmarks/asl-citizen-raw-v1-cpu-hands-pose'
    for split,data in datasets.items():
        groups={}
        for i,row in enumerate(data['rows']):groups.setdefault(row['parent_index']//100,[]).append((i,row))
        for shard,items in groups.items():
            with ZipFile(raw/f'{split}-{shard:05d}.zip') as archive:
                records=json.loads(archive.read('records.json'))['records']
                for i,row in items:
                    record=records[row['parent_index']%100]
                    if record['video_file']!=row['video_file']:raise ValueError('Raw row mismatch')
                    with np.load(io.BytesIO(archive.read(record['member'])),allow_pickle=False) as npz:arrays={k:npz[k] for k in npz.files}
                    normalized,_=normalize_sequence(arrays,record['frame_shape'])
                    delta=float(np.max(np.abs(normalized.astype(np.float16).astype(np.float32)-data['clips'][i])))
                    report['raw_records'].append({'split':split,**row,'source_frames':record['frames'],
                        'fps':record['fps'],'duration_seconds':record['frames']/record['fps'],
                        'raw_to_cache_max_delta':delta,'statistics':observation_statistics(arrays,record['frame_shape'])})
        print('RAW_AUDIT',split,len(data['rows']),flush=True)
    report.update(status='complete',checkpoint_unchanged=before==sha256(checkpoint),
                  cache_unchanged=payload=={f:sha256(cache/f) for f in payload},test_loaded=False,
                  production_model_trained=False,scratch_heads_fitted_on_training_only=True)
    if not report['checkpoint_unchanged'] or not report['cache_unchanged']:raise AssertionError('Source mutation')
    with (output/'offline-bank.pkl').open('wb') as stream:pickle.dump(bank,stream)
    write_json(output/'offline-report.json',report)
    print('OFFLINE_COMPLETE',output,flush=True)


if __name__=='__main__':main()
