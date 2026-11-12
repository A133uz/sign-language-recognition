"""Step 1: fixed train-only binary geometry probe; no changes to main recognizers."""
import json
import pickle
from pathlib import Path
import platform
import shutil
import sys
import time

import torch # Initialize Windows native libraries consistently; fitting below is CPU-only.
import numpy as np
import sklearn

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.five_sign_pilot import sha256,write_json
from sign_language.hand_articulation import summarize_articulation
from sign_language.yes_no_geometry import FEATURES,LABELS,vectorize,make_probe,evaluate,grouped_cross_validation


def predictions(rows,probabilities):
    return [{'video_file':r.get('video_file'),'participant_id':r.get('participant_id'),
        'capture_id':r.get('capture_id'),'expected':r.get('gloss',r.get('expected')),
        'predicted':LABELS[int(p.argmax())],'yes_score':float(p[1]),
        'correct':LABELS[int(p.argmax())]==r.get('gloss',r.get('expected'))} for r,p in zip(rows,probabilities,strict=True)]


def main():
    output=ROOT/'models/asl-yes-no-geometry-diagnostic-v1'
    if output.exists():raise FileExistsError('Diagnostic already exists; preserve it and use a new version')
    cache=ROOT/'data/training/asl-five-signs-128-v1';manifest=json.loads((cache/'manifest.json').read_text())
    source_names=[f'{split}-{kind}.{extension}' for split in ('train','val')
                  for kind,extension in [('features','npy'),('lengths','npy'),('targets','npy'),('index','json')]]
    sources={name:sha256(cache/name) for name in source_names}
    if any(digest!=manifest['files_sha256'][name] for name,digest in sources.items()):raise ValueError('Source cache differs')
    protected={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'models').rglob('best.pt')}
    data={}
    for split in ('train','val'):
        rows=json.loads((cache/f'{split}-index.json').read_text());f=np.load(cache/f'{split}-features.npy',mmap_mode='r')
        lengths=np.load(cache/f'{split}-lengths.npy');targets=np.load(cache/f'{split}-targets.npy');selected=[]
        for i,row in enumerate(rows):
            if row['gloss'] not in LABELS:continue
            if targets[i]!=manifest['labels'][row['gloss']]:raise ValueError('Index/target disagreement')
            articulation=summarize_articulation(f[i,:int(lengths[i])])
            selected.append({**row,'articulation':articulation})
        data[split]={'rows':selected,'x':np.stack([vectorize(r['articulation']) for r in selected]),
            'y':np.asarray([LABELS.index(r['gloss']) for r in selected]),
            'groups':np.asarray([r['participant_id'] for r in selected])}
    if set(data['train']['groups'])&set(data['val']['groups']):raise ValueError('Signer leakage')
    output.mkdir(parents=True)
    config={'experiment':'step_1_fixed_yes_no_geometry_probe','diagnostic_only':True,
        'labels':list(LABELS),'features':list(FEATURES),'cache_id':manifest['cache_id'],
        'model':'Median imputation + missingness indicators + StandardScaler + logistic regression',
        'parameters':{'C':1.,'solver':'lbfgs','max_iter':2000,'random_state':42},
        'evaluation':'3-fold signer-grouped OOF on training only, then fixed fit on all training and evaluation on official validation',
        'hyperparameter_search':False,'official_test_examples_read':False,'webcam_opened':False,
        'main_models_changed':False,'real_time_velocities_included':False,
        'feature_scope':'Clip-level geometry/angle/motion-excursion summaries; no participant, filename, duration or presence-fraction features',
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'scikit_learn':sklearn.__version__},
        'source_sha256':{name:sha256(ROOT/'src/sign_language'/name) for name in ('hand_articulation.py','yes_no_geometry.py')},
        'train_val_cache_sha256':sources}
    write_json(output/'config.json',config)
    train=data['train'];val=data['val'];started=time.perf_counter()
    oof,folds=grouped_cross_validation(train['x'],train['y'],train['groups'])
    model=make_probe();model.fit(train['x'],train['y'])
    if not np.array_equal(model.classes_,[0,1]):raise AssertionError('Label order changed')
    with (output/'model.pkl').open('wb') as stream:pickle.dump(model,stream)
    report={'status':'complete','config':config,'counts':{s:{g:int((d['y']==i).sum()) for i,g in enumerate(LABELS)} for s,d in data.items()},
        'unique_signers':{s:len(set(d['groups'])) for s,d in data.items()},
        'train_fit':evaluate(train['y'],model.predict_proba(train['x'])),
        'train_grouped_oof':evaluate(train['y'],oof),'folds':folds,
        'validation':evaluate(val['y'],model.predict_proba(val['x'])),
        'predictions':{'train_fit':predictions(train['rows'],model.predict_proba(train['x'])),
                       'train_grouped_oof':predictions(train['rows'],oof),
                       'validation':predictions(val['rows'],model.predict_proba(val['x']))},
        'dataset_scalar_records':{s:d['rows'] for s,d in data.items()},
        'seconds':time.perf_counter()-started,'model_sha256':sha256(output/'model.pkl')}
    names=model.named_steps['impute'].get_feature_names_out(FEATURES)
    report['standardized_coefficients']=[{'feature':str(n),'coefficient_toward_yes':float(w)}
        for n,w in zip(names,model.named_steps['classifier'].coef_[0],strict=True)]
    # A frozen, retrospective check only: previous live summaries influenced the
    # diagnostic design, so these four rows are not a fresh independent benchmark.
    live_path=ROOT/'docs/assets/compact-hand-live-2026-11-12.json'
    if live_path.exists():
        previous=json.loads(live_path.read_text())['live']
        if previous['user_completion_confirmed'] is not True:raise ValueError('Retrospective live labels unconfirmed')
        rows=[t for t in previous['trials'] if t['expected'] in LABELS]
        x=np.stack([vectorize(t['articulation']) for t in rows]);y=np.asarray([LABELS.index(t['expected']) for t in rows])
        report['retrospective_live']={'metrics':evaluate(y,model.predict_proba(x)),
            'predictions':predictions(rows,model.predict_proba(x)),
            'source_sha256':sha256(live_path),'used_for_fitting_or_parameter_selection':False,
            'limits':'Previously inspected four scalar summaries; design-informed, not fresh live validation; only NO/YES inputs.'}
    report['main_weights_unchanged']=all(sha256(ROOT/name)==digest for name,digest in protected.items())
    report['train_val_cache_unchanged']=all(sha256(cache/name)==digest for name,digest in sources.items())
    if not report['main_weights_unchanged'] or not report['train_val_cache_unchanged']:raise AssertionError('Protected sources changed')
    snapshot=output/'source_snapshot';snapshot.mkdir()
    for name in config['source_sha256']:shutil.copy2(ROOT/'src/sign_language'/name,snapshot/name)
    shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    write_json(output/'report.json',report)
    write_json(ROOT/'docs/assets/yes-no-geometry-diagnostic-2026-11-12.json',report)
    print(json.dumps({key:report[key] for key in ('counts','unique_signers','train_fit','train_grouped_oof','validation','main_weights_unchanged')},indent=2))
    if 'retrospective_live' in report:print('RETROSPECTIVE_LIVE',json.dumps(report['retrospective_live']))


if __name__=='__main__':main()
