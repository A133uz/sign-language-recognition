"""Bounded GRU sensitivity checks on 75 train / 19 val clips; no webcam or training.

Interventions are destructive controls, not label-preserving augmentation or fixes.
No test examples are rescored. Source features and checkpoint remain unchanged.
"""
from collections import Counter
import json
from pathlib import Path
import sys

import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sign_language.five_sign_pilot import sha256, write_json
from sign_language.recognition import SignRecognizer

VARIANTS=('original','static_final_frame','zero_pose_features','zero_hand_features',
          'crop_to_detected_hand_span','fill_trailing_missing_hands')


def transform(features,lengths,variant):
    result=features.copy();new_lengths=lengths.copy()
    for i,length in enumerate(lengths):
        n=int(length);clip=result[i,:n]
        if variant=='original':continue
        if variant=='static_final_frame':clip[:]=clip[-1].copy()
        elif variant=='zero_pose_features':clip[:,:39]=0
        elif variant=='zero_hand_features':clip[:,39:253]=0
        elif variant in ('crop_to_detected_hand_span','fill_trailing_missing_hands'):
            ids=np.flatnonzero(np.any(clip[:,249:251]>0,axis=1))
            if len(ids):
                if variant=='crop_to_detected_hand_span':
                    trimmed=clip[ids[0]:ids[-1]+1].copy();result[i]=0
                    result[i,:len(trimmed)]=trimmed;new_lengths[i]=len(trimmed)
                else:clip[ids[-1]+1:,39:253]=clip[ids[-1],39:253].copy()
        else:raise ValueError(variant)
    return result,new_lengths


def score(recognizer,features,lengths,targets,rows):
    predicted=[];probabilities=[];loss=0.
    with torch.inference_mode():
        for start in range(0,len(features),16):
            frames=torch.from_numpy(features[start:start+16]).cuda()
            true=torch.from_numpy(targets[start:start+16]).cuda()
            with torch.autocast('cuda'):
                logits=recognizer.model(frames,torch.as_tensor(lengths[start:start+16],dtype=torch.long))
                loss+=float(torch.nn.functional.cross_entropy(logits,true,reduction='sum'))
            probabilities.extend(logits.float().softmax(-1).cpu().tolist())
            predicted.extend(logits.argmax(-1).cpu().tolist())
    names=recognizer.names
    predictions=[{'video_file':row['video_file'],'expected':names[int(y)],'predicted':names[p],
                  'no_score':probs[recognizer.labels['NO']]} for row,y,p,probs in zip(rows,targets,predicted,probabilities,strict=True)]
    return {'clips':len(features),'correct':int(np.sum(np.asarray(predicted)==targets)),
            'top1':float(np.mean(np.asarray(predicted)==targets)),'loss':loss/len(features),
            'predicted_counts':{g:sum(names[p]==g for p in predicted) for g in names},
            'per_class_correct':{g:sum(r['expected']==g and r['predicted']==g for r in predictions) for g in names},
            'predictions':predictions}


def main():
    output=ROOT/'.cache/five-sign-gru-no-collapse-20261111-v1'
    if output.exists():raise FileExistsError('Use a new diagnostic output directory')
    cache=ROOT/'data/training/asl-five-signs-128-v1'
    checkpoint=ROOT/'models/asl-gru-five-signs-early-v2/regularized/best.pt'
    torch.set_num_threads(2)
    recognizer=SignRecognizer(checkpoint,ROOT/'models')
    before=sha256(checkpoint)
    session_path=ROOT/'.cache/webcam-gru5-regularized-epoch18-20261111-01.json'
    session=json.loads(session_path.read_text())
    if before!=session['checkpoint_sha256']:raise ValueError('Live/offline checkpoint differs')
    manifest=json.loads((cache/'manifest.json').read_text())
    hashes_before={name:sha256(cache/name) for name in manifest['files_sha256']}
    if hashes_before!=manifest['files_sha256']:raise ValueError('Subset payload differs')
    report={'checkpoint_sha256':before,'model_class':type(recognizer.model).__name__,
            'epoch':recognizer.epoch,'labels':recognizer.labels,'model_in_eval_mode':not recognizer.model.training,
            'scope':'Existing 75 training and 19 validation rows only; no training, video decode, camera or full audit.',
            'controls_are_label_preserving':False,'splits':{}}
    for split in ('train','val'):
        features=np.load(cache/f'{split}-features.npy').astype(np.float32)
        lengths=np.load(cache/f'{split}-lengths.npy').astype(np.int64)
        targets=np.load(cache/f'{split}-targets.npy')
        index=json.loads((cache/f'{split}-index.json').read_text())
        coverage=[float(np.any(features[i,:int(n),249:251]>0,axis=1).mean()) for i,n in enumerate(lengths)]
        fraction_ending_no_hands=float(np.mean([not np.any(features[i,int(n)-1,249:251]>0) for i,n in enumerate(lengths)]))
        outcomes={}
        for variant in VARIANTS:
            changed,new_lengths=transform(features,lengths,variant)
            outcomes[variant]=score(recognizer,changed,new_lengths,targets,index)
            r=outcomes[variant]
            print(f'{split} {variant}: {r["correct"]}/{r["clips"]} correct; NO={r["predicted_counts"]["NO"]}',flush=True)
        report['splits'][split]={'median_hand_frame_fraction':float(np.median(coverage)),
                                'fraction_ending_without_hands':fraction_ending_no_hands,'variants':outcomes}
    zero_lengths=np.asarray([t['input_frames'] for t in session['trials']],dtype=np.int64)
    zero_features=np.zeros((len(zero_lengths),128,254),np.float32)
    with torch.inference_mode(),torch.autocast('cuda'):
        zero_probs=recognizer.model(torch.from_numpy(zero_features).cuda(),torch.from_numpy(zero_lengths)).float().softmax(-1).cpu().numpy()
    report['invalid_zero_input_control']=[{'frames':int(n),'predicted':recognizer.names[int(p.argmax())],
                                           'scores':{g:float(p[i]) for i,g in enumerate(recognizer.names)}}
                                          for n,p in zip(zero_lengths,zero_probs,strict=True)]
    offline_folder=checkpoint.parent
    stored_predictions=json.loads((offline_folder/'test-predictions.json').read_text())
    report['existing_test_prediction_counts']=dict(Counter(r['prediction'] for r in stored_predictions))
    report['test_rescored']=False
    report['live_session']={'capture_count':len(session['trials']),
                            'no_first_guesses':sum(t['top5'][0]['label']=='NO' for t in session['trials']),
                            'intended_labels_recorded':False,'raw_frames_or_landmarks_available':False}
    report['checkpoint_unchanged']=before==sha256(checkpoint)
    report['cache_unchanged']=hashes_before=={name:sha256(cache/name) for name in hashes_before}
    if not report['checkpoint_unchanged'] or not report['cache_unchanged']:raise AssertionError('Input changed during read-only checks')
    output.mkdir(parents=True)
    write_json(output/'report.json',report)
    write_json(ROOT/'docs/assets/five-sign-gru-no-collapse-2026-11-11.json',report)
    print('Weights/cache unchanged; diagnostic evidence saved.',flush=True)


if __name__=='__main__':main()
