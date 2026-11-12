"""Matched train-only readouts distinguish refitting from adding path features."""
import json
from pathlib import Path
import sys
import warnings

import torch
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.recognition import SignRecognizer
from sign_language.five_sign_pilot import write_json
from refit_trajectory_readout import collect,CACHE
from sign_language.trajectory_geometry import fit_statistics


def main():
    output=ROOT/'docs/assets/please-trajectory-readout-ablation-2026-11-12.json'
    if output.exists():raise FileExistsError('Preserve prior evidence')
    torch.set_num_threads(2)
    study=json.loads((ROOT/'docs/assets/please-trajectory-readout-study-2026-11-12.json').read_text())
    chosen=study['selected'];c=chosen['C']
    baseline=SignRecognizer(ROOT/'models/asl-gru-fifteen-signs-temporal-v1/temporal/best.pt',ROOT/'models')
    hidden,geometry=collect(baseline,'train');val_hidden,val_geometry=collect(baseline,'val')
    stats=fit_statistics(geometry)
    def zscore(values):
        return (np.where(np.isnan(values),stats['median'],values)-stats['mean'])/stats['scale']
    train_z,val_z=zscore(geometry),zscore(val_geometry)
    targets=np.load(CACHE/'train-targets.npy');val_targets=np.load(CACHE/'val-targets.npy')
    rows=json.loads((CACHE/'val-index.json').read_text())
    results={}
    for dimensions in (17,23):
        x=np.concatenate([hidden,train_z[:,:dimensions]],axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            head=LogisticRegression(C=c,solver='lbfgs',max_iter=2000,random_state=42).fit(x,targets)
        scores=head.predict_proba(np.concatenate([val_hidden,val_z[:,:dimensions]],axis=1))
        correct=scores.argmax(1)==val_targets
        results[str(dimensions)]={'correct':int(correct.sum()),'samples':len(correct),
            'per_class':{g:{'correct':int(correct[val_targets==i].sum()),'support':int((val_targets==i).sum())}
                         for i,g in enumerate(baseline.names)},
            'predictions':[{'video_file':row['video_file'],'expected':baseline.names[int(t)],
                            'prediction':baseline.names[int(p.argmax())],'scores':p.tolist()}
                           for row,t,p in zip(rows,val_targets,scores,strict=True)]}
    report={'status':'complete','protocol':'Same frozen backbone, same original training rows, same C, solver and train-only statistics; only six path features differ.',
            'C':c,'results':results,'official_test_read':False,'deployed_model_changed':False,
            'development_validation_not_independent':True}
    write_json(output,report)
    for k,r in results.items():print(k,r['correct'],{g:r['per_class'][g] for g in ('PLEASE','MY')},flush=True)


if __name__=='__main__':main()
