"""Stage two: keep the learned GRU, refit one L2 readout on training clips only."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import warnings

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.trajectory_geometry import FEATURES,CONFIG,trajectory_batch,fit_statistics
from sign_language.gru_trajectory_readout import SignTrajectoryReadoutGRU
from sign_language.recognition import SignRecognizer
from sign_language.five_sign_pilot import sha256,write_json
from sign_language.training import LandmarkDataset,atomic_checkpoint
from sign_language.expanded_pilot import SIGNS
from run_fifteen_sign_temporal_trial import CACHE,evaluate
from run_please_trajectory_fix import qualifies

OUTPUT=ROOT/'models/asl-gru-fifteen-signs-trajectory-readout-v1'


def collect(backbone,split):
    pooled=[];geometry=[]
    hook=backbone.model.classifier.register_forward_pre_hook(lambda module,args:pooled.append(args[0][:,:256].detach().float().cpu().numpy()))
    try:
        with torch.inference_mode():
            for frames,lengths,_ in DataLoader(LandmarkDataset(CACHE,split),batch_size=16):
                geometry.append(trajectory_batch(frames,lengths))
                with torch.autocast('cuda',dtype=torch.float16):
                    backbone.model(frames.cuda(),lengths)
    finally:hook.remove()
    return np.concatenate(pooled),np.concatenate(geometry)


def main(regularization=1., output=OUTPUT):
    global OUTPUT
    OUTPUT=Path(output)
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError('C must be positive and finite')
    if OUTPUT.exists():raise FileExistsError('Preserve prior attempt')
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    manifest=json.loads((CACHE/'manifest.json').read_text())
    for n,h in manifest['files_sha256'].items():
        if sha256(CACHE/n)!=h:raise ValueError('Cache checksum differs')
    baseline_path=ROOT/'models/asl-gru-fifteen-signs-temporal-v1/temporal/best.pt'
    protected={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'models').rglob('*')
               if p.is_file() and (p.name=='best.pt' or p.suffix=='.pkl')}
    baseline=SignRecognizer(baseline_path,ROOT/'models',device='cuda')
    baseline_state=torch.load(baseline_path,map_location='cpu',weights_only=True)
    train_hidden,train_geometry=collect(baseline,'train')
    statistics=fit_statistics(train_geometry)
    args=baseline.config['model']
    candidate=SignTrajectoryReadoutGRU(**args)
    candidate.gru.load_state_dict(baseline.model.gru.state_dict())
    candidate.set_geometry_statistics(statistics)
    if not all(torch.equal(candidate.gru.state_dict()[n].cpu(),v.cpu()) for n,v in baseline.model.gru.state_dict().items()):
        raise AssertionError('Backbone changed')
    z=candidate.standardize_geometry(torch.tensor(train_geometry)).detach().numpy()
    x=np.concatenate([train_hidden,z],axis=1)
    targets=np.load(CACHE/'train-targets.npy')
    OUTPUT.mkdir()
    protocol={'status':'prespecified','stage':'frozen_backbone_linear_readout','backbone_checkpoint_sha256':baseline.checkpoint_sha256,
              'backbone_epoch':baseline.epoch,'features':list(FEATURES),'training_rows':len(x),
              'optimizer':f'L2 multinomial logistic regression, C={regularization:g}, lbfgs, max_iter=2000, random_state=42',
              'inverse_regularization_C':regularization,
              'pooled_hidden_scaling':'None; bounded GRU output retained. Geometry uses train-only z-score.',
              'fit_on':'original official training clips only; no validation rows or augmentation',
              'selection_disclosure':'C=1 was the initial attempt. Follow-up C values are development validation tuning; see study report.',
              'acceptance':'PLEASE 4/4, MY 4/4, overall >=50/60 development validation',
              'test_read':False,'webcam_opened':False,'validation_design_informed':True,'defaults_changed':False}
    write_json(OUTPUT/'experiment.json',protocol)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        head=LogisticRegression(C=regularization,solver='lbfgs',max_iter=2000,random_state=42).fit(x,targets)
    if not np.array_equal(head.classes_,np.arange(15)):raise AssertionError('Label order differs')
    with torch.no_grad():
        candidate.classifier.weight.copy_(torch.from_numpy(head.coef_).float())
        candidate.classifier.bias.copy_(torch.from_numpy(head.intercept_).float())
    source=ROOT/'src/sign_language'
    config={'architecture':'gru-mean-pool-trajectory-readout-v1','model':args,'preprocessing':baseline.config['preprocessing'],
            'cache_id':manifest['cache_id'],'geometry_features':list(FEATURES),'geometry_statistics':statistics,
            'trajectory_geometry':CONFIG,'cudnn_enabled':False,'amp':True,'head_arithmetic':'float32',
            'training_stage':'frozen_backbone_linear_readout','backbone_epoch':baseline.epoch,
            'head_fit_iterations':head.n_iter_.tolist(),'head_training':protocol,
            'source_sha256':{n:sha256(source/n) for n in ('gru_model.py','gru_mean_pool_model.py','gru_geometry_model.py',
              'hand_articulation.py','gru_trajectory_model.py','trajectory_geometry.py','wrist_trajectory.py','gru_trajectory_readout.py','preprocessing.py')}}
    write_json(OUTPUT/'config.json',config);write_json(OUTPUT/'labels.json',manifest['labels'])
    atomic_checkpoint(OUTPUT/'best.pt',{'config':config,'model':candidate.state_dict(),'labels':manifest['labels'],
                                     'epoch':baseline.epoch,'backbone_epoch':baseline.epoch,'head_fit_iterations':head.n_iter_.tolist()})
    candidate=candidate.cuda().eval()
    train=evaluate(candidate,'train',list(SIGNS));validation=evaluate(candidate,'val',list(SIGNS))
    _,val_geometry=collect(baseline,'val')
    val_hidden,_=collect(baseline,'val')
    with torch.no_grad():
        val_z=candidate.standardize_geometry(torch.from_numpy(val_geometry).cuda()).cpu().numpy()
    reference=head.predict_proba(np.concatenate([val_hidden,val_z],axis=1))
    actual=np.asarray([p['scores'] for p in validation['predictions']])
    if not np.array_equal(reference.argmax(1),actual.argmax(1)):raise AssertionError('Export changed readout decisions')
    report={'status':'complete','protocol':protocol,'train':train,'validation':validation,'acceptance_met':qualifies(validation),
            'parameters':sum(p.numel() for p in candidate.parameters()),'readout_coefficients':int(head.coef_.size+head.intercept_.size),
            'checkpoint_sha256':sha256(OUTPUT/'best.pt'),'export_max_score_difference':float(np.abs(reference-actual).max()),
            'backbone_identical':True,'previous_weights_and_probes_unchanged':all(sha256(ROOT/n)==h for n,h in protected.items()),
            'subset_unchanged':all(sha256(CACHE/n)==h for n,h in manifest['files_sha256'].items())}
    if not report['previous_weights_and_probes_unchanged'] or not report['subset_unchanged']:raise AssertionError('Protected sources changed')
    snapshot=OUTPUT/'source_snapshot';snapshot.mkdir()
    for n in config['source_sha256']:shutil.copy2(source/n,snapshot/n)
    shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    write_json(OUTPUT/'report.json',report)
    if OUTPUT.name=='asl-gru-fifteen-signs-trajectory-readout-v1':
        write_json(ROOT/'docs/assets/please-trajectory-readout-2026-11-12.json',report)
    print(json.dumps({'acceptance_met':report['acceptance_met'],'correct':validation['correct'],'per_class':validation['per_class']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--c',type=float,default=1.)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    options=parser.parse_args()
    main(options.c,options.output)
