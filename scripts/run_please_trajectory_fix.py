"""Train-only trajectory-feature experiment; target-specific development acceptance."""
import json
from pathlib import Path
import shutil
import sys

import torch
import numpy as np
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sign_language.five_sign_pilot import sha256,write_json,select_epoch
from sign_language.gru_trajectory_model import SignTrajectoryMeanPoolGRU
from sign_language.trajectory_geometry import FEATURES,CONFIG,trajectory_batch,fit_statistics
from sign_language.training import LandmarkDataset,cuda_check
from sign_language.expanded_pilot import SIGNS,verify_parent_chunks
from run_fifteen_sign_temporal_trial import train_variant,evaluate,CACHE,PARENT

OUTPUT=ROOT/'models/asl-gru-fifteen-signs-trajectory-v1'


def read(path):return json.loads(path.read_text())


def qualifies(metrics):
    return (metrics['per_class']['PLEASE']['correct']==4 and
            metrics['per_class']['MY']['correct']==4 and metrics['correct']>=50)


def main():
    if OUTPUT.exists():raise FileExistsError('Preserve prior experiment; use a new version')
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False;torch.backends.cudnn.benchmark=False
    gpu=cuda_check()
    manifest=read(CACHE/'manifest.json')
    for name,h in manifest['files_sha256'].items():
        if sha256(CACHE/name)!=h:raise ValueError('Source subset checksum differs')
    verify_parent_chunks(PARENT,manifest)
    protected={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'models').rglob('*')
               if p.is_file() and (p.name=='best.pt' or p.suffix=='.pkl')}
    x=np.concatenate([trajectory_batch(frames,lengths) for frames,lengths,_ in
                      DataLoader(LandmarkDataset(CACHE,'train'),batch_size=16)])
    statistics=fit_statistics(x)
    OUTPUT.mkdir()
    protocol={'status':'prespecified','gpu':gpu,'features':list(FEATURES),'trajectory_geometry':CONFIG,
              'cache_id':manifest['cache_id'],'fit_split':'219 official training clips only',
              'acceptance':'PLEASE 4/4 and MY 4/4 and overall >=50/60 on development validation',
              'selection':'Keep normal small-gap best.pt; additionally select target-best.pt among qualifying epochs using the same competitive-validation/small-gap rule',
              'validation_design_informed':True,'test_read':False,'webcam_opened':False,
              'missing_training_values':int(np.isnan(x).sum()),'defaults_changed':False,
              'baseline_report_sha256':sha256(ROOT/'docs/assets/fifteen-sign-temporal-2026-11-12.json')}
    write_json(OUTPUT/'experiment.json',protocol)
    snapshot=OUTPUT/'source_snapshot';snapshot.mkdir()
    for name in ('trajectory_geometry.py','gru_trajectory_model.py','wrist_trajectory.py','gru_geometry_model.py',
                 'gru_model.py','gru_mean_pool_model.py','hand_articulation.py','preprocessing.py','regularization.py','temporal_augmentation.py','training.py','five_sign_pilot.py'):
        shutil.copy2(ROOT/'src/sign_language'/name,snapshot/name)
    for path in (Path(__file__),ROOT/'scripts/run_fifteen_sign_temporal_trial.py'):
        shutil.copy2(path,snapshot/path.name)
    try:
        run=train_variant('temporal',manifest,statistics,output_root=OUTPUT,model_class=SignTrajectoryMeanPoolGRU,
                          geometry_features=FEATURES,architecture='gru-mean-pool-trajectory-v1',
                          additional_config={'trajectory_geometry':CONFIG,'development_target':protocol['acceptance']},
                          extra_sources=('trajectory_geometry.py','gru_trajectory_model.py','wrist_trajectory.py'),
                          capture_validation=True)
        eligible=[r for r in run['epoch_metrics'] if qualifies(r['validation_details'])]
        target=None
        if eligible:
            chosen,selection=select_epoch(eligible)
            folder=OUTPUT/'temporal'
            shutil.copy2(folder/f'epoch-{chosen["epoch"]:03d}.pt',folder/'target-best.pt')
            state=torch.load(folder/'target-best.pt',map_location='cpu',weights_only=True)
            model=SignTrajectoryMeanPoolGRU(**state['config']['model']).cuda().eval();model.load_state_dict(state['model'])
            target={'selection':selection,'checkpoint_sha256':sha256(folder/'target-best.pt'),
                    'train':evaluate(model,'train',list(SIGNS)),'validation':evaluate(model,'val',list(SIGNS))}
            target['acceptance_met']=qualifies(target['validation'])
            if not target['acceptance_met']:raise AssertionError('Target checkpoint rescoring differs')
            del model;torch.cuda.empty_cache()
        baseline=read(ROOT/'docs/assets/fifteen-sign-temporal-2026-11-12.json')['temporal']
        report={'status':'complete','protocol':protocol,'training':run,'target_checkpoint':target,
                'qualifying_epochs':[r['epoch'] for r in eligible],
                'baseline':{k:baseline[k] for k in ('selection','checkpoint_sha256','train','validation')},
                'previous_weights_and_probes_unchanged':all(sha256(ROOT/n)==h for n,h in protected.items()),
                'subset_unchanged':all(sha256(CACHE/n)==h for n,h in manifest['files_sha256'].items()),
                'parent_selected_chunks_unchanged':verify_parent_chunks(PARENT,manifest)}
        if not report['previous_weights_and_probes_unchanged'] or not report['subset_unchanged']:raise AssertionError('Protected artifacts changed')
        write_json(OUTPUT/'report.json',report);write_json(ROOT/'docs/assets/please-trajectory-fix-2026-11-12.json',report)
        print('TRAJECTORY_FIX_COMPLETE',json.dumps({'qualifying_epochs':report['qualifying_epochs'],
              'target_validation':target['validation']['correct'] if target else None,
              'normal_selected_epoch':run['selection']['selected_epoch']}),flush=True)
    except BaseException as exc:
        write_json(OUTPUT/'failure.json',{'type':type(exc).__name__,'error':str(exc)});raise


if __name__=='__main__':main()
