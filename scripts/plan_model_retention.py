"""Build a reviewable checkpoint-pruning plan; performs no deletion."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.five_sign_pilot import sha256,write_json

KEEP=(
 ('gru_regularized_full','Regularized GRU — full vocabulary','models/asl-gru-128-cv-regularized-v1/official-refit/best.pt','Plain last-state GRU reference for the original pipeline and full vocabulary.'),
 ('gru_regularized_five','Regularized GRU — five signs','models/asl-gru-five-signs-early-v2/regularized/best.pt','Like-for-like five-label plain GRU baseline; preserves the existing webcam default.'),
 ('gru_mean_five','Mean-pooled GRU — five signs','models/asl-gru-five-signs-mean-pool-v1/best.pt','Isolates temporal mean pooling from the plain five-label GRU.'),
 ('gru_geometry_five','Mean-pooled GRU + geometry — five signs','models/asl-gru-five-signs-geometry-v1/geometry/epoch-005.pt','Strongest recorded five-label checkpoint; explicit geometry implementation. Retrospective test leader, not independently selected by validation.'),
 ('gru_readout_fifteen','Frozen GRU + trajectory readout — fifteen signs','models/asl-gru-fifteen-signs-trajectory-readout-c01/best.pt','Current fifteen-label implementation; one representative of the near-equivalent C-grid, selected before test evaluation.'),
 ('stgcn_full','ST-GCN — full vocabulary','models/asl-stgcn27-v1/best.pt','Distinct graph-convolution sequence architecture with completed full test evaluation.'),
 ('i3d_official','Official pretrained I3D — full vocabulary','models/asl-citizen-official-i3d-v1/ASL_citizen_I3D_weights.pt','Authors\' independent video-pixel baseline; keeps the original 2731-output weights.'),
)
TEXT={'.json','.py','.md','.txt','.csv','.yaml','.yml','.toml','.sha256','.ps1','.ipynb'}
WEIGHTS={'.pt','.pth','.pkl','.onnx','.h5','.ckpt','.safetensors'}


def main():
    target=ROOT/'docs/assets/model-retention-plan-2026-11-14.json'
    if target.exists():raise FileExistsError('Preserve previous plan')
    benchmark=json.loads((ROOT/'docs/assets/small-vocab-snapshot-test-2026-11-14.json').read_text())
    retained=[];allowed=set()
    for ident,name,checkpoint,reason in KEEP:
        path=ROOT/checkpoint
        if not path.is_file():raise FileNotFoundError(path)
        parent=path.parent;config=json.loads((parent/'config.json').read_text())
        evaluation=None;epoch=None
        if ident in ('gru_regularized_full','stgcn_full'):
            folder='asl-gru-regularized-test-v1' if ident=='gru_regularized_full' else 'asl-stgcn27-test-v1'
            old=json.loads((ROOT/f'data/evaluations/{folder}/report.json').read_text())
            evaluation={'scope':'Full official 2731-label test','classes':2731,**old['metrics']};epoch=old['checkpoint_epoch']
        elif ident=='i3d_official':
            old=json.loads((ROOT/'docs/assets/official-citizen-i3d-baseline-2026-11-14.json').read_text())
            evaluation={'scope':'182-video matched subset, original 2731-label outputs',**{k:v for k,v in old['full_vocabulary'].items() if k!='predictions'}}
        else:
            old=next(r for r in benchmark['checkpoints'] if any(a['checkpoint']==checkpoint for a in r['aliases']))
            evaluation={'scope':'Official selected-vocabulary test','classes':old['classes'],**old['metrics']};epoch=old['epoch']
        if ident!='i3d_official' and config['model']['classes']!=evaluation['classes']:raise ValueError('Vocabulary mismatch')
        dependencies=[checkpoint,str((parent/'config.json').relative_to(ROOT)).replace('\\','/'),str((parent/'labels.json').relative_to(ROOT)).replace('\\','/')]
        allowed.update(dependencies)
        retained.append({'id':ident,'name':name,'checkpoint':checkpoint,'checkpoint_sha256':sha256(path),
                         'epoch':epoch,'architecture':config.get('architecture','gru'),'reason':reason,
                         'evaluation':evaluation,'dependencies':dependencies})
    allowed.update(('models/hand_landmarker.json','models/hand_landmarker.task','models/pose_landmarker_lite.task'))
    files=[]
    for path in sorted((ROOT/'models').rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(ROOT).as_posix();keep=relative in allowed
        preserve_metadata=not keep and path.suffix.lower() in TEXT and '__pycache__' not in path.parts and 'locks' not in path.parts
        files.append({'path':relative,'bytes':path.stat().st_size,'sha256':sha256(path),
                      'action':'keep' if keep else 'preserve_metadata_then_remove' if preserve_metadata else 'remove',
                      'is_checkpoint':path.suffix.lower() in WEIGHTS})
    if allowed-{f['path'] for f in files}:raise ValueError('Missing retained dependencies')
    catalog={'schema':'curated-sign-models-v1','date_local':'2026-11-14','retained_models':retained,
             'history_root':'data/model-history/2026-11-14','training_implementation_rule':'Notebook-first; webcam integration/runtime remains Python.',
             'policy':'Keep meaningful implementations and vocabularies with useful results; remove smoke/fold/resume/epoch duplicates and marginal variants.',
             'new_archive':'models/archives/curated-models-2026-11-14.zip',
             'limitations':'Different vocabularies/splits are not directly ranked. Epoch-5 five-sign result is retrospective. No live reliability claim.'}
    plan={'status':'planned','schema':'model-retention-plan-v1','root':str(ROOT),'date_local':'2026-11-14',
          'history_root':catalog['history_root'],'catalog':catalog,'files':files,
          'before_bytes':sum(f['bytes'] for f in files),'checkpoint_files_before':sum(f['is_checkpoint'] for f in files),
          'checkpoint_files_retained':len(retained),
          'metadata_files_preserved':sum(f['action']=='preserve_metadata_then_remove' for f in files),
          'removed_checkpoint_policy':'All snapshots not explicitly retained; old bulk/dedicated archives removed and replaced by one curated-only backup.'}
    write_json(target,plan)
    print(json.dumps({k:plan[k] for k in ('before_bytes','checkpoint_files_before','checkpoint_files_retained','metadata_files_preserved')},indent=2))
    for r in retained:print('KEEP',r['checkpoint'],r['evaluation']['top1'])


if __name__=='__main__':main()
