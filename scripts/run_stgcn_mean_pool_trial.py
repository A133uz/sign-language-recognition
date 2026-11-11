"""Fresh five-sign ST-GCN mean-pooling run; the existing ST-GCN already averages.

This is a preserved repeat of the current architecture, not a new readout ablation.
No source model, cache, default checkpoint or earlier weights are overwritten.
"""
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.five_sign_pilot import SIGNS,sha256,write_json
from sign_language.five_sign_training import train_variant,test_selected
from sign_language.preprocessing import CONFIG
from sign_language.training import cuda_check


def main():
    output=ROOT/'models/asl-stgcn27-five-signs-mean-pool-repeat-v1'
    if output.exists():raise FileExistsError('ST-GCN mean-pool repeat exists; preserved unchanged')
    cache=ROOT/'data/training/asl-five-signs-128-v1';manifest=json.loads((cache/'manifest.json').read_text())
    if manifest['labels']!={g:i for i,g in enumerate(SIGNS)} or manifest['identity']['preprocessing']!=CONFIG:
        raise ValueError('Incompatible five-sign cache')
    old=ROOT/'models/asl-stgcn27-five-signs-early-v1';old_config=json.loads((old/'config.json').read_text())
    if old_config['architecture']!='stgcn27-v1' or old_config['cache_id']!=manifest['cache_id']:
        raise ValueError('Historical ST-GCN does not match the requested comparison')
    protected={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'models').rglob('best.pt')}
    inputs={name:sha256(cache/name) for name in manifest['files_sha256']}
    if inputs!=manifest['files_sha256']:raise ValueError('Source cache checksum mismatch')
    gpu=cuda_check();print('GPU',json.dumps(gpu),flush=True)
    print('EXISTING_MEAN_POOL_CONFIRMED: valid-time / all-27-node average; fresh repeat, no architecture change',flush=True)
    spec={'architecture':'stgcn27-v1','experiment_type':'repeat_existing_masked_mean_pool_STGCN',
        'pooling':'sum learned features over valid time and all 27 nodes, divided by valid reduced length times 27',
        'architecture_changed':False,'historical_checkpoint_sha256':sha256(old/'best.pt'),
        'cache_id':manifest['cache_id'],'classes':list(SIGNS),'seed':42,'epochs_cap':33,
        'settings':'Same 10 blocks / 27 XY joints / dropout .5 / AdamW .001 / weight decay .001 / batch 4 / existing augmentation and early selection',
        'protected_weights_before':protected,
        'test_protocol':'One test pass after validation-only selection; same previously reported official test, not a new unseen benchmark'}
    try:
        trained=train_variant(cache,output,'regularized-v1',33,gpu,batch_size=4,architecture='stgcn27-v1')
        config=json.loads((output/'config.json').read_text())
        same_fields=('architecture','model','preprocessing','cache_id','batch_size','epochs','patience',
            'seed','optimizer','learning_rate','weight_decay','profile','regularization','cudnn_enabled','amp',
            'loader_workers','selection','partition','graph')
        if any(config[k]!=old_config[k] for k in same_fields):raise AssertionError('Repeat settings differ from baseline')
        write_json(output/'experiment.json',spec)
        snapshot=output/'source_snapshot';snapshot.mkdir()
        for name,digest in config['source_sha256'].items():
            source=ROOT/'src/sign_language'/name
            if sha256(source)!=digest:raise AssertionError('Training source changed')
            shutil.copy2(source,snapshot/name)
        shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
        tested=test_selected(cache,output)
        weights_ok=all(sha256(ROOT/name)==digest for name,digest in protected.items())
        cache_ok=all(sha256(cache/name)==digest for name,digest in inputs.items())
        if not weights_ok or not cache_ok:raise AssertionError('Protected sources changed')
        report={'status':'complete','specification':spec,'training':trained,'test':tested,
            'previous_test':json.loads((old/'test-report.json').read_text()),
            'previous_weights_unchanged':weights_ok,'cache_unchanged':cache_ok,
            'same_architecture_and_settings':True,'default_checkpoint_changed':False}
        write_json(output/'report.json',report)
        write_json(output/'status.json',{'status':'completed','epoch':tested['epoch'],
                     'epochs_completed':trained['epochs_completed'],'test_evaluated':True})
        print('STGCN_MEAN_POOL_REPEAT_COMPLETE',json.dumps({'epoch':tested['epoch'],'epochs_run':trained['epochs_completed'],
                     'test_top1':tested['top1'],'test_macro_f1':tested['macro_f1'],
                     'parameters':trained['parameters'],'previous_weights_unchanged':weights_ok}),flush=True)
    except BaseException as exc:
        if output.exists():write_json(output/'failure.json',{'error':str(exc),'type':type(exc).__name__})
        raise


if __name__=='__main__':main()
