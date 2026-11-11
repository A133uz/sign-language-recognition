"""Train one parameter-matched pooled GRU, preserving all original models/features."""
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.five_sign_pilot import sha256,write_json
from sign_language.five_sign_training import train_variant,test_selected
from sign_language.training import cuda_check


def main():
    output=ROOT/'models/asl-gru-five-signs-mean-pool-v1'
    if output.exists():raise FileExistsError('Mean-pool experiment exists; use a new version, never overwrite')
    cache=ROOT/'data/training/asl-five-signs-128-v1';manifest=json.loads((cache/'manifest.json').read_text())
    baseline=ROOT/'models/asl-gru-five-signs-early-v2/regularized/best.pt'
    protected={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'models').rglob('best.pt')}
    before={name:sha256(cache/name) for name in manifest['files_sha256']}
    if before!=manifest['files_sha256']:raise ValueError('Source cache changed')
    gpu=cuda_check();print('GPU',json.dumps(gpu),flush=True)
    specification={'architecture':'gru-mean-pool-v1','change':'Final hidden state replaced by masked mean of time outputs only',
        'baseline_checkpoint_sha256':sha256(baseline),'baseline_epoch':18,'cache_id':manifest['cache_id'],
        'unchanged':['two layers','256 units','254 features','5 classes','dropout .5','AdamW .001',
                     'weight decay .001','batch 4','seed 42','augmentation','28-epoch cap','early stopping/selection policy'],
        'no_temporal_augmentation_or_depth_changes':True,'initialized_from_scratch':True,
        'test_protocol':'One evaluation after validation-only selection; same official test as historical baseline, not a fresh unseen benchmark.',
        'protected_weights_before':protected}
    try:
        trained=train_variant(cache,output,'regularized-v1',28,gpu,architecture='gru-mean-pool-v1')
        write_json(output/'experiment.json',specification)
        config=json.loads((output/'config.json').read_text())
        snapshot=output/'source_snapshot';snapshot.mkdir()
        for name in config['source_sha256']:shutil.copyfile(ROOT/'src/sign_language'/name,snapshot/name)
        tested=test_selected(cache,output)
        after={name:sha256(ROOT/name) for name in protected}
        sources_ok=before=={name:sha256(cache/name) for name in before}
        if protected!=after or not sources_ok:raise AssertionError('Protected source mutation')
        report={'status':'complete','specification':specification,'training':trained,'test':tested,
                'previous_weights_unchanged':protected==after,'source_cache_unchanged':sources_ok,
                'historical_baseline_test':json.loads((baseline.parent/'test-report.json').read_text()),
                'default_checkpoint_changed':False}
        write_json(output/'report.json',report)
        write_json(output/'status.json',{'status':'completed','epoch':trained['selection']['selected_epoch'],
                     'epochs_completed':trained['epochs_completed'],'test_evaluated':True})
        print('MEAN_POOL_COMPLETE',json.dumps({'epoch':tested['epoch'],'test_top1':tested['top1'],
                          'test_macro_f1':tested['macro_f1'],'unchanged_sources':sources_ok}),flush=True)
    except BaseException as exc:
        if output.exists():write_json(output/'failure.json',{'error':str(exc),'type':type(exc).__name__})
        raise


if __name__=='__main__':main()
