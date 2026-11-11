"""Five-sign ST-GCN comparison with validation-only early selection, no cleaning."""
import json
from pathlib import Path
import shutil

from sign_language.five_sign_pilot import POLICY, SIGNS, sha256, write_json
from sign_language.five_sign_training import train_variant, test_selected
from sign_language.preprocessing import CONFIG
from sign_language.training import cuda_check


def run(root):
    root=Path(root)
    cache=root/'data/training/asl-five-signs-128-v1'
    output=root/'models/asl-stgcn27-five-signs-early-v1'
    if output.exists():raise FileExistsError('ST-GCN pilot output exists; use a new run directory')
    subset=json.loads((cache/'manifest.json').read_text())
    status=json.loads((cache/'status.json').read_text())
    parent=json.loads((root/'data/training/asl-gru-128-v1/manifest.json').read_text())
    if (status['status']!='complete' or status['cache_id']!=subset['cache_id'] or
        subset['identity']['parent_cache_id']!=parent['cache_id'] or
        subset['identity']['preprocessing']!=CONFIG or
        subset['identity']['preprocessing_sha256']!=sha256(Path(__file__).with_name('preprocessing.py')) or
        subset['labels']!={g:i for i,g in enumerate(SIGNS)}):
        raise ValueError('Incompatible five-sign cache')
    before_cache={name:sha256(cache/name) for name in subset['files_sha256']}
    if before_cache!=subset['files_sha256']:raise ValueError('Subset payload changed')
    protected=[root/'models'/name/'best.pt' for name in
               ('asl-stgcn27-v1','asl-gru-128-baseline','asl-gru-128-cv-regularized-v1/official-refit',
                'asl-gru-five-signs-early-v2/baseline','asl-gru-five-signs-early-v2/regularized')]
    before={str(p.relative_to(root)):sha256(p) for p in protected}
    gpu=cuda_check();print(json.dumps(gpu),flush=True)
    training=train_variant(cache,output,'regularized-v1',33,gpu,batch_size=4,architecture='stgcn27-v1')
    # Selection is committed before accessing test labels or predictions.
    test=test_selected(cache,output)
    after={str(p.relative_to(root)):sha256(p) for p in protected}
    after_cache={name:sha256(cache/name) for name in subset['files_sha256']}
    snapshot=output/'engine_snapshot/src/sign_language';snapshot.mkdir(parents=True)
    config=json.loads((output/'config.json').read_text())
    for name,wanted in config['source_sha256'].items():
        source=Path(__file__).with_name(name)
        if sha256(source)!=wanted:raise ValueError('Training source changed during the run')
        shutil.copy2(source,snapshot/name)
    shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    report={'status':'complete','architecture':'stgcn27-v1','classes':list(SIGNS),
            'counts':subset['counts'],'participants':subset['participants'],'policy':POLICY,
            'training':training,'test':test,'gpu':gpu,
            'protected_weights_before':before,'protected_weights_after':after,
            'previous_weights_unchanged':before==after,'cache_unchanged':before_cache==after_cache,
            'note':'Same five-class subset and gap policy as the GRU pilot. Different graph representation/backbone. No cleaning or new extraction.'}
    write_json(output/'report.json',report)
    state=json.loads((output/'status.json').read_text());state.update(status='completed',test_evaluated=True)
    write_json(output/'status.json',state)
    print(json.dumps({'selected_epoch':training['selection']['selected_epoch'],
                      'epochs_run':training['epochs_completed'],'test_top1':test['top1'],
                      'old_weights_unchanged':before==after,'cache_unchanged':before_cache==after_cache},indent=2),flush=True)
    return report
