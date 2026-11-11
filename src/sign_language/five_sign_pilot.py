"""Five short, distinct intended glosses and validation-only epoch selection."""
import hashlib
import json
from pathlib import Path

import numpy as np

SIGNS = ('HELLO', 'NO', 'PLEASE', 'THANKYOU', 'YES')
POLICY = {
    'accuracy_tolerance': 'one validation example below best observed top-1',
    'loss_tolerance': .10,
    'selection': 'eligible minimum absolute clean-train/validation accuracy gap; earliest tie',
    'patience': 3, 'loss_min_delta': .001,
    'sustained_gap_limit': .10, 'sustained_gap_epochs': 3,
}


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def select_epoch(records):
    """Do not reward an untrained tiny-gap epoch; require competitive validation."""
    if not records: raise ValueError('No epochs to select')
    samples=records[0]['val']['samples']
    if samples<1 or any(r['val']['samples']!=samples for r in records):
        raise ValueError('Validation sample count must be positive and fixed')
    best_accuracy=max(r['val']['top1'] for r in records)
    accuracy_floor=best_accuracy-1/samples
    eligible=[r for r in records if r['val']['top1']>=accuracy_floor-1e-12]
    loss_ceiling=min(r['val']['loss'] for r in eligible)+POLICY['loss_tolerance']
    eligible=[r for r in eligible if r['val']['loss']<=loss_ceiling+1e-12]
    chosen=min(eligible,key=lambda r:(round(abs(r['train_clean']['top1']-r['val']['top1']),12),r['epoch']))
    return chosen, {'policy':POLICY,'validation_samples':samples,'accuracy_floor':accuracy_floor,
                    'validation_loss_ceiling':loss_ceiling,'eligible_epochs':[r['epoch'] for r in eligible],
                    'selected_epoch':chosen['epoch'],'selected_accuracy_gap':abs(chosen['train_clean']['top1']-chosen['val']['top1']),
                    'test_used':False}


def early_stop_reason(records, loss_stale, accuracy_stale, warmup_epochs=5):
    """A tiny-gap chance-level run needs learning, not an overfitting stop."""
    if len(records)<warmup_epochs or max(r['train_clean']['top1'] for r in records)<.5:
        return None
    if loss_stale>=POLICY['patience']:
        return 'validation_loss_patience'
    if accuracy_stale>=POLICY['sustained_gap_epochs'] and all(r['accuracy_gap']>POLICY['sustained_gap_limit'] for r in records[-3:]):
        return 'sustained_generalization_gap'
    return None


def prepare_subset(parent, output):
    """Copy selected rows with contiguous target IDs; never modify the full cache."""
    from sign_language.preprocessing import CONFIG
    parent,output=Path(parent),Path(output)
    manifest=json.loads((parent/'manifest.json').read_text())
    if json.loads((parent/'status.json').read_text())['status']!='complete':
        raise ValueError('Parent feature cache is not complete')
    if manifest['identity']['preprocessing']!=CONFIG:raise ValueError('Unexpected preprocessor')
    if sha256(Path(__file__).with_name('preprocessing.py'))!=manifest['identity']['preprocessing_sha256']:
        raise ValueError('Preprocessing source differs')
    labels={g:i for i,g in enumerate(SIGNS)}
    if any(g not in manifest['labels'] for g in SIGNS):raise ValueError('Missing pilot class')
    if output.exists():raise FileExistsError('Use a new subset directory; source caches are never overwritten')
    output.mkdir(parents=True)
    identity={'parent_cache_id':manifest['cache_id'],'preprocessing':CONFIG,
              'preprocessing_sha256':manifest['identity']['preprocessing_sha256'],
              'signs':list(SIGNS),'selection':'all existing rows of the five glosses; official splits unchanged',
              'normalization_changed':False,'landmarks_reextracted':False}
    counts={};split_people={};source_rows={}
    for split in ('train','val','test'):
        index=json.loads((parent/f'{split}-index.json').read_text())
        selected=[i for i,r in enumerate(index) if r['gloss'] in labels]
        source_rows[split]=selected
        arrays={k:np.load(parent/f'{split}-{k}.npy',mmap_mode='r') for k in ('features','lengths','targets')}
        features=np.array(arrays['features'][selected],copy=True)
        lengths=np.array(arrays['lengths'][selected],copy=True)
        targets=np.array([labels[index[i]['gloss']] for i in selected],dtype=np.int64)
        expected_old=np.array([manifest['labels'][index[i]['gloss']] for i in selected])
        if not np.array_equal(arrays['targets'][selected],expected_old):raise ValueError('Parent target/index mismatch')
        if not np.isfinite(features).all() or np.any((lengths<1)|(lengths>128)):
            raise ValueError('Invalid selected features/lengths')
        rows=[{**index[i],'class_id':labels[index[i]['gloss']],
               'parent_class_id':int(expected_old[n]),'parent_index':i} for n,i in enumerate(selected)]
        split_people[split]={r['participant_id'] for r in rows}
        counts[split]={g:sum(r['gloss']==g for r in rows) for g in SIGNS}
        for key,array in [('features',features),('lengths',lengths),('targets',targets)]:np.save(output/f'{split}-{key}.npy',array,allow_pickle=False)
        write_json(output/f'{split}-index.json',rows)
    if any(split_people[a]&split_people[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Signer overlap')
    identity['parent_row_indices']=source_rows
    cache_id=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    files={p.name:sha256(p) for p in output.iterdir() if p.is_file()}
    result={'cache_id':cache_id,'identity':identity,'labels':labels,'counts':counts,
            'participants':{s:sorted(v) for s,v in split_people.items()},'files_sha256':files}
    write_json(output/'manifest.json',result)
    write_json(output/'status.json',{'status':'complete','cache_id':cache_id,'counts':{s:sum(c.values()) for s,c in counts.items()}})
    return result
