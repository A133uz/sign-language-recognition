"""Validate the unsampled diagnostic adapter on the 37 already-audited clips."""
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.capture_timing_diagnostics import verify_adapter,sample_features,feature_difference,CONFIG
from sign_language.five_sign_pilot import sha256,write_json


def main():
    output=ROOT/'docs/assets/timing-normalization-preflight-2026-11-14.json'
    if output.exists():raise FileExistsError('Preserve prior verification')
    audit=json.loads((ROOT/'docs/assets/pronoun-investigation-2026-11-14.json').read_text())
    protected=dict(audit['protected_sha256'])
    for name in ('src/sign_language/frame_rate.py','src/sign_language/recognition.py',
                 'src/sign_language/recognition_webcam.py','scripts/recognize_webcam.py'):
        protected[name]=sha256(ROOT/name)
    if not all(sha256(ROOT/p)==h for p,h in protected.items()):raise ValueError('Protected baseline differs')
    rows=[];cache=ROOT/'data/training/asl-fifteen-signs-128-v1'
    for row in audit['records']:
        split=row['split'];parent=row['parent_index']
        path=ROOT/f'data/landmarks/asl-citizen-raw-v1-cpu-hands-pose/{split}-{parent//100:05d}.zip'
        with zipfile.ZipFile(path) as archive:
            record=json.loads(archive.read('records.json'))['records'][parent%100]
            if record['video_file']!=row['video_file']:raise ValueError('Raw identity differs')
            with np.load(io.BytesIO(archive.read(record['member'])),allow_pickle=False) as loaded:
                arrays={k:loaded[k] for k in loaded.files}
        before={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}
        full,context=verify_adapter(arrays,record['frame_shape'])
        # Also compare the adapter to actual persisted input bytes.
        index=json.loads((cache/f'{split}-index.json').read_text())
        i=next(i for i,r in enumerate(index) if r['video_file']==row['video_file'])
        stored=np.load(cache/f'{split}-features.npy',mmap_mode='r')
        lengths=np.load(cache/f'{split}-lengths.npy',mmap_mode='r')
        if not np.array_equal(sample_features(full),stored[i,:int(lengths[i])].astype(np.float32)):
            raise AssertionError('Adapter differs from prepared input')
        seen=np.flatnonzero(arrays['hand_present'].any(1));stamps=arrays['timestamp_ms']
        windows={'first_two_seconds':np.flatnonzero(stamps-stamps[0]<2000),
                 'detected_hand_envelope':np.arange(seen[0],seen[-1]+1) if len(seen) else np.array([],int)}
        comparisons={}
        for name,ids in windows.items():
            if not len(ids):comparisons[name]={'status':'empty'};continue
            subset={k:v[ids] for k,v in arrays.items()}
            part,part_context=verify_adapter(subset,record['frame_shape'])
            comparisons[name]={'status':'verified','raw_frames':len(ids),'full_context':context,
                               'window_context':part_context,
                               'feature_differences':feature_difference(sample_features(full[ids]),sample_features(part))}
        if any(hashlib.sha256(v.tobytes()).hexdigest()!=before[k] for k,v in arrays.items()):
            raise AssertionError('Raw observation arrays changed')
        rows.append({'video_file':row['video_file'],'split':split,'gloss':row['gloss'],
                     'full_adapter_equals_production_and_prepared':True,'raw_arrays_unchanged':True,'windows':comparisons})
        print(f'{len(rows)}/37 adapter verified {split} {row["gloss"]} {row["participant_id"]}',flush=True)
    if len(rows)!=37:raise AssertionError('Wrong validation scope')
    unchanged=all(sha256(ROOT/p)==h for p,h in protected.items())
    if not unchanged:raise AssertionError('Protected files changed')
    report={'status':'verified','date_local':'2026-11-14','config':CONFIG,'rows':rows,
            'protected_sha256':protected,'protected_inputs_unchanged':True,
            'checkpoint_sha256':audit['checkpoint_sha256'],'baseline_checkpoint_sha256':audit['baseline_checkpoint_sha256'],
            'training_performed':False,'test_read':False,'new_extraction':False,'webcam_opened':False,
            'limiter_changed':False,'production_preprocessing_changed':False}
    write_json(output,report);print('TIMING_NORMALIZATION_PREFLIGHT_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
