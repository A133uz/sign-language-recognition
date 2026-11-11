"""Read-only ASL Citizen video/landmark/cache audit; never extracts or trains.

Writes only to a new audit directory. Resume uses the same input identity and
completed shard reports. Two workers by default, each with one decoder thread.
"""
import argparse
import csv
import hashlib
import io
import json
import multiprocessing
import os
import random
import sys
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sign_language.preprocessing import normalize_sequence

RAW = ROOT / 'data/landmarks/asl-citizen-raw-v1-cpu-hands-pose'
CACHE = ROOT / 'data/training/asl-gru-128-v1'
DATASET = ROOT / 'data/asl_citizen/ASL_Citizen'
FOCUS = ('HELLO', 'THANKYOU', 'NO', 'ME', 'YES')
PROTECTED = [ROOT / 'models/asl-stgcn27-v1/best.pt',
             ROOT / 'models/asl-gru-128-cv-regularized-v1/official-refit/best.pt']
SCHEMA = {
    'hands_xyz': ('float32', (2,21,3)), 'hand_present': ('bool', (2,)),
    'handedness': ('int8', (2,)), 'handedness_score': ('float32', (2,)),
    'pose_xyz': ('float32', (33,3)), 'pose_visibility': ('float32', (33,)),
    'pose_presence': ('float32', (33,)), 'pose_detected': ('bool', ()),
    'pose_valid_mask': ('bool', (33,)), 'timestamp_ms': ('int64', ()),
    'frame_index': ('int32', ()),
}
STATE = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def validate_arrays(a, record):
    """Strict format/semantic checks; out-of-image coordinates are not corruption."""
    errors = []
    t = record['frames']
    if set(a) != set(SCHEMA):
        errors.append('array_keys')
    for key, (dtype, tail) in SCHEMA.items():
        if key not in a or a[key].shape != (t, *tail) or a[key].dtype != np.dtype(dtype):
            errors.append('shape_or_dtype:' + key)
        elif not np.isfinite(a[key]).all():
            errors.append('nonfinite:' + key)
    if errors:
        return errors
    if not np.array_equal(a['frame_index'], np.arange(t)):
        errors.append('frame_index')
    times = []; fps = record['fps']
    for i in range(t):
        times.append(max(times[-1] + 1 if times else 0, round(i * 1000 / fps)))
    if not np.array_equal(a['timestamp_ms'], times):
        errors.append('timestamps')
    if not np.array_equal(a['pose_valid_mask'], (a['pose_visibility'] >= .5) & (a['pose_presence'] >= .5)):
        errors.append('pose_mask')
    hp = a['hand_present']; pd = a['pose_detected']
    if np.any(a['hands_xyz'][~hp] != 0) or np.any(a['handedness'][~hp] != -1) or np.any(a['handedness_score'][~hp] != 0):
        errors.append('absent_hand_fill')
    if np.any(~np.isin(a['handedness'][hp], [0,1])):
        errors.append('present_hand_label')
    if any(np.any(a[k][~pd] != 0) for k in ('pose_xyz','pose_visibility','pose_presence')):
        errors.append('absent_pose_fill')
    for key in ('handedness_score','pose_visibility','pose_presence'):
        if np.any((a[key] < 0) | (a[key] > 1)):
            errors.append('confidence_range:' + key)
    counts = {'frames_with_hands': int(hp.any(axis=1).sum()),
              'frames_with_pose': int(pd.sum()), 'hand_detections': int(hp.sum()),
              'pose_valid_points': int(a['pose_valid_mask'].sum())}
    errors.extend('record_count:' + k for k,v in counts.items() if record.get(k) != v)
    return errors


def initialize(config):
    global STATE
    cv2.setNumThreads(1)
    STATE = config
    STATE['arrays'] = {s: tuple(np.load(CACHE / f'{s}-{k}.npy', mmap_mode='r')
                               for k in ('features','lengths','targets')) for s in ('train','val','test')}


def decode_video(path, record):
    cap = cv2.VideoCapture(str(path)); errors = []
    n = 0; black = 0; same = 0; previous = None; blur = []
    fps = float(cap.get(cv2.CAP_PROP_FPS)); declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened(): errors.append('decode_open')
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            n += 1
            if list(frame.shape) != record['frame_shape']:
                errors.append('decode_shape'); break
            thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160,120))
            black += int(thumb.mean() < 3 and thumb.std() < 3)
            same += int(previous is not None and np.array_equal(thumb, previous))
            blur.append(float(cv2.Laplacian(thumb, cv2.CV_64F).var()))
            previous = thumb
    finally:
        cap.release()
    if n != record['frames'] or declared != record['declared_frames']: errors.append('decode_count')
    if not np.isfinite(fps) or not np.isclose(fps, record['fps'], rtol=1e-6): errors.append('decode_fps')
    return {'decoded_frames': n, 'black_frame_fraction': black/max(n,1),
            'identical_thumbnail_fraction': same/max(n-1,1),
            'median_thumbnail_laplacian': float(np.median(blur)) if blur else None,
            'errors': errors}


def audit_shard(job):
    name, expected = job
    begin = time.monotonic(); errors = []; clips = []
    split = expected[0]['split']; start = int(name.rsplit('-',1)[1])*100
    archive_path = RAW / f'{name}.zip'
    marker = json.loads((RAW / f'{name}.complete.json').read_text())
    if marker['run_id'] != STATE['run_id'] or marker['rows_sha256'] != fingerprint(expected): errors.append('raw_marker_identity')
    if marker['bytes'] != archive_path.stat().st_size or marker['sha256'] != digest(archive_path): errors.append('raw_archive_hash')
    features, lengths, targets = STATE['arrays'][split]
    prep = json.loads((CACHE / f'{name}.prepared.json').read_text())
    cache_hash = hashlib.sha256(features[start:start+len(expected)].tobytes()+lengths[start:start+len(expected)].tobytes()+targets[start:start+len(expected)].tobytes()).hexdigest()
    if (prep['start'] != start or prep['count'] != len(expected) or prep['cache_id'] != STATE['cache_id'] or prep['sha256'] != cache_hash): errors.append('prepared_marker')
    with zipfile.ZipFile(archive_path) as archive:
        info = json.loads(archive.read('records.json')); records = info['records']
        if info['run_id'] != STATE['run_id'] or info['rows_sha256'] != fingerprint(expected): errors.append('raw_archive_identity')
        if len(records) != len(expected): raise ValueError('Wrong record count: ' + name)
        if len(archive.namelist()) != len(set(archive.namelist())): errors.append('duplicate_zip_members')
        for offset, (record, row) in enumerate(zip(records, expected, strict=True)):
            item = dict(row); item['shard'] = name; item['offset'] = offset; item['errors'] = []
            try:
                if record['status'] != 'ok' or any(record.get(k) != v for k,v in row.items()): item['errors'].append('raw_row_identity')
                with np.load(io.BytesIO(archive.read(record['member'])), allow_pickle=False) as loaded:
                    arrays = {k: loaded[k] for k in loaded.files}
                item['errors'].extend(validate_arrays(arrays, record))
                sequence, quality = normalize_sequence(arrays, record['frame_shape'])
                padded = np.zeros((128,254), np.float16); padded[:len(sequence)] = sequence
                i = start + offset
                if not np.array_equal(features[i], padded): item['errors'].append('recomputed_features_differ')
                if int(lengths[i]) != len(sequence): item['errors'].append('prepared_length')
                if int(targets[i]) != row['class_id']: item['errors'].append('prepared_target')
                hp = arrays['hand_present'].any(axis=1); ids = np.flatnonzero(hp)
                internal = int(np.sum(~hp[ids[0]:ids[-1]+1])) if len(ids) else 0
                hxy = arrays['hands_xyz'][arrays['hand_present']][...,:2]
                scale = np.array([record['frame_shape'][1],record['frame_shape'][0]])
                palm = np.linalg.norm((hxy[:,5]-hxy[:,17])*scale, axis=-1)
                item.update(frames=record['frames'], fps=record['fps'], frame_shape=record['frame_shape'],
                            hands_frames=int(hp.sum()), pose_frames=record['frames_with_pose'],
                            hand_coverage=float(hp.mean()), internal_missing_hand_frames=internal,
                            anchor_coverage=quality['frames_with_body_anchor']/record['frames'],
                            palm_pixels_median=float(np.median(palm)) if len(palm) else None,
                            hand_xy_outside_fraction=float(np.mean(np.any((hxy < 0)|(hxy > 1), axis=-1))) if len(hxy) else None,
                            feature_clip_fraction=float(np.mean(np.abs(sequence) >= 8)),
                            hand_quarter_coverage=[float(hp[s].mean()) for s in np.array_split(np.arange(len(hp)),4) if len(s)])
                video = DATASET / 'videos' / row['video_file']
                original = STATE['original'][row['video_file']]
                h = hashlib.sha256(); crc = 0; size = 0
                with video.open('rb') as f:
                    for chunk in iter(lambda: f.read(1024*1024), b''):
                        h.update(chunk); crc = zlib.crc32(chunk,crc); size += len(chunk)
                item.update(video_sha256=h.hexdigest(), video_bytes=size)
                if size != record['source_bytes'] or size != original['bytes']: item['errors'].append('video_size')
                if crc != original['crc']: item['errors'].append('video_archive_crc')
                if row['video_file'] in STATE['decode_sample']:
                    item['decode'] = decode_video(video, record)
                    item['errors'].extend(item['decode']['errors'])
            except Exception as exc:
                item['errors'].append(type(exc).__name__ + ': ' + str(exc))
            clips.append(item)
    result = {'identity': STATE['identity'], 'shard': name, 'errors': errors,
              'clips': clips, 'elapsed_seconds': time.monotonic()-begin}
    write(Path(STATE['output']) / 'parts' / f'{name}.json', result)
    return name, len(clips), sum(bool(c['errors']) for c in clips), errors


def summarize(output, identity, metadata):
    clips = []; shard_errors = []
    for p in sorted((output / 'parts').glob('*.json')):
        d = json.loads(p.read_text())
        if d['identity'] != identity: raise ValueError('Audit part identity changed')
        clips.extend(d['clips']); shard_errors.extend({'shard': d['shard'], 'error': e} for e in d['errors'])
    errors = [c for c in clips if c['errors']]
    duplicates = defaultdict(list)
    for c in clips:
        if 'video_sha256' in c: duplicates[c['video_sha256']].append(c)
    groups = [[{k:c[k] for k in ('video_file','split','gloss','participant_id')} for c in g]
              for g in duplicates.values() if len(g)>1]
    quality = {}
    for key, selected in [('all',clips), *[(s,[c for c in clips if c['split']==s]) for s in ('train','val','test')],
                          *[(g,[c for c in clips if c['gloss']==g]) for g in FOCUS]]:
        valid = [c for c in selected if 'hand_coverage' in c]
        quality[key] = {'clips':len(selected), 'zero_hand_clips':sum(c['hands_frames']==0 for c in valid),
                        'under_10pct_hand_coverage':sum(c['hand_coverage']<.1 for c in valid),
                        'under_90pct_anchor_coverage':sum(c['anchor_coverage']<.9 for c in valid),
                        'any_internal_missing_hands':sum(c['internal_missing_hand_frames']>0 for c in valid),
                        'any_feature_clipping':sum(c['feature_clip_fraction']>0 for c in valid)}
        for metric in ('hand_coverage','anchor_coverage','palm_pixels_median','feature_clip_fraction'):
            values = [c[metric] for c in valid if c[metric] is not None]
            quality[key][metric + '_quantiles'] = np.quantile(values,[0,.05,.25,.5,.75,.95,1]).tolist() if values else []
    before = metadata['checkpoint_hashes_before']
    after = {str(p.relative_to(ROOT)):digest(p) for p in PROTECTED}
    decoded = [c for c in clips if 'decode' in c]
    result = {**metadata, 'status':'complete', 'completed_shards':len(list((output/'parts').glob('*.json'))),
              'clips':len(clips), 'video_bytes_hashed':sum(c.get('video_bytes',0) for c in clips),
              'raw_frames_checked':sum(c.get('frames',0) for c in clips),
              'clips_with_errors':len(errors), 'shard_errors':shard_errors,
              'error_counts':dict(Counter(e for c in clips for e in c['errors'])),
              'exact_duplicate_groups':groups,
              'cross_split_duplicate_groups':sum(len({c['split'] for c in g})>1 for g in groups),
              'decoded_sample_clips':len(decoded), 'decoded_sample_frames':sum(c['decode']['decoded_frames'] for c in decoded),
              'decoded_sample_errors':sum(bool(c['decode']['errors']) for c in decoded),
              'quality':quality, 'checkpoint_hashes_after':after, 'checkpoint_unchanged':before==after,
              'finished_utc':datetime.now(timezone.utc).isoformat()}
    write(output/'summary.json',result); write(output/'issues.json',errors)
    write(output/'decode-sample.json',decoded)
    columns = ['video_file','split','gloss','participant_id','shard','offset','frames','hand_coverage',
               'anchor_coverage','palm_pixels_median','internal_missing_hand_frames','feature_clip_fraction',
               'video_bytes','video_sha256','errors']
    with (output/'per-clip.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(clips)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    print('Validating CSV and prepared-cache identities...',flush=True)
    rows=[]; by_split={}; hashes={}
    for split in ('train','val','test'):
        path=DATASET/'splits'/f'{split}.csv';hashes[split]=digest(path)
        with path.open(newline='',encoding='utf-8-sig') as f:
            by_split[split]=[{'video_file':r['Video file'].strip(),'participant_id':r['Participant ID'].strip(),
                              'gloss':r['Gloss'].strip(),'split':split} for r in csv.DictReader(f)]
        rows.extend(by_split[split])
    labels={g:i for i,g in enumerate(sorted({r['gloss'] for r in rows if r['split']=='train'}))}
    if len({r['video_file'] for r in rows})!=len(rows): raise ValueError('Duplicate CSV filenames')
    if {s:len(r) for s,r in by_split.items()}!={'train':40154,'val':10304,'test':32941}: raise ValueError('CSV counts')
    if len(labels)!=2731: raise ValueError('Class count')
    people={s:{r['participant_id'] for r in rs} for s,rs in by_split.items()}
    if any(people[a]&people[b] for a,b in [('train','val'),('train','test'),('val','test')]): raise ValueError('Signer overlap')
    for r in rows: r['class_id']=labels[r['gloss']]
    raw=json.loads((RAW/'run-manifest.json').read_text());cache=json.loads((CACHE/'manifest.json').read_text())
    if hashes!=raw['identity']['csv_sha256'] or labels!=json.loads((RAW/'labels.json').read_text()): raise ValueError('Raw metadata differs')
    if digest(ROOT/'src/sign_language/preprocessing.py')!=cache['identity']['preprocessing_sha256']: raise ValueError('Preprocessor differs')
    plan=[]
    for split,rs in by_split.items():
        selected=sorted(rs,key=lambda r:r['video_file']);random.Random(0).shuffle(selected)
        if selected!=json.loads((CACHE/f'{split}-index.json').read_text()): raise ValueError('Prepared index differs')
        for suffix,shape,dtype in [('features',(len(rs),128,254),'float16'),('lengths',(len(rs),),'int16'),('targets',(len(rs),),'int64')]:
            arr=np.load(CACHE/f'{split}-{suffix}.npy',mmap_mode='r')
            if arr.shape!=shape or arr.dtype!=np.dtype(dtype): raise ValueError('Prepared array header')
        plan.extend((f'{split}-{i//100:05d}',selected[i:i+100]) for i in range(0,len(selected),100))
    print('Reading original archive directory and video inventory...',flush=True)
    original={}
    with zipfile.ZipFile(ROOT/'data/downloads/ASL_Citizen.zip') as z:
        for member in z.infolist():
            if member.filename.startswith('ASL_Citizen/videos/') and not member.is_dir():
                name=Path(member.filename).name
                if name in original: raise ValueError('Duplicate original archive filename')
                original[name]={'crc':member.CRC,'bytes':member.file_size}
    with os.scandir(DATASET/'videos') as entries:
        actual={entry.name for entry in entries if entry.is_file()}
    expected={r['video_file'] for r in rows}
    if expected-set(original) or expected-actual: raise ValueError('Missing source/archive videos')
    sample={r['video_file'] for r in rows if r['split']!='test' and r['gloss'] in FOCUS}
    rng=random.Random(42)
    for person in sorted({r['participant_id'] for r in rows}):
        selected=sorted((r['video_file'] for r in rows if r['participant_id']==person))
        sample.update(rng.sample(selected,min(4,len(selected))))
    identity=fingerprint({'run_id':raw['run_id'],'cache_id':cache['cache_id'],'csv_sha256':hashes,
                          'audit_source_sha256':digest(Path(__file__)),'decode_sample':sorted(sample)})
    meta_path=output/'audit-manifest.json'
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta['identity']!=identity: raise ValueError('Choose new output: audit identity changed')
    else:
        meta={'identity':identity,'started_utc':datetime.now(timezone.utc).isoformat(),'workers':args.workers,
              'run_id':raw['run_id'],'cache_id':cache['cache_id'],'csv_sha256':hashes,
              'audit_source_sha256':digest(Path(__file__)),'counts':{s:len(r) for s,r in by_split.items()},
              'participants':{s:len(v) for s,v in people.items()},'classes':len(labels),
              'extra_video_files':sorted(actual-expected),'decode_sample_requested':len(sample),
              'checkpoint_hashes_before':{str(p.relative_to(ROOT)):digest(p) for p in PROTECTED},
              'scope':'All video SHA256/CRC, all NPZ arrays and prepared values; sampled full video decode; no training or data changes.'}
        write(meta_path,meta)
    config={'run_id':raw['run_id'],'cache_id':cache['cache_id'],'identity':identity,
            'output':str(output),'original':original,'decode_sample':sample}
    pending=[job for job in plan if not (output/'parts'/f'{job[0]}.json').exists()]
    began=time.monotonic();completed=len(plan)-len(pending)
    print(f'Audit: {len(rows)} clips, {len(plan)} shards, {len(sample)} complete-decode samples; {args.workers} workers; {completed} reusable reports.',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(config,)) as pool:
        for batch_start in range(0,len(pending),args.workers*2):
            batch=pending[batch_start:batch_start+args.workers*2]
            for name,n,failures,errors in pool.map(audit_shard,batch):
                completed+=1
                write(output/'status.json',{'status':'auditing','completed_shards':completed,'total_shards':len(plan),
                                          'elapsed_seconds':time.monotonic()-began,'last_shard':name})
                if completed%10==0 or failures or errors or completed==len(plan):
                    print(f'{completed}/{len(plan)} {name}: {n} clips, {failures} clip errors, shard errors={errors}; elapsed {time.monotonic()-began:.1f}s',flush=True)
    result=summarize(output,identity,meta)
    write(output/'status.json',{'status':'complete','completed_shards':completed,'total_shards':len(plan),'elapsed_seconds':time.monotonic()-began})
    print(json.dumps({k:result[k] for k in ('clips','clips_with_errors','decoded_sample_clips','decoded_sample_errors','cross_split_duplicate_groups','checkpoint_unchanged')},indent=2),flush=True)


if __name__=='__main__':
    main()
