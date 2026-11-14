"""Extract frame-aligned MediaPipe face landmarks for the existing 15-sign subset."""
import argparse
import hashlib
import io
import json
import math
import multiprocessing
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import torch  # Load before OpenCV/MediaPipe on this Windows installation.
import cv2
import mediapipe as mp
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.expanded_pilot import SIGNS
from sign_language.dataset import write_json

RAW=ROOT/'data/landmarks/asl-citizen-raw-v1-cpu-hands-pose'
PARENT=ROOT/'data/training/asl-gru-128-v1'
SUBSET=ROOT/'data/training/asl-fifteen-signs-128-v1'
VIDEOS=ROOT/'data/asl_citizen/ASL_Citizen/videos'
MODEL=ROOT/'models/face_landmarker.task'
OUTPUT=ROOT/'data/landmarks/asl-citizen-face-fifteen-v1'
MODEL_SHA='64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def select_rows():
    manifest=json.loads((SUBSET/'manifest.json').read_text(encoding='utf-8'))
    parent=json.loads((PARENT/'manifest.json').read_text(encoding='utf-8'))
    if list(manifest['labels'])!=list(SIGNS) or manifest['identity']['parent_cache_id']!=parent['cache_id']:
        raise ValueError('15-sign label/cache mismatch')
    rows=[]
    for split in ('train','val','test'):
        all_rows=json.loads((PARENT/f'{split}-index.json').read_text(encoding='utf-8'))
        selection=[r for r in all_rows if r['gloss'] in SIGNS]
        if split!='test':
            reference=json.loads((SUBSET/f'{split}-index.json').read_text(encoding='utf-8'))
            if [r['video_file'] for r in selection]!=[r['video_file'] for r in reference]:raise ValueError('Subset row mismatch')
        rows.extend({k:r[k] for k in ('video_file','participant_id','gloss','split')} for r in selection)
    if len(rows)!=461 or len({r['video_file'] for r in rows})!=len(rows):raise ValueError('Unexpected subset size or duplicate')
    people={s:{r['participant_id'] for r in rows if r['split']==s} for s in ('train','val','test')}
    if any(people[a]&people[b] for a,b in (('train','val'),('train','test'),('val','test'))):raise ValueError('Signer leakage')
    return rows,manifest,parent


def raw_lookup(rows):
    needed={r['video_file'] for r in rows};result={};files_read=0
    for archive in sorted(RAW.glob('*.zip')):
        with zipfile.ZipFile(archive) as z:
            data=json.loads(z.read('records.json'));selected=[r for r in data['records'] if r['video_file'] in needed]
            if not selected:continue
            marker=json.loads(archive.with_suffix('.complete.json').read_text(encoding='utf-8'))
            if archive.stat().st_size!=marker['bytes'] or sha(archive)!=marker['sha256']:raise ValueError('Raw archive hash mismatch: '+archive.name)
            files_read+=1
            for record in selected:
                if record['status']!='ok' or record['video_file'] in result:raise ValueError('Invalid raw record')
                payload=z.read(record['member'])
                with np.load(io.BytesIO(payload),allow_pickle=False) as a:
                    indexes=a['frame_index'];timestamps=a['timestamp_ms']
                if len(indexes)!=record['frames'] or not np.array_equal(indexes,np.arange(len(indexes))):raise ValueError('Invalid raw index')
                result[record['video_file']]={'archive':archive.name,'archive_sha256':marker['sha256'],'raw_member':record['member'],
                    'raw_member_sha256':hashlib.sha256(payload).hexdigest(),'frames':record['frames'],'fps':record['fps'],
                    'frame_shape':record['frame_shape'],'source_bytes':record['source_bytes'],'timestamps':timestamps.tolist()}
                needed.remove(record['video_file'])
        if not needed:break
    if needed:raise ValueError(f'Missing {len(needed)} original raw records')
    print('RAW_ALIGNMENT_INPUTS_VERIFIED',len(result),'clips from',files_read,'verified shards',flush=True)
    return result


def extract_one(job):
    row,expected,output=job;video=VIDEOS/row['video_file'];output=Path(output)
    video_hash=sha(video)
    options=mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL),delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,num_faces=1,output_face_blendshapes=True,
        min_face_detection_confidence=.5,min_face_presence_confidence=.5,min_tracking_confidence=.5)
    path=output/row['split']/(Path(row['video_file']).stem+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
    capture=cv2.VideoCapture(str(video));frames=[];present=[];blends=[];timestamps=[];names=None;shape=None
    started=time.monotonic()
    try:
        if not capture.isOpened():raise ValueError('Video cannot be opened')
        fps=float(capture.get(cv2.CAP_PROP_FPS));declared=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps<=0 or abs(fps-expected['fps'])>1e-6:raise ValueError('FPS mismatch')
        if video.stat().st_size!=expected['source_bytes']:raise ValueError('Video size differs from raw extraction')
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
            while True:
                ok,frame=capture.read()
                if not ok:break
                if len(frames)>=3600:raise ValueError('Exceeded max frames')
                if shape is not None and tuple(frame.shape)!=shape:raise ValueError('Frame shape changed')
                shape=frame.shape;timestamp=max(timestamps[-1]+1 if timestamps else 0,round(len(frames)*1000/fps))
                rgb=np.ascontiguousarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),timestamp)
                points=np.zeros((478,3),np.float32);blend=np.zeros(52,np.float32);found=bool(result.face_landmarks)
                if found:
                    if len(result.face_landmarks)!=1 or len(result.face_landmarks[0])!=478:raise ValueError('Unexpected face landmark count')
                    points[:]=[(v.x,v.y,v.z) for v in result.face_landmarks[0]]
                    if len(result.face_blendshapes)!=1 or len(result.face_blendshapes[0])!=52:raise ValueError('Unexpected blendshape count')
                    categories=result.face_blendshapes[0];current=[v.category_name for v in categories]
                    if names is None:names=current
                    if current!=names:raise ValueError('Blendshape category order changed')
                    blend[:]=[v.score for v in categories]
                if not np.isfinite(points).all() or not np.isfinite(blend).all():raise ValueError('Nonfinite face features')
                frames.append(points);present.append(found);blends.append(blend);timestamps.append(timestamp)
        if len(frames)!=expected['frames'] or (declared>0 and len(frames)!=declared):raise ValueError('Frame count mismatch')
        if list(shape)!=expected['frame_shape'] or timestamps!=expected['timestamps']:raise ValueError('Frame/timestamp alignment mismatch')
        if sha(video)!=video_hash:raise ValueError('Source video changed during extraction')
        temp=path.with_suffix('.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,face_xyz=np.stack(frames),face_present=np.asarray(present,dtype=bool),
              face_blendshapes=np.stack(blends),frame_index=np.arange(len(frames),dtype=np.int32),timestamp_ms=np.asarray(timestamps,dtype=np.int64))
        os.replace(temp,path)
        result={**row,'status':'ok','features':str(path.relative_to(output)),'sha256':sha(path),'frames':len(frames),
                'face_present_frames':int(sum(present)),'fps':fps,'frame_shape':list(shape),'raw_archive':expected['archive'],
                'raw_archive_sha256':expected['archive_sha256'],'raw_member':expected['raw_member'],
                'raw_member_sha256':expected['raw_member_sha256'],'source_bytes':video.stat().st_size,
                'source_video_sha256':video_hash,'blendshape_names':names,'seconds':time.monotonic()-started}
        write_json(path.with_suffix('.json'),result)
        return result
    finally:capture.release()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=2);parser.add_argument('--limit',type=int)
    args=parser.parse_args();output=args.output.resolve();allowed=(ROOT/'data/landmarks').resolve();cache=(ROOT/'.cache').resolve()
    if not (output.is_relative_to(allowed) or output.is_relative_to(cache)):raise ValueError('Output must be under local landmarks or cache')
    if not 1<=args.workers<=4:raise ValueError('Workers must be between 1 and 4')
    if sha(MODEL)!=MODEL_SHA:raise ValueError('Unexpected face model hash')
    rows,subset,parent=select_rows()
    if args.limit:rows=rows[:args.limit]
    expected=raw_lookup(rows)
    output.mkdir(parents=True,exist_ok=True)
    run_identity={'schema':'asl-citizen-face-fifteen-v1','labels':list(SIGNS),'subset_cache_id':subset['cache_id'],
        'parent_cache_id':parent['cache_id'],'model_sha256':MODEL_SHA,'model_url':'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        'mediapipe':mp.__version__,'delegate':'CPU','workers':args.workers,'frame_selection':'all decoded original frames; aligned to existing raw frame index and timestamps',
        'landmarks':'478 face image-normalized xyz per frame','blendshapes':'52 model scores per detected face','missing':'zero-filled arrays plus face_present mask',
        'videos':len(rows),'video_filenames':[r['video_file'] for r in rows],'video_hash_scope':'SHA-256 computed at face extraction; raw extraction records source size but not video hash'}
    identity_path=output/'run-manifest.json'
    if identity_path.exists():
        old=json.loads(identity_path.read_text(encoding='utf-8'));old.pop('workers',None);current=dict(run_identity);current.pop('workers')
        if old!=current:raise ValueError('Existing run identity differs')
    else:write_json(identity_path,run_identity)
    jobs=[];results=[]
    for row in rows:
        path=output/row['split']/(Path(row['video_file']).stem+'.npz');metadata=path.with_suffix('.json')
        if path.exists() and metadata.exists():
            saved=json.loads(metadata.read_text(encoding='utf-8'))
            if saved['sha256']!=sha(path) or saved['raw_member_sha256']!=expected[row['video_file']]['raw_member_sha256'] or saved['source_video_sha256']!=sha(VIDEOS/row['video_file']):
                raise ValueError('Existing completed clip differs; refusing overwrite')
            results.append(saved)
        else:jobs.append((row,expected[row['video_file']],str(output)))
    started=time.monotonic();failures=[]
    print('EXTRACTION_START',json.dumps({'total':len(rows),'completed':len(results),'pending':len(jobs),'workers':args.workers}),flush=True)
    write_json(output/'status.json',{'status':'running','total':len(rows),'completed':len(results),'failed':0})
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=context) as executor:
        futures={executor.submit(extract_one,j):j[0] for j in jobs}
        for future in as_completed(futures):
            row=futures[future]
            try:result=future.result();results.append(result)
            except Exception as e:failures.append({**row,'error':f'{type(e).__name__}: {e}'})
            count=len(results)+len(failures)
            write_json(output/'status.json',{'status':'running','total':len(rows),'completed':len(results),'failed':len(failures),'last_video':row['video_file']})
            if count%10==0 or count==len(rows):print('PROGRESS',count,'/',len(rows),'failed',len(failures),'elapsed',round(time.monotonic()-started,1),flush=True)
    named=[r['blendshape_names'] for r in results if r['blendshape_names']]
    if named and any(n!=named[0] for n in named):raise ValueError('Blendshape labels differ across clips')
    final={'status':'complete' if not failures and len(results)==len(rows) else 'failed','clips':len(results),'expected_clips':len(rows),'failures':failures,
        'frames':sum(r['frames'] for r in results),'face_present_frames':sum(r['face_present_frames'] for r in results),
        'clips_without_detected_face':sum(r['face_present_frames']==0 for r in results),'blendshape_names':named[0] if named else None,
        'newly_processed':len(jobs),'session_seconds':time.monotonic()-started,'records':sorted(results,key=lambda r:(r['split'],r['video_file']))}
    write_json(output/'summary.json',final);write_json(output/'status.json',{'status':final['status'],'total':len(rows),'completed':len(results),'failed':len(failures)})
    print('EXTRACTION_FINISHED',json.dumps({k:v for k,v in final.items() if k not in ('records','blendshape_names')}),flush=True)
    if final['status']!='complete':raise SystemExit(1)

if __name__=='__main__':multiprocessing.freeze_support();main()
