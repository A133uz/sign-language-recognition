"""Bounded read-only PLEASE/MY source-video and wrist-trajectory investigation."""
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import torch  # Supported Windows native-library initialization order; no model fit.
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sign_language.five_sign_pilot import sha256, write_json
from sign_language.preprocessing import canonical_hands, normalize_sequence
from sign_language.wrist_trajectory import contiguous_spans, trajectory_summary

CACHE = ROOT/'data/training/asl-fifteen-signs-128-v1'
RAW = ROOT/'data/landmarks/asl-citizen-raw-v1-cpu-hands-pose'
VIDEOS = ROOT/'data/asl_citizen/ASL_Citizen/videos'
OUTPUT = ROOT/'docs/assets/please-my-trajectory-2026-11-12.json'
BOARDS = ROOT/'.cache/please-my-trajectory-20261112'
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),
         (10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]


def read(path):
    return json.loads(path.read_text())


def video_check(row, record, arrays, make_board):
    path = VIDEOS/row['video_file']
    before = sha256(path)
    if path.stat().st_size != record['source_bytes']:
        raise ValueError('Video size differs from extraction record')
    seen = np.flatnonzero(arrays['hand_present'].any(1))
    start, end = (int(seen[0]), int(seen[-1])) if len(seen) else (0, record['frames']-1)
    sample_ids = np.linspace(start, end, 12).round().astype(int)
    samples = {}
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    try:
        if not capture.isOpened():
            raise ValueError('Video cannot be opened')
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if list(frame.shape) != record['frame_shape']:
                raise ValueError('Frame shape differs from extraction')
            if make_board and count in sample_ids:
                samples[count] = frame.copy()
            count += 1
    finally:
        capture.release()
    if count != record['frames'] or declared != count or not np.isclose(fps, record['fps'], rtol=1.e-6):
        raise ValueError('Video decode count or FPS differs')
    if before != sha256(path):
        raise AssertionError('Source video changed')
    board = None
    if make_board:
        tile_w, tile_h = 320, 280
        canvas = np.full((3*tile_h+40, 4*tile_w, 3), 22, np.uint8)
        cv2.putText(canvas, f'{row["gloss"]} {row["participant_id"]} {row["video_file"]}', (8,25),
                    cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
        for tile, fid in enumerate(sample_ids):
            frame = samples[int(fid)]
            h,w = frame.shape[:2]
            for slot in (0,1):
                if arrays['hand_present'][fid,slot]:
                    points = (arrays['hands_xyz'][fid,slot,:,:2]*[w,h]).astype(int)
                    for a,b in EDGES:
                        cv2.line(frame,tuple(points[a]),tuple(points[b]),(0,220,0),1,cv2.LINE_AA)
                    for point in points:
                        cv2.circle(frame,tuple(point),2,(0,255,255),-1)
            scale = min(tile_w/w, (tile_h-30)/h)
            thumb = cv2.resize(frame,(round(w*scale),round(h*scale)))
            y,x = 40+(tile//4)*tile_h, (tile%4)*tile_w
            cv2.putText(canvas,f'frame {fid} / {count-1}',(x+5,y+21),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            canvas[y+30:y+30+len(thumb),x:x+thumb.shape[1]] = thumb
        target = BOARDS/f'{row["gloss"].lower()}-{row["participant_id"]}-{row["video_file"][:-4]}.png'
        if not cv2.imwrite(str(target),canvas):
            raise RuntimeError('Contact sheet write failed')
        board = str(target.relative_to(ROOT))
    return {'sha256': before, 'decoded_frames': count, 'fps': fps, 'contact_sheet': board}


def plot_validation(records, paths):
    chosen = [r for r in records if r['split']=='val']
    fig, axes = plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    for ax,row in zip(axes.flat, chosen, strict=True):
        p,valid,model,model_valid = paths[row['video_file']]
        for span in contiguous_spans(valid):
            ax.plot(p[span,0],p[span,1],color='#94a3b8',lw=1,alpha=.7)
        ids = np.flatnonzero(valid)
        if len(ids):
            ax.scatter(p[ids,0],p[ids,1],c=ids,cmap='viridis',s=12)
        for span in contiguous_spans(model_valid):
            ax.plot(model[span,0],model[span,1],'r.',ms=3,alpha=.6)
        ax.set(title=f'{row["gloss"]} {row["participant_id"]}\n{row["video_file"][:12]}...; slot {row["dominant_slot"]}',
               xlabel='Body-relative wrist X',ylabel='Body-relative wrist Y')
        ax.invert_yaxis();ax.set_aspect('equal',adjustable='datalim')
    fig.suptitle('Validation PLEASE / MY: full raw wrist paths and retained model samples',fontsize=15)
    fig.supxlabel('Raw path colored by time; red dots: cached input. Gaps are not joined. Shoulder-width units; not ground truth.',fontsize=10)
    fig.savefig(OUTPUT.with_suffix('.png'),dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,metric,title in zip(axes,('loop_probe_score','axis_ratio'),('Best window loop probe (heuristic)','Projected trajectory axis ratio'),strict=True):
        for x,(split,gloss) in enumerate((('train','PLEASE'),('train','MY'),('val','PLEASE'),('val','MY'))):
            selected=[r for r in records if r['split']==split and r['gloss']==gloss and r['model_trajectory']['best_loop_probe']]
            values=[r['model_trajectory']['best_loop_probe'][metric] for r in selected]
            ax.scatter(np.full(len(values),x)+np.linspace(-.1,.1,len(values)),values,color='#2563eb' if gloss=='PLEASE' else '#b45309',s=35)
        ax.set(xticks=range(4),xticklabels=['Train\nPLEASE','Train\nMY','Val\nPLEASE','Val\nMY'],title=title,ylim=(-.05,1.05))
    fig.supxlabel('Exploratory geometric descriptors: no fitted thresholds, no new sign classifier or label-cleaning decisions.',fontsize=10)
    fig.savefig(OUTPUT.with_name(OUTPUT.stem+'-metrics.png'),dpi=170)
    plt.close(fig)


def main():
    if OUTPUT.exists() or BOARDS.exists():
        raise FileExistsError('Preserve the previous investigation; use a new version')
    cv2.setNumThreads(1)
    BOARDS.mkdir(parents=True)
    manifest = read(CACHE/'manifest.json')
    cache_hashes = {n:sha256(CACHE/n) for n in manifest['files_sha256']}
    if cache_hashes != manifest['files_sha256']:
        raise ValueError('Subset checksum differs')
    frozen = read(ROOT/'docs/assets/fifteen-sign-temporal-2026-11-12.json')
    weights = {str(ROOT/'models/asl-gru-fifteen-signs-temporal-v1'/name/'best.pt'):frozen[name]['checkpoint_sha256'] for name in ('control','temporal')}
    if any(sha256(Path(p))!=h for p,h in weights.items()):
        raise ValueError('Checkpoint identity differs')
    raw_hashes, records, paths = {}, [], {}
    for split in ('train','val'):
        rows = read(CACHE/f'{split}-index.json')
        cached = np.load(CACHE/f'{split}-features.npy',mmap_mode='r')
        lengths = np.load(CACHE/f'{split}-lengths.npy')
        predictions = {n:{r['video_file']:r for r in frozen[n]['train' if split=='train' else 'validation']['predictions']} for n in ('control','temporal')}
        for i,row in enumerate(rows):
            if row['gloss'] not in ('PLEASE','MY'):
                continue
            parent = row['parent_index']; name=f'{split}-{parent//100:05d}'
            path=RAW/f'{name}.zip'
            if name not in raw_hashes:
                marker=read(RAW/f'{name}.complete.json');raw_hashes[name]=sha256(path)
                if raw_hashes[name]!=marker['sha256'] or path.stat().st_size!=marker['bytes']:
                    raise ValueError('Raw archive integrity mismatch')
            with zipfile.ZipFile(path) as archive:
                record=json.loads(archive.read('records.json'))['records'][parent%100]
                if record['status']!='ok' or record['class_id']!=row['parent_class_id'] or any(record[k]!=row[k] for k in ('video_file','participant_id','gloss','split')):
                    raise ValueError('Raw record identity mismatch')
                with np.load(io.BytesIO(archive.read(record['member'])),allow_pickle=False) as loaded:
                    arrays={k:loaded[k] for k in loaded.files}
            normalized, quality=normalize_sequence(arrays,record['frame_shape'])
            encoded=normalized.astype(np.float16)
            length=int(lengths[i]);features=np.asarray(cached[i,:length],np.float32)
            if len(encoded)!=length or not np.array_equal(encoded,cached[i,:length]):
                raise ValueError('Raw-to-prepared cache disagreement')
            aspect=record['frame_shape'][0]/record['frame_shape'][1]
            pose=arrays['pose_xyz'][...,:2].copy();pose[...,1]*=aspect
            mask=arrays['pose_valid_mask'];hands,present=canonical_hands(arrays,pose,mask,aspect)
            widths=np.linalg.norm(pose[:,11]-pose[:,12],axis=1)
            anchors=mask[:,11]&mask[:,12]&(widths>=.02)
            scale=float(np.median(widths[anchors])) if anchors.any() else 1.
            center=(pose[:,11]+pose[:,12])/2
            body=(hands[...,:2]-center[:,None,None])/scale
            usable=present&anchors[:,None]
            slot=int(usable.sum(0).argmax())
            wrist=body[:,slot,0];valid=usable[:,slot]
            sampled=np.linspace(0,len(wrist)-1,length).round().astype(int) if len(wrist)>128 else np.arange(len(wrist))
            model=features[:,39:123].reshape(length,2,21,2)[:,slot,0]
            model_valid=(features[:,249+slot]>0)&(features[:,253]>0)
            eligible=valid[sampled]&model_valid
            max_quant=float(np.max(np.abs(np.clip(wrist[sampled][eligible],-8,8)-model[eligible]))) if eligible.any() else None
            probe=trajectory_summary(wrist,valid,record['fps'])
            model_probe=trajectory_summary(model,model_valid,record['fps']*(length-1)/max(len(wrist)-1,1))
            result={**row,'source_frames':record['frames'],'model_frames':length,'dominant_slot':slot,
                    'raw_to_prepared_exact_at_float16':True,'retained_raw_body_to_cached_wrist_max_difference':max_quant,
                    'quality':quality,'raw_trajectory':probe,'model_trajectory':model_probe,
                    'video':video_check(row,record,arrays,split=='val'), 'predictions':{}}
            for n in predictions:
                prediction=predictions[n][row['video_file']];p=np.asarray(prediction['scores']);order=np.argsort(-p)
                please=manifest['labels']['PLEASE'];my=manifest['labels']['MY']
                result['predictions'][n]={'top1':prediction['prediction'],'please_score':float(p[please]),'my_score':float(p[my]),
                    'please_rank':int(np.flatnonzero(order==please)[0]+1),
                    'top1_if_my_excluded_diagnostic_only':list(manifest['labels'])[int(next(v for v in order if v!=my))]}
            records.append(result)
            if split=='val':
                paths[row['video_file']]=(wrist,valid,model,model_valid)
            print(split,row['gloss'],row['video_file'],'frames',record['frames'],'cache exact',flush=True)
    plot_validation(records,paths)
    report={'status':'complete','scope':'All PLEASE/MY training and validation clips only; bounded diagnostic, not resumed corpus audit or cleaning',
            'records':records,'counts':{s:{g:sum(r['split']==s and r['gloss']==g for r in records) for g in ('PLEASE','MY')} for s in ('train','val')},
            'cache_id':manifest['cache_id'],'raw_archives_sha256':raw_hashes,'source_cache_sha256':cache_hashes,
            'previous_weights_unchanged':all(sha256(Path(p))==h for p,h in weights.items()),
            'source_cache_unchanged':all(sha256(CACHE/n)==h for n,h in cache_hashes.items()),
            'raw_archives_unchanged':all(sha256(RAW/f'{n}.zip')==h for n,h in raw_hashes.items()),
            'training_performed':False,'new_extraction_performed':False,'webcam_opened':False,'test_read':False,
            'limits':['Path metrics are exploratory, not semantic annotations or measured joint accuracy.',
                      'Best loop window is selected geometrically and can include non-sign motion or tracking artifacts.',
                      'No diagnosis of dataset mislabeling, no cleaning, and no threshold/model change.']}
    if not all(report[k] for k in ('previous_weights_unchanged','source_cache_unchanged','raw_archives_unchanged')):
        raise AssertionError('Protected source changed')
    write_json(OUTPUT,report)
    print('PLEASE_MY_AUDIT_COMPLETE',json.dumps(report['counts']),flush=True)


if __name__=='__main__':main()
