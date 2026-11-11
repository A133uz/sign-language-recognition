"""Six matched frozen-GRU trials. Save scores/aggregate statistics only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
from pathlib import Path
import sys
import time

import torch
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.recognition import SignRecognizer
from sign_language.recognition_webcam import SkeletonPreview
from sign_language.frame_rate import FrameRateLimiter,capture_rate_metrics
from sign_language.five_sign_pilot import write_json,sha256
from sign_language.live_forensics import analyze_capture
from sign_language.mean_pool_comparison import compare_pooling_capture,compare_compact_capture


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--camera',type=int,default=0)
    p.add_argument('--candidate-checkpoint',type=Path,help='Optional masked-mean model for paired inference on the same observations')
    p.add_argument('--compact-reference',type=Path,help='NO/YES diagnostic reference; switches prompts to NO/YES/HELLO twice')
    a=p.parse_args()
    if a.report.exists():p.error('Use a new report path')
    # Only the locally generated diagnostic bank is accepted; no external pickle.
    bank_path=a.bank.resolve()
    if not bank_path.is_relative_to((ROOT/'.cache').resolve()):p.error('Bank must be a local project .cache artifact')
    with bank_path.open('rb') as f:bank=pickle.load(f)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    r=SignRecognizer(ROOT/'models/asl-gru-five-signs-early-v2/regularized/best.pt',ROOT/'models')
    if bank['checkpoint_sha256']!=r.checkpoint_sha256:raise ValueError('Diagnostic bank uses another checkpoint')
    candidate=None
    if a.candidate_checkpoint is not None:
        candidate=SignRecognizer(a.candidate_checkpoint,ROOT/'models')
        if candidate.config.get('architecture')!='gru-mean-pool-v1' or candidate.labels!=r.labels or candidate.config['cache_id']!=r.config['cache_id']:
            raise ValueError('Candidate must be a mean-pool GRU using the baseline labels/cache')
    prompts=['HELLO','THANKYOU','NO']*2
    reference=None
    if a.compact_reference is not None:
        if candidate is None:p.error('Compact-hand diagnostic requires a candidate checkpoint')
        reference=json.loads(a.compact_reference.read_text())
        if reference['cache_id']!=r.config['cache_id']:raise ValueError('Articulation reference uses another cache')
        prompts=['NO','YES','HELLO']*2
    report={'status':'starting','session_id':a.report.stem,'model':'SignGRU','epoch':r.epoch,
            'checkpoint_sha256':r.checkpoint_sha256,'prompts':prompts,'capture_seconds':4.,
            'media_saved':False,'coordinates_saved':False,'embeddings_saved':False,
            'production_model_trained':False,'user_completion_confirmed':None,'trials':[]}
    if candidate is not None:
        report['candidate']={'model':type(candidate.model).__name__,'epoch':candidate.epoch,
                             'checkpoint_sha256':candidate.checkpoint_sha256,'architecture':candidate.config['architecture']}
    if reference is not None:report.update(mode='compact_hand_articulation',reference_file=str(a.compact_reference),
        interpretation='Projected overlap is not confirmed tracking failure; scalar ranges include detected rest; compare train only.')
    write_json(a.report,report)
    camera=cv2.VideoCapture(a.camera);pool=ThreadPoolExecutor(max_workers=1);preview=SkeletonPreview(r.hand_model)
    preview_job=job=None;annotated=None;last=None;index=0;state='ready';deadline=0.;start=0.
    frames=[];timestamps=[];source=0;clock=time.perf_counter();previous=-1;limiter=FrameRateLimiter(30.)
    title=f'GRU epoch {r.epoch} | six-trial causal diagnostic | {a.report.stem}'
    if candidate is not None:title=f'GRU baseline {r.epoch} vs MEAN-POOL {candidate.epoch} | {a.report.stem}'
    try:
        if not camera.isOpened():raise RuntimeError('Cannot open camera')
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,640);camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480);camera.set(cv2.CAP_PROP_FPS,30)
        report.update(status='running',camera_open=True);write_json(a.report,report)
        print('GUIDED_WINDOW_READY | GRU epoch 18 | six trials | Space once per prompt | automatic 4s | no media saved',flush=True)
        if candidate is not None:print(f'PAIRED_MODELS_READY | baseline=SignGRU epoch {r.epoch} | candidate={type(candidate.model).__name__} epoch {candidate.epoch}',flush=True)
        if reference is not None:print('COMPACT_HAND_CHECK_READY | NO, YES, HELLO twice | finger separation + wrist/palm motion | no coordinates saved',flush=True)
        while True:
            ok,frame=camera.read()
            if not ok:raise RuntimeError('Camera read failure')
            now=time.perf_counter();stamp=max(previous+1,round((now-clock)*1000));previous=stamp
            if preview_job is not None and preview_job.done():
                annotated,_=preview_job.result();preview_job=None
            if job is not None and job.done():
                try:
                    last=job.result();last.update(capture_id=index+1,**capture_metrics)
                    report['trials'].append(last)
                    print('TRIAL_COMPLETE',index+1,last['expected'],
                          {n:s['top5'][0]['label'] for n,s in last['variants'].items()},flush=True)
                except Exception as exc:
                    report['trials'].append({'capture_id':index+1,'expected':prompts[index],'error':str(exc)})
                    print('TRIAL_ERROR',str(exc),flush=True)
                job=None;index+=1;state='done' if index==len(prompts) else 'ready';write_json(a.report,report)
            if state=='done':
                # Release the camera after all requested captures; no unbounded preview.
                report['status']='complete';break
            if preview_job is None and job is None:preview_job=pool.submit(preview.draw,frame.copy(),stamp)
            if state=='countdown' and now>=deadline:
                state='recording';start=now;deadline=now+4.;frames=[];timestamps=[];source=0;limiter=FrameRateLimiter(30.)
            if state=='recording':
                source+=1
                if limiter.accept(now-start):
                    frames.append(frame.copy());timestamps.append(max(timestamps[-1]+1 if timestamps else 0,round((now-start)*1000)))
                if now>=deadline:
                    capture_metrics=capture_rate_metrics(timestamps,source,now-start,30.)
                    capture_metrics.update(capture_stop_reason='time_limit',capture_wall_seconds=now-start)
                    if reference is not None:job=pool.submit(compare_compact_capture,r,candidate,bank,frames,timestamps,prompts[index],reference)
                    elif candidate is None:job=pool.submit(analyze_capture,r,bank,frames,timestamps,prompts[index])
                    else:job=pool.submit(compare_pooling_capture,r,candidate,bank,frames,timestamps,prompts[index])
                    state='processing';frames=[];timestamps=[]
            view=np.full((frame.shape[0]+230,frame.shape[1],3),22,np.uint8)
            view[230:]=cv2.flip(frame if job is not None or annotated is None else annotated,1)
            if state=='recording':headline=f'SIGN NOW: {prompts[index]} | {max(0,deadline-now):.1f}s remaining'
            elif state=='countdown':headline=f'GET READY: {prompts[index]} | {deadline-now:.1f}s'
            elif state=='processing':headline='Comparing controls on the SAME capture; please wait'
            else:headline=f'Trial {index+1}/6: {prompts[index]} | Space when ready'
            lines=['MODEL: SignGRU | epoch 18 | 5 labels | frozen weights',headline,
                   'Sign ONCE, then relax; no Space press to finish.',
                   'Keep shoulders/hands visible during the sign.',
                   'Only scores/aggregate measurements saved; Q cancels.']
            if candidate is not None:lines[0]=f'BASELINE GRU {r.epoch} vs MEAN-POOL GRU {candidate.epoch} | frozen'
            if last:
                lines.extend([f'Previous {last["expected"]}: original {last["variants"]["original"]["top5"][0]["label"]}',
                              'Variants are diagnostic controls, not fixes.'])
                if candidate is not None:lines[-1]='Mean-pool guess: '+last['variants']['mean_pool']['top5'][0]['label']
                if reference is not None:
                    m=last['articulation']['metrics'];gap=m.get('thumb_index_xy_range');angle=m.get('palm_axis_xy_excursion_deg')
                    if gap is not None:lines.append(f'Finger gap range {gap:.2f} | palm motion {angle:.1f} deg' if angle is not None else f'Finger gap range {gap:.2f} | palm motion unavailable')
            for line,text in enumerate(lines):cv2.putText(view,text,(10,25+line*28),cv2.FONT_HERSHEY_SIMPLEX,.48,(120,255,180),1,cv2.LINE_AA)
            cv2.imshow(title,view);key=cv2.waitKey(1)&255
            if key in (27,ord('q')) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
            if key==32 and state=='ready':state='countdown';deadline=now+1.5
        if report['status']!='complete':report['status']='partial'
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        camera.release();cv2.destroyAllWindows();pool.submit(preview.close);pool.shutdown(wait=True)
        if job is not None:
            try:report['trials'].append({**job.result(),'capture_id':index+1,**capture_metrics})
            except Exception as exc:report['trials'].append({'error':str(exc),'capture_id':index+1})
        report.update(camera_open=False,elapsed_seconds=time.perf_counter()-clock,
                      checkpoint_unchanged=sha256(r.checkpoint)==r.checkpoint_sha256)
        if candidate is not None:report['candidate_checkpoint_unchanged']=sha256(candidate.checkpoint)==candidate.checkpoint_sha256
        write_json(a.report,report)
    print('GUIDED_FINISHED',report['status'],len(report['trials']),flush=True)


if __name__=='__main__':main()
