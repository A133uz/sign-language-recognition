"""Guided frozen-model pronoun trials; live inputs stay in RAM."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time

import torch
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.dataset import write_json
from sign_language.five_sign_pilot import sha256
from sign_language.frame_rate import FrameRateLimiter,capture_rate_metrics
from sign_language.recognition import SignRecognizer
from sign_language.recognition_webcam import SkeletonPreview
from sign_language.pronoun_diagnostics import check_models,diagnose_capture,PROMPTS
from sign_language.capture_timing_diagnostics import CaptureTiming,diagnose_timing_capture
from sign_language.pose_transition_diagnostics import TRIALS as POSE_TRIALS,diagnose_pose_capture


def run(current,baseline,path,camera_index=0,*,timing_normalization=False,prompts=None,capture_seconds=4.,sign_seconds=2.,pose_entry_exit=False):
    if path.exists():raise FileExistsError('Preserve prior live evidence')
    check_models(current,baseline)
    if pose_entry_exit and (timing_normalization or prompts is not None or capture_seconds!=3.):
        raise ValueError('Pose comparison uses its fixed eight-trial, three-second protocol')
    prompts=[t['label'] for t in POSE_TRIALS] if pose_entry_exit else list(PROMPTS if prompts is None else prompts)
    if not prompts or any(p not in current.names for p in prompts):raise ValueError('Prompts must be checkpoint labels')
    if not 1<=capture_seconds<=4 or not 0<sign_seconds<=capture_seconds:raise ValueError('Invalid capture/sign duration')
    report={'status':'starting','session_id':path.stem,'prompts':prompts,
            'model':type(current.model).__name__,'checkpoint_sha256':current.checkpoint_sha256,
            'baseline_checkpoint_sha256':baseline.checkpoint_sha256,'classes':current.names,
            'capture_seconds':capture_seconds,'sign_during_first_seconds':sign_seconds,'capture_fps_limit':30.,
            'training_performed':False,'media_saved':False,'coordinates_saved':False,'embeddings_saved':False,
            'preview_separate_from_recognition':True,'user_completion_confirmed':None,'trials':[],
            'timing_normalization_mode':timing_normalization,'limiter_changed':False,'production_preprocessing_changed':False}
    report['pose_entry_exit_mode']=pose_entry_exit
    if pose_entry_exit:
        report['trial_protocol']=list(POSE_TRIALS)
        report['sign_during_first_seconds']=None
        report['protocol_limit']='Separate held and moving captures are repeated experimental conditions, not identical performances. Fixed-context controls compare windows within each single capture.'
    write_json(path,report)
    camera=cv2.VideoCapture(camera_index);pool=ThreadPoolExecutor(max_workers=1)
    preview=SkeletonPreview(current.hand_model);preview_job=job=None
    annotated=last=None;index=0;state='ready';deadline=began=0.
    frames=[];stamps=[];source=0;clock=time.perf_counter();previous=-1;metrics={}
    title=f'{"POSE / ENTRY-EXIT CHECK" if pose_entry_exit else "Guided pronoun diagnostic"} | Frozen trajectory GRU + original head | {path.stem}'
    window_ready=False;timing=None;previous_loop_end=None
    analyze=diagnose_pose_capture if pose_entry_exit else diagnose_timing_capture if timing_normalization else diagnose_capture
    def finish_job():
        nonlocal job,index,state,last
        try:
            last={**job.result(),'capture_id':index+1,'prompted_label':prompts[index],
                  'user_confirmation':None,'trial_status':'captured_pending_confirmation',**metrics}
        except Exception as exc:
            last={'capture_id':index+1,'prompted_label':prompts[index],
                  'trial_status':'processing_error','error':str(exc),'user_confirmation':None,**metrics}
        report['trials'].append(last)
        if pose_entry_exit:last.update(condition=POSE_TRIALS[index]['condition'],repetition=POSE_TRIALS[index]['repetition'])
        print('TRIAL_COMPLETE',index+1,prompts[index],
              last.get('windows',{}).get('full',{}).get('variants',{}).get('original',{}).get('current',{}).get('top5',[])[:2],flush=True)
        job=None;index+=1;state='done' if index==len(prompts) else 'ready'
        write_json(path,report)
    try:
        if not camera.isOpened():raise RuntimeError('Cannot open camera')
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,640);camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        camera.set(cv2.CAP_PROP_FPS,30)
        device_fps=camera.get(cv2.CAP_PROP_FPS) if hasattr(camera,'get') else None
        report['camera_reported_fps']=float(device_fps) if device_fps is not None and np.isfinite(device_fps) and device_fps>0 else None
        report.update(status='running',camera_open=True);write_json(path,report)
        while True:
            read_started=time.perf_counter() if timing_normalization else None
            ok,frame=camera.read()
            if not ok:raise RuntimeError('Camera read failure')
            now=time.perf_counter();stamp=max(previous+1,round((now-clock)*1000));previous=stamp
            if preview_job is not None and preview_job.done():
                annotated,_=preview_job.result();preview_job=None
            if job is not None and job.done():finish_job()
            if state=='done':report['status']='complete';break
            if preview_job is None and job is None:preview_job=pool.submit(preview.draw,frame.copy(),stamp)
            frame_timing=None;pending_submit=False
            if state=='countdown' and now>=deadline:
                state='recording';began=now;deadline=now+capture_seconds;frames=[];stamps=[];source=0;limiter=FrameRateLimiter(30.)
                if timing_normalization:timing=CaptureTiming(30.)
            if state=='recording':
                in_window=now-began<=capture_seconds
                if in_window:source+=1
                accepted=limiter.accept(now-began) if in_window else False
                decision_done=time.perf_counter() if timing_normalization else None
                if accepted:
                    frames.append(frame.copy());stamps.append(max(stamps[-1]+1 if stamps else 0,round((now-began)*1000)))
                if timing_normalization and in_window:
                    copied=time.perf_counter()
                    frame_timing={'arrival':now-began,'kept':accepted,'credit':limiter.credit,
                                  'stages':{'camera_read':(now-read_started)*1000.,
                                            'predecision':(decision_done-now)*1000.,
                                            'frame_copy':(copied-decision_done)*1000.,
                                            'between_loop':max(0.,(read_started-previous_loop_end)*1000.) if previous_loop_end is not None else 0.}}
                if now>=deadline:
                    metrics=capture_rate_metrics(stamps,source,now-began,30.)
                    metrics.update(capture_wall_seconds=now-began,capture_stop_reason=f'automatic_{capture_seconds:g}_seconds')
                    if timing_normalization:pending_submit=True
                    else:
                        job=pool.submit(analyze,current,baseline,frames,stamps)
                        frames=[];stamps=[]
                    state='processing'
            if state=='recording':
                headline=(f'SIGN NOW: {prompts[index]}' if now-began<sign_seconds else 'RELAX: let capture stop automatically')
                if pose_entry_exit:
                    headline=(f'HOLD {prompts[index]}: keep the pose steady' if POSE_TRIALS[index]['condition']=='hold_pose' else
                              f'SIGN {prompts[index]} ONCE' if now-began<2. else 'RELAX: lower your hand')
                headline+=f' | {max(0.,deadline-now):.1f}s'
            elif state=='countdown':headline=f'GET READY: {prompts[index]} | {max(0.,deadline-now):.1f}s'
            elif state=='processing':headline='Comparing the SAME observations; please wait'
            else:headline=f'Trial {index+1}/{len(prompts)}: {prompts[index]} | Space when ready'
            lines=['MODEL: Frozen trajectory readout | epoch 13 | CUDA | 15 labels',headline,
                   (f'Sign ONCE during {capture_seconds:g}s. Capture ends automatically.' if sign_seconds==capture_seconds else
                    f'Sign ONCE in first {sign_seconds:g}s, then relax. Auto stop at {capture_seconds:g}s.'),
                   'Keep shoulders and hands visible during the sign.',
                   'No video/landmarks saved. Q cancels; X marks previous trial invalid.']
            if pose_entry_exit:
                condition=POSE_TRIALS[index]['condition']
                lines[2]=(f'HOLD: form {prompts[index]} during GET READY; hold through all 3s.' if condition=='hold_pose' else
                          'NATURAL: start relaxed; sign once in first 2s, then lower hand.')
                lines[3]='ME: index into chest. YOU: index toward camera. Same hand each trial.'
            if last:
                top=last.get('windows',{}).get('full',{}).get('variants',{}).get('original',{}).get('current',{}).get('top5',[])
                lines.append(f'Previous #{last["capture_id"]} {last["prompted_label"]}: '+(top[0]['label'] if top else last['trial_status']))
            if timing_normalization:lines[0]='TIMING/NORMALIZATION CHECK | frozen GRU epoch 13 | 15 labels'
            render_started=time.perf_counter() if timing_normalization else None
            view=np.full((frame.shape[0]+215,frame.shape[1],3),22,np.uint8)
            view[215:]=cv2.flip(frame if job is not None or annotated is None else annotated,1)
            for i,line in enumerate(lines):cv2.putText(view,line,(8,25+29*i),cv2.FONT_HERSHEY_SIMPLEX,.46,(120,255,180),1,cv2.LINE_AA)
            show_started=time.perf_counter() if timing_normalization else None
            cv2.imshow(title,view)
            key_started=time.perf_counter() if timing_normalization else None
            key=cv2.waitKey(1)&255
            if timing_normalization:
                loop_end=time.perf_counter()
                if frame_timing is not None:
                    stages=frame_timing['stages']
                    stages.update(render=(show_started-render_started)*1000.,
                                  imshow=(key_started-show_started)*1000.,wait_key=(loop_end-key_started)*1000.,
                                  total_loop=(loop_end-read_started)*1000.)
                    timing.record(frame_timing['arrival'],frame_timing['kept'],stages,frame_timing['credit'])
                previous_loop_end=loop_end
                if pending_submit:
                    metrics['capture_timing']=timing.summary()
                    job=pool.submit(analyze,current,baseline,frames,stamps)
                    frames=[];stamps=[]
            if not window_ready:
                print(f'WINDOW_READY | {", ".join(prompts)} | Space once per prompt | {capture_seconds:g}s auto | no live inputs saved',flush=True)
                window_ready=True
            if key in (27,ord('q')) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
            if key==32 and state=='ready':state='countdown';deadline=now+(2.5 if pose_entry_exit else 1.5)
            if key==ord('x') and state=='ready' and last is not None:
                last.update(user_confirmation=False,trial_status='user_marked_invalid');write_json(path,report)
        if report['status']!='complete':report['status']='partial'
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        camera.release();cv2.destroyAllWindows()
        if job is not None:
            finish_job()
        elif state in ('recording','countdown'):
            report['trials'].append({'capture_id':index+1,'prompted_label':prompts[index],
                                     'trial_status':'interrupted','user_confirmation':False,
                                     **({'condition':POSE_TRIALS[index]['condition']} if pose_entry_exit else {})})
        pool.submit(preview.close);pool.shutdown(wait=True)
        frames=[];stamps=[]
        report.update(camera_open=False,elapsed_seconds=time.perf_counter()-clock,
                      checkpoints_unchanged=sha256(current.checkpoint)==current.checkpoint_sha256 and
                                            sha256(baseline.checkpoint)==baseline.checkpoint_sha256)
        write_json(path,report)
    print('PRONOUN_LIVE_FINISHED',report['status'],len(report['trials']),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--timing-normalization',action='store_true',help='Aggregate frame-arrival/limiter/stage timings and compare full-context-fixed normalization windows')
    parser.add_argument('--prompts',nargs='+',type=str.upper,help='Explicit ordered prompt list; all checkpoint labels still compete')
    parser.add_argument('--capture-seconds',type=float,default=3.,help='Automatic capture cap, 1-3 seconds (default 3)')
    parser.add_argument('--pose-entry-exit',action='store_true',help='Eight ME/YOU trials: hold posture versus natural entry/exit, twice each; fixed three-second captures')
    args=parser.parse_args()
    if args.report.exists():parser.error('Use a new report path')
    if not 1<=args.capture_seconds<=3:parser.error('Capture duration must be 1-3 seconds')
    if args.pose_entry_exit and (args.prompts or args.timing_normalization or args.capture_seconds!=3.):
        parser.error('Pose comparison requires default three seconds and no other mode/prompts')
    torch.set_num_threads(2);cv2.setNumThreads(1)
    current=SignRecognizer(ROOT/'models/asl-gru-fifteen-signs-trajectory-readout-c01/best.pt',ROOT/'models')
    baseline=SignRecognizer(ROOT/'models/asl-gru-fifteen-signs-temporal-v1/temporal/best.pt',ROOT/'models')
    run(current,baseline,args.report,args.camera,timing_normalization=args.timing_normalization,
        prompts=args.prompts,capture_seconds=args.capture_seconds,
        sign_seconds=args.capture_seconds if args.prompts is not None else min(2.,args.capture_seconds),
        pose_entry_exit=args.pose_entry_exit)


if __name__=='__main__':main()
