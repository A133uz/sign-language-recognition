"""Four guided captures for the unchanged official pretrained I3D; no media saved."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys
import time

import torch
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.official_citizen_i3d import OfficialCitizenI3D,preprocess_captured_frames,sha
from sign_language.five_sign_pilot import write_json
from sign_language.frame_rate import FrameRateLimiter,capture_rate_metrics

PROMPTS=('HELLO','THANKYOU','YOU','ME')


def score_capture(baseline,frames,stamps,pilot):
    tensor,metadata=preprocess_captured_frames(frames,baseline.dataset_module,baseline.transforms)
    z=baseline.logits(tensor.unsqueeze(0))[0].astype(float)
    def scores(ids):
        a=z[ids];p=np.exp(a-a.max());p/=p.sum();order=np.argsort(-p,kind='stable')[:5]
        return [{'label':baseline.names[ids[int(i)]],'score':float(p[i])} for i in order]
    return {'full_top5':scores(list(range(len(baseline.names)))),
            'pilot15_top5':scores([baseline.labels[g] for g in pilot]),'input':metadata,
            'captured_seconds':(stamps[-1]-stamps[0])/1000.,'media_saved':False,'coordinates_saved':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    if args.report.exists():raise FileExistsError('Preserve old session')
    torch.set_num_threads(2);cv2.setNumThreads(1);baseline=OfficialCitizenI3D(ROOT)
    pilot=list(json.loads((ROOT/'models/asl-gru-fifteen-signs-trajectory-readout-c01/labels.json').read_text()))
    report={'status':'starting','session_id':args.report.stem,'model':'Official ASL Citizen InceptionI3d',
            'checkpoint_sha256':baseline.config['checkpoint_sha256'],'commit':baseline.config['commit'],
            'full_output_classes':2731,'secondary_output_view':pilot,'capture_seconds':3,'fps_limit':30,
            'prompts':list(PROMPTS),'training':False,'landmark_extraction':False,'existing_recognizer_changed':False,
            'media_saved':False,'coordinates_saved':False,'user_completion_confirmed':None,'trials':[]}
    write_json(args.report,report);camera=cv2.VideoCapture(0);pool=ThreadPoolExecutor(max_workers=1)
    state='ready';index=0;job=None;last=None;frames=[];stamps=[];deadline=began=0.;metrics={};started=time.perf_counter();ready=False
    title='OFFICIAL ASL CITIZEN I3D | released weights | '+args.report.stem
    def finish():
        nonlocal job,last,index,state
        try:last={**job.result(),**metrics,'capture_id':index+1,'prompted_label':PROMPTS[index],
                  'user_confirmation':None,'trial_status':'captured_pending_confirmation'}
        except Exception as e:last={'capture_id':index+1,'prompted_label':PROMPTS[index],'error':str(e),'trial_status':'processing_error','user_confirmation':None}
        report['trials'].append(last);index+=1;job=None;state='done' if index==len(PROMPTS) else 'ready'
        print('TRIAL_COMPLETE',index,last.get('full_top5',[])[:1],last.get('pilot15_top5',[])[:1],flush=True);write_json(args.report,report)
    try:
        if not camera.isOpened():raise RuntimeError('Cannot open webcam')
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,640);camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480);camera.set(cv2.CAP_PROP_FPS,30)
        report.update(status='running',camera_open=True);write_json(args.report,report)
        while True:
            ok,frame=camera.read();now=time.perf_counter()
            if not ok:raise RuntimeError('Camera read failed')
            if job is not None and job.done():finish()
            if state=='done':report['status']='complete';break
            warmed=now-started>=2.
            if state=='countdown' and now>=deadline:
                state='recording';began=now;deadline=now+3.;frames=[];stamps=[];source=0;limiter=FrameRateLimiter(30.)
            if state=='recording':
                if now-began<=3.:
                    source+=1
                    if limiter.accept(now-began):frames.append(frame.copy());stamps.append(max(stamps[-1]+1 if stamps else 0,round((now-began)*1000)))
                if now>=deadline:
                    metrics=capture_rate_metrics(stamps,source,now-began,30.)
                    job=pool.submit(score_capture,baseline,frames,stamps,pilot);frames=[];stamps=[];state='processing'
            heading=(f'SIGN NOW: {PROMPTS[index]} | {max(0,deadline-now):.1f}s' if state=='recording' else
                     f'GET READY: {PROMPTS[index]}' if state=='countdown' else
                     'Processing video with official I3D...' if state=='processing' else
                     f'#{index+1}/4 {PROMPTS[index]} | Space when ready' if warmed else 'Camera warming up...')
            view=np.full((frame.shape[0]+180,frame.shape[1],3),22,np.uint8);view[180:]=cv2.flip(frame,1)
            lines=['MODEL: Official ASL Citizen I3D | CUDA | 2731 signs',heading,
                   'Space once; sign ONCE; automatic stop after 3s. Q cancels.',
                   'Video model: keep face, shoulders and hands inside camera.']
            if last:
                if last.get('full_top5'):
                    a=last['full_top5'][0];b=last['pilot15_top5'][0]
                    lines += [f'#{last["capture_id"]} Official full vocabulary: {a["label"]} ({a["score"]:.1%})',
                              f'Same model, 15-label view: {b["label"]} ({b["score"]:.1%})']
                else:lines += ['Previous capture could not be scored.']
            for i,line in enumerate(lines):cv2.putText(view,line,(8,24+i*27),cv2.FONT_HERSHEY_SIMPLEX,.46,(150,255,180),1,cv2.LINE_AA)
            cv2.imshow(title,view);key=cv2.waitKey(1)&255
            if not ready and warmed:print('WINDOW_READY | Official released I3D | HELLO, THANKYOU, YOU, ME | 3s auto | no media saved',flush=True);ready=True
            if key in (27,ord('q')) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
            if key==32 and state=='ready' and warmed:state='countdown';deadline=now+1.5
        if report['status']!='complete':report['status']='partial'
    except Exception as e:report.update(status='failed',error=str(e));raise
    finally:
        camera.release();cv2.destroyAllWindows()
        if job is not None:finish()
        elif state in ('countdown','recording'):
            report['trials'].append({'capture_id':index+1,'prompted_label':PROMPTS[index],'trial_status':'interrupted','user_confirmation':False})
        pool.shutdown(wait=True);frames=[];stamps=[]
        report.update(camera_open=False,checkpoint_unchanged=sha(baseline.checkpoint)==baseline.config['checkpoint_sha256'])
        write_json(args.report,report)
    print('OFFICIAL_I3D_WEBCAM_FINISHED',report['status'],flush=True)


if __name__=='__main__':main()
