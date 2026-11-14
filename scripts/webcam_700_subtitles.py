"""Display one 700-sign GRU prediction as a subtitle in a live webcam window."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import cv2
import numpy as np
from sign_language.fitted_700_recognition import Fitted700Recognizer
from sign_language.frame_rate import FrameRateLimiter
from sign_language.recognition_webcam import SkeletonPreview
from sign_language.dataset import write_json


def draw_subtitle(image, text):
    """Draw exactly one centered line; fit even long sign labels to the frame."""
    if not text:return image
    h,w=image.shape[:2];scale=.85;thickness=2
    (tw,th),baseline=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,thickness)
    if tw>w-32:
        scale*=max(.1,(w-32)/tw)
        (tw,th),baseline=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,thickness)
    x=max(8,(w-tw)//2);y=h-max(22,round(h*.08))
    layer=image.copy();cv2.rectangle(layer,(max(0,x-12),max(0,y-th-9)),(min(w-1,x+tw+12),min(h-1,y+baseline+8)),(0,0,0),-1)
    cv2.addWeighted(layer,.68,image,.32,0,image)
    cv2.putText(image,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),thickness,cv2.LINE_AA)
    return image


def run(recognizer,report_path,camera=0,capture_seconds=3.,capture_fps=30.,title=None):
    capture=cv2.VideoCapture(camera);title=title or 'ASL 700 - GRU 2x256 + fitted linear classifier'
    pool=ThreadPoolExecutor(max_workers=2);preview=SkeletonPreview(recognizer.hand_model)
    preview_job=prediction_job=None;preview_frame=None;state='ready';subtitle='SPACE: START  |  Q: CLOSE'
    frames=[];timestamps=[];deadline=started=0.;limiter=FrameRateLimiter(capture_fps);last_preview=-1
    report={'status':'starting','backbone_sha256':recognizer.checkpoint_sha256,'head_sha256':getattr(recognizer,'head_sha256',None),
            'backbone_epoch':recognizer.epoch,'classifier_epoch':getattr(recognizer,'head_epoch',None),'classes':len(recognizer.names),
            'video_saved':False,'landmarks_saved':False,'trials':[]}
    def save():write_json(report_path,report)
    def completed(job):
        result=job.result();item={'capture_id':len(report['trials'])+1,'prediction':result['top1']['label'],
             'captured_frames':result['captured_frames'],'input_frames':result['input_frames'],
             'capture_seconds':result['capture_seconds'],'processing_seconds':result['processing_seconds']}
        report['trials'].append(item);save();print('PREDICTION',json.dumps(item),flush=True)
        return item['prediction']
    try:
        if not capture.isOpened():raise RuntimeError(f'Cannot open camera {camera}')
        capture.set(cv2.CAP_PROP_FRAME_WIDTH,640);capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480);capture.set(cv2.CAP_PROP_FPS,capture_fps)
        cv2.namedWindow(title,cv2.WINDOW_NORMAL)
        report.update(status='running',camera_open=True);save()
        print(f'WINDOW_READY | {title} | Space starts {capture_seconds:g}-second capture | Q closes',flush=True)
        begin=time.perf_counter()
        while True:
            ok,frame=capture.read()
            if not ok or frame is None:raise RuntimeError('Camera stopped returning frames')
            now=time.perf_counter();timestamp=max(last_preview+1,round((now-begin)*1000));last_preview=timestamp
            if preview_job is not None and preview_job.done():
                try:preview_frame,_=preview_job.result()
                except Exception as e:print('PREVIEW_ERROR',str(e),flush=True)
                preview_job=None
            if preview_job is None:preview_job=pool.submit(preview.draw,frame.copy(),timestamp)
            if prediction_job is not None and prediction_job.done():
                try:subtitle=completed(prediction_job)
                except Exception as e:
                    subtitle='PLEASE TRY AGAIN';report['trials'].append({'capture_id':len(report['trials'])+1,'error':str(e)});save();print('PREDICTION_ERROR',str(e),flush=True)
                prediction_job=None;state='ready'
            if state=='countdown':
                subtitle='GET READY'
                if now>=deadline:state='recording';frames=[];timestamps=[];started=now;limiter=FrameRateLimiter(capture_fps)
            if state=='recording':
                subtitle='SIGN NOW'
                if now-started<=capture_seconds and limiter.accept(now-started):
                    frames.append(frame.copy());stamp=round((now-started)*1000)
                    timestamps.append(max(stamp,timestamps[-1]+1 if timestamps else 0))
                if now-started>=capture_seconds:
                    if len(frames)<8:subtitle='PLEASE TRY AGAIN';state='ready'
                    else:prediction_job=pool.submit(recognizer.predict_frames,frames,timestamps);state='processing';subtitle='RECOGNIZING...'
                    frames=[];timestamps=[]
            image=preview_frame if preview_frame is not None else frame
            view=draw_subtitle(cv2.flip(image,1),subtitle)
            cv2.imshow(title,view);key=cv2.waitKey(1)&0xff
            if key in (27,ord('q')) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
            if key==ord(' ') and state=='ready':state='countdown';deadline=now+1.5;subtitle='GET READY'
        report.update(status='complete',reason='user_closed')
    except BaseException as exc:
        report.update(status='failed',reason=str(exc));raise
    finally:
        capture.release();cv2.destroyAllWindows()
        if prediction_job is not None:
            try:completed(prediction_job)
            except Exception as exc:print('PREDICTION_ERROR',str(exc),flush=True)
        if preview_job is not None:
            try:preview_job.result()
            except Exception:pass
        preview.close();pool.shutdown(wait=True)
        report['camera_open']=False;save()
    return report


def main():
    torch.set_num_threads(2)
    backbone=ROOT/'models/asl-700-sign-gru-comparison-v1/gru_2x256/best-backbone.pt'
    head=ROOT/'models/asl-700-sign-gru-2x256-two-layer-head-v1/linear.pt'
    recognizer=Fitted700Recognizer(backbone,head,ROOT/'models')
    report=ROOT/f'.cache/webcam-700-fitted-linear-{datetime.now():%Y%m%d-%H%M%S}.json'
    run(recognizer,report)

if __name__=='__main__':main()
