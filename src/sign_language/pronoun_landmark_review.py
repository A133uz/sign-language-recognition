"""Bounded RAM-only annotated YOU review; production geometry stays enabled."""
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

import cv2
import numpy as np
from sign_language.hands import CONNECTIONS
from sign_language.preprocessing import canonical_hands
from sign_language.recognition_diagnostics import hand_visibility_summary
from sign_language.preprocessing import normalize_sequence


class MemoryBoards:
    """Loopback-only, bounded PNG buffer; no image filesystem or request logs."""
    def __init__(self,limit=4):
        self.limit=limit;self.images=[];self.lock=threading.Lock();self.cleared=threading.Event()
        token=secrets.token_urlsafe(24);owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                prefix='/'+token+'/'
                if not self.path.startswith(prefix):self.send_error(404);return
                suffix=self.path[len(prefix):]
                with owner.lock:
                    if suffix=='status':body=json.dumps({'boards':len(owner.images),'cleared':owner.cleared.is_set()}).encode();mime='application/json'
                    elif suffix.isdigit() and int(suffix)<len(owner.images):body=owner.images[int(suffix)];mime='image/png'
                    else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',mime)
                self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
                self.end_headers();self.wfile.write(body)
            def do_POST(self):
                if self.path!='/'+token+'/clear':self.send_error(404);return
                owner.clear();self.send_response(204);self.end_headers()
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.url=f'http://127.0.0.1:{self.server.server_port}/{token}/'
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()

    def add(self,board):
        ok,encoded=cv2.imencode('.png',board)
        if not ok:raise RuntimeError('Cannot encode review board')
        with self.lock:
            if self.cleared.is_set() or len(self.images)>=self.limit:raise ValueError('RAM preview limit reached or cleared')
            index=len(self.images);self.images.append(encoded.tobytes());return index

    def clear(self):
        with self.lock:self.images.clear();self.cleared.set()

    def close(self):
        self.clear();self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)


def canonical(arrays,shape):
    pose=arrays['pose_xyz'][...,:2].copy();aspect=shape[0]/shape[1];pose[...,1]*=aspect
    return canonical_hands(arrays,pose,arrays['pose_valid_mask'],aspect)


def overlay(frame,hands,present,index):
    """Project actual canonical model observations onto unmirrored source pixels."""
    out=frame.copy();width=frame.shape[1]
    for slot in (0,1):
        if not present[index,slot]:continue
        points=np.clip(np.rint(hands[index,slot,:,:2]*width),-10000,10000).astype(int)
        for a,b in CONNECTIONS:cv2.line(out,tuple(points[a]),tuple(points[b]),(255,180,0),2,cv2.LINE_AA)
        for a,b in ((0,5),(5,6),(6,7),(7,8)):cv2.line(out,tuple(points[a]),tuple(points[b]),(0,255,0),3,cv2.LINE_AA)
        for j,p in enumerate(points):
            cv2.circle(out,tuple(p),3,(0,0,255) if j==8 else (0,255,255),-1)
            if j in (0,5,6,7,8):cv2.putText(out,str(j),tuple(p+[4,-3]),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        x,y=np.clip(points[0],[5,20],[max(5,width-160),frame.shape[0]-5])
        cv2.putText(out,f'canonical slot {slot}',(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    return out


def fit_tile(frame,width,height):
    scale=min(width/frame.shape[1],height/frame.shape[0]);out=np.full((height,width,3),20,np.uint8)
    view=cv2.resize(frame,(max(1,round(frame.shape[1]*scale)),max(1,round(frame.shape[0]*scale))))
    y=(height-len(view))//2;x=(width-view.shape[1])//2;out[y:y+len(view),x:x+view.shape[1]]=view
    return out


def hand_crop(frame,hands,present,index):
    slots=np.flatnonzero(present[index])
    if not len(slots):return frame.copy()
    points=hands[index,slots[0],:,:2]*frame.shape[1]
    lo=points.min(0);hi=points.max(0);margin=max(25.,float((hi-lo).max())*.35)
    x0,y0=np.maximum(0,np.floor(lo-margin).astype(int));x1,y1=np.minimum(frame.shape[1::-1],np.ceil(hi+margin).astype(int))
    return frame[y0:y1,x0:x1].copy() if x1>x0 and y1>y0 else frame.copy()


def hand_shape_summary(hands,present,index,width):
    summaries=[]
    for slot in np.flatnonzero(present[index]):
        p=hands[index,slot];palm=float(np.linalg.norm(p[[5,9,13,17],:2]-p[0,:2],axis=1).mean())
        if palm<=0:continue
        segments=np.linalg.norm(np.diff(p[[5,6,7,8],:2],axis=0),axis=1)
        summaries.append({'slot':int(slot),'palm_size_pixels':palm*width,
            'projected_tip_to_mcp_over_palm':float(np.linalg.norm(p[8,:2]-p[5,:2])/palm),
            'projected_tip_to_wrist_over_palm':float(np.linalg.norm(p[8,:2]-p[0,:2])/palm),
            'projected_index_chord_over_bone_path':float(np.linalg.norm(p[8,:2]-p[5,:2])/segments.sum()) if segments.sum()>0 else None,
            'estimated_local_index_z_span_over_palm':float(np.ptp(p[[5,6,7,8],2])/palm)})
    return summaries


def review_board(frame,arrays,shape,index,stamp,reference,capture_id,top):
    hands,present=canonical(arrays,shape);annotated=overlay(frame,hands,present,index)
    board=np.full((900,1280,3),20,np.uint8)
    cv2.putText(board,f'YOU capture #{capture_id} | t={stamp/1000:.2f}s | first: {top} | GEOMETRY ON',(12,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1)
    board[60:540,:640]=fit_tile(frame,640,480);board[60:540,640:]=fit_tile(annotated,640,480)
    cv2.putText(board,'Unmirrored camera input',(10,53),cv2.FONT_HERSHEY_SIMPLEX,.5,(220,220,220),1)
    cv2.putText(board,'Same observations used for classification',(650,53),cv2.FONT_HERSHEY_SIMPLEX,.5,(220,220,220),1)
    tiles=[hand_crop(frame,hands,present,index),hand_crop(annotated,hands,present,index),reference['raw'],reference['annotated']]
    labels=['Live hand crop','Live tracked index: 5-6-7-8','Public YOU reference (P12)','Public cached landmarks']
    for i,(tile,label) in enumerate(zip(tiles,labels)):
        board[580:840,i*320:(i+1)*320]=fit_tile(tile,320,260)
        cv2.putText(board,label,(i*320+5,565),cv2.FONT_HERSHEY_SIMPLEX,.42,(235,235,235),1)
    cv2.putText(board,'Index chain green; index tip 8 red. Slots are assignments, not verified anatomical ground truth.',(10,865),cv2.FONT_HERSHEY_SIMPLEX,.51,(225,225,225),1)
    cv2.putText(board,'Forward-pointing fingers are foreshortened. No depth correctness can be established from one image.',(10,890),cv2.FONT_HERSHEY_SIMPLEX,.48,(225,225,225),1)
    return board


class PronounReviewRecognizer:
    """Use unchanged production predict_observations; annotate that same tracking pass."""
    def __init__(self,recognizer,boards,reference):
        self.recognizer=recognizer;self.boards=boards;self.reference=reference;self.captures=0
    def __getattr__(self,name):return getattr(self.recognizer,name)
    def review_instructions(self):
        if self.captures>=2:return 'GEOMETRY ON | Two captures complete: press Q to close.'
        return ('GEOMETRY ON | #1 YOU: start relaxed, sign once, then relax.' if self.captures==0 else
                'GEOMETRY ON | #2 YOU: form pose during GET READY; hold 3s.')
    def predict_frames(self,frames,timestamps):
        if self.captures>=2:raise ValueError('Two YOU captures complete. Press Q to close the camera.')
        self.captures+=1;capture_id=self.captures;started=time.perf_counter()
        arrays,shape,metadata=self.recognizer.observe_frames(frames,timestamps)
        # No variants, geometry ablations, alternate head or alternate preprocessing.
        try:prediction=self.recognizer.predict_observations(arrays,shape)
        except ValueError as exc:
            _,quality=normalize_sequence(arrays,shape)
            prediction={'top5':[],'quality':quality,'prediction_error':str(exc),'input_frames':quality['frames']}
        first=prediction['top5'][0]['label'] if prediction['top5'] else 'UNSCORABLE'
        hands,present=canonical(arrays,shape)
        records=[]
        for target in (1000,1600):
            index=min(range(len(timestamps)),key=lambda i:abs(timestamps[i]-timestamps[0]-target))
            board=review_board(frames[index],arrays,shape,index,timestamps[index]-timestamps[0],
                               self.reference,capture_id,first)
            number=self.boards.add(board)
            records.append({'board_id':number,'frame_index':index,'timestamp_ms':int(timestamps[index]-timestamps[0]),
                            'raw_hand_count':int(arrays['hand_present'][index].sum()),
                            'hand_shape_aggregates':hand_shape_summary(hands,present,index,shape[1]),
                            'selection':'Nearest retained frame to fixed 1.0s or 1.6s, independent of labels/predictions'})
        return {**prediction,**metadata,'review_boards':records,'geometry_enabled':True,
                'geometry_features':23,'head_interventions_performed':False,
                'recognition_tracking_passes':1,'media_saved_to_files':False,'coordinates_saved':False,
                'embeddings_saved':False,'user_confirmation':None,
                'intended_condition':'natural_entry_exit' if capture_id==1 else 'hold_pose',
                'tracking_summary':hand_visibility_summary(arrays['hand_present']),
                'processing_seconds':time.perf_counter()-started}
