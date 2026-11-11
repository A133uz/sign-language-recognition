"""Export exactly two consented annotated stills from one in-memory capture."""
from pathlib import Path
import time

import cv2
import numpy as np

from sign_language.five_sign_pilot import write_json
from sign_language.hands import CONNECTIONS
from sign_language.preprocessing import canonical_hands
from sign_language.recognition_diagnostics import hand_visibility_summary

POSE_EDGES=((11,12),(11,13),(13,15),(12,14),(14,16))


def annotate(frame, arrays, hands, present, index, timestamp):
    """Draw canonical hand assignments on original unmirrored source pixels."""
    image=frame.copy();height,width=image.shape[:2]
    point=lambda xy:tuple(np.clip(np.rint(xy),-10000,10000).astype(np.int32))
    pose=arrays['pose_xyz'][index,:,:2]*[width,height]
    valid=arrays['pose_valid_mask'][index]
    for a,b in POSE_EDGES:
        if valid[a] and valid[b]:cv2.line(image,point(pose[a]),point(pose[b]),(255,170,0),2)
    for slot in (0,1):
        if not present[index,slot]:continue
        points=hands[index,slot,:,:2]*width
        color=(255,255,0) if slot==0 else (255,0,255)
        for a,b in CONNECTIONS:cv2.line(image,point(points[a]),point(points[b]),color,2)
        for p in points:cv2.circle(image,point(p),3,color,-1)
        x,y=point(points[0]);x=max(2,min(width-160,x));y=max(18,min(height-5,y))
        cv2.putText(image,f'assigned {"L" if slot==0 else "R"}',(x,y),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
    board=np.full((height+46,width,3),20,np.uint8);board[46:]=image
    cv2.putText(board,f'UNMIRRORED INPUT | t={timestamp/1000:.2f}s | hands={int(present[index].sum())}',
                (8,18),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1)
    cv2.putText(board,'cyan/magenta: assigned hand slots | blue: valid upper pose',
                (8,37),cv2.FONT_HERSHEY_SIMPLEX,.4,(220,220,220),1)
    return board


class StillReviewRecognizer:
    review_stills=True

    def __init__(self, recognizer, output):
        self.recognizer=recognizer;self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.used=False;self.saved_still_images=[]

    def __getattr__(self,name):return getattr(self.recognizer,name)

    def predict_frames(self,frames,timestamps):
        if self.used:raise ValueError('Two review stills already captured. Press Q to close.')
        self.used=True
        started=time.perf_counter()
        arrays,shape,metadata=self.recognizer.observe_frames(frames,timestamps)
        prediction=self.recognizer.predict_observations(arrays,shape)
        xy=arrays['pose_xyz'][...,:2].copy();xy[...,1]*=shape[0]/shape[1]
        hands,present=canonical_hands(arrays,xy,arrays['pose_valid_mask'],shape[0]/shape[1])
        gesture=min(range(len(timestamps)),key=lambda i:abs(timestamps[i]-1000))
        selected=[('during-hello.png',gesture),('ending.png',len(frames)-1)]
        image_records=[]
        for filename,index in selected:
            board=annotate(frames[index],arrays,hands,present,index,timestamps[index])
            path=self.output/filename
            if not cv2.imwrite(str(path),board):raise OSError('Failed to save review still')
            self.saved_still_images.append(str(path.resolve()))
            image_records.append({'file':str(path.resolve()),'frame_index':index,
                                  'timestamp_ms':timestamps[index],'detected_hands':int(present[index].sum())})
        record={'consent':'User approved two annotated stills visible in chat; kept out of training data.',
                'checkpoint_sha256':self.recognizer.checkpoint_sha256,'images':image_records,
                'frame_shape':list(shape),'video_saved':False,'coordinates_saved':False,'training_performed':False,
                'selection':'Nearest retained frame to 1 second, then final frame; no label-based selection.'}
        write_json(self.output/'review.json',record)
        print('REVIEW_IMAGES_READY',self.saved_still_images,flush=True)
        return {**prediction,**metadata,'tracking_summary':hand_visibility_summary(arrays['hand_present']),
                'review_image_files':list(self.saved_still_images),'processing_seconds':time.perf_counter()-started}
