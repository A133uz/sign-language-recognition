"""Inference-only adapter for the notebook-trained 15-sign face-aware GRU."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import cv2
import mediapipe as mp
import numpy as np

from sign_language.hands import HandTracker
from sign_language.preprocessing import normalize_sequence
from sign_language.raw_landmarks import frame_observation
from sign_language.trajectory_geometry import FEATURES,trajectory_batch
from sign_language.recognition_diagnostics import hand_visibility_summary


def hash_file(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def normalize_joined(body_arrays,face_xyz,face_present,blendshapes,frame_shape):
    """Match notebook 08's full-clip face normalization and shared 128 indices."""
    body,quality=normalize_sequence(body_arrays,frame_shape)
    points=np.asarray(face_xyz,dtype=np.float32).copy()
    present=np.asarray(face_present,dtype=bool)
    blend=np.asarray(blendshapes,dtype=np.float32).copy()
    total=len(points)
    if points.shape!=(total,478,3) or present.shape!=(total,) or blend.shape!=(total,52) or len(body_arrays['pose_xyz'])!=total or total<1:
        raise ValueError('Face and body frame shapes differ')
    if not np.isfinite(points).all() or not np.isfinite(blend).all():raise ValueError('Nonfinite face data')
    points[:,:,1]*=frame_shape[0]/frame_shape[1]
    dist=np.linalg.norm(points[:,33,:2]-points[:,263,:2],axis=1)
    valid=present & np.isfinite(dist) & (dist>1.e-5)
    scale=float(np.median(dist[valid])) if valid.any() else 1.
    local=(points-points[:,1:2,:])/scale;local[~valid]=0;local=np.clip(local,-8.,8.);blend[~valid]=0
    face=np.concatenate((local.reshape(total,-1),blend,valid[:,None].astype(np.float32)),axis=1)
    if total>128:face=face[np.linspace(0,total-1,128).round().astype(int)]
    if len(body)!=len(face):raise ValueError('Face/body sample length mismatch')
    joined=np.concatenate((body.astype(np.float16).astype(np.float32),face.astype(np.float16).astype(np.float32)),axis=1)
    if joined.shape!=(min(total,128),1741) or not np.isfinite(joined).all():raise ValueError('Invalid joined features')
    return joined,{**quality,'frames_with_face':int(valid.sum())}


class FaceTrajectoryInference(nn.Module):
    """Runtime mirror of notebook 08; model design/training lives in the notebook."""
    def __init__(self):
        super().__init__();self.gru=nn.GRU(1741,256,num_layers=2,batch_first=True,dropout=.5)
        self.dropout=nn.Dropout(.5);self.classifier=nn.Linear(279,15)
        for name in ('median','mean','scale'):
            self.register_buffer('geometry_'+name,torch.ones(23) if name=='scale' else torch.zeros(23))
    def standardized_geometry(self,g):return (torch.where(torch.isnan(g),self.geometry_median,g)-self.geometry_mean)/self.geometry_scale
    def encode(self,x,lengths):
        packed=pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False);packed,_=self.gru(packed)
        values,actual=pad_packed_sequence(packed,batch_first=True)
        mask=(torch.arange(values.shape[1],device=values.device)[None,:]<actual.to(values.device)[:,None]).unsqueeze(-1)
        return (values*mask).sum(1)/actual.to(values.device).clamp_min(1)[:,None]
    def score(self,x,lengths):
        with torch.autocast('cuda',dtype=torch.float16,enabled=x.device.type=='cuda'):
            h=self.encode(x,lengths).float()
        g=torch.as_tensor(trajectory_batch(x[:,:,:254],lengths),device=x.device,dtype=torch.float32)
        with torch.autocast(device_type=x.device.type,enabled=False):
            return self.classifier(torch.cat((h,self.standardized_geometry(g)),1).float())


class FaceAwareRecognizer:
    def __init__(self,checkpoint,models,device='cuda'):
        self.checkpoint=Path(checkpoint);self.checkpoint_sha256=hash_file(self.checkpoint);self.device=torch.device(device)
        if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        state=torch.load(self.checkpoint,map_location='cpu',weights_only=True);self.config=state['config'];self.labels=state['labels'];self.epoch=state['epoch']
        if self.config!=json.loads(self.checkpoint.with_name('config.json').read_text(encoding='utf-8')):raise ValueError('Config mismatch')
        if self.labels!=json.loads(self.checkpoint.with_name('labels.json').read_text(encoding='utf-8')) or sorted(self.labels.values())!=list(range(15)):raise ValueError('Label map mismatch')
        expected={'architecture':'notebook-face-trajectory-gru-v1','input_size':1741,'hidden_size':256,'layers':2,'classes':15,'max_frames':128}
        if any(self.config.get(k)!=v for k,v in expected.items()) or self.config['geometry_features']!=list(FEATURES):raise ValueError('Unsupported model configuration')
        self.names=[name for name,_ in sorted(self.labels.items(),key=lambda item:item[1])]
        self.hand_model=Path(models)/'hand_landmarker.task';self.pose_model=Path(models)/'pose_landmarker_lite.task';self.face_model=Path(models)/'face_landmarker.task'
        for model in (self.hand_model,self.pose_model,self.face_model):
            if not model.is_file():raise FileNotFoundError(model)
        if hash_file(self.face_model)!=self.config['face_model_sha256']:raise ValueError('Face detector checksum mismatch')
        torch.backends.cudnn.enabled=False;self.model=FaceTrajectoryInference().to(self.device).eval();self.model.load_state_dict(state['model'],strict=True)
        for name in ('median','mean','scale'):
            if not torch.equal(getattr(self.model,'geometry_'+name),torch.tensor(self.config['geometry_statistics'][name],device=self.device,dtype=torch.float32)):
                raise ValueError('Geometry statistics mismatch')

    def predict_frames(self,frames,timestamps):
        if len(frames)<8 or len(frames)!=len(timestamps):raise ValueError('Capture at least eight frames')
        started=time.perf_counter();shape=frames[0].shape;observations=[];xyz=[];present=[];blend=[];previous=-1
        with ExitStack() as resources:
            hands=resources.enter_context(HandTracker(self.hand_model))
            pose_options=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(self.pose_model),delegate=mp.tasks.BaseOptions.Delegate.CPU),running_mode=mp.tasks.vision.RunningMode.VIDEO,num_poses=1,output_segmentation_masks=False,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
            face_options=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(self.face_model),delegate=mp.tasks.BaseOptions.Delegate.CPU),running_mode=mp.tasks.vision.RunningMode.VIDEO,num_faces=1,output_face_blendshapes=True,min_face_detection_confidence=.5,min_face_presence_confidence=.5,min_tracking_confidence=.5)
            pose=mp.tasks.vision.PoseLandmarker.create_from_options(pose_options);resources.callback(pose.close)
            face=mp.tasks.vision.FaceLandmarker.create_from_options(face_options);resources.callback(face.close)
            for frame,timestamp in zip(frames,timestamps,strict=True):
                if frame.shape!=shape or timestamp<=previous:raise ValueError('Frame dimensions or timestamps invalid')
                previous=timestamp;hand_result=hands.detect(frame,timestamp)
                rgb=np.ascontiguousarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                observations.append(frame_observation(hand_result,pose.detect_for_video(image,timestamp)))
                result=face.detect_for_video(image,timestamp);points=np.zeros((478,3),np.float32);scores=np.zeros(52,np.float32);detected=bool(result.face_landmarks)
                if detected:
                    if len(result.face_landmarks)!=1 or len(result.face_landmarks[0])!=478 or len(result.face_blendshapes)!=1 or len(result.face_blendshapes[0])!=52:raise ValueError('Unexpected face output')
                    categories=result.face_blendshapes[0]
                    if [v.category_name for v in categories]!=self.config['blendshape_names']:raise ValueError('Blendshape order differs')
                    points[:]=[(v.x,v.y,v.z) for v in result.face_landmarks[0]];scores[:]=[v.score for v in categories]
                xyz.append(points);present.append(detected);blend.append(scores)
        arrays={key:np.stack([item[key] for item in observations]) for key in observations[0]}
        joined,quality=normalize_joined(arrays,np.stack(xyz),np.asarray(present),np.stack(blend),shape)
        if not quality['frames_with_hands'] or not quality['frames_with_body_anchor']:raise ValueError('Keep your hands and shoulders visible')
        if not quality['frames_with_face']:raise ValueError('Keep your face visible')
        x=torch.from_numpy(joined).unsqueeze(0).to(self.device);lengths=torch.tensor([len(joined)],dtype=torch.long)
        with torch.inference_mode():logits=self.model.score(x,lengths)
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite prediction')
        probabilities=logits.softmax(-1);idx=int(probabilities.argmax(-1).item())
        return {'top1':{'label':self.names[idx],'score':float(probabilities[0,idx])},'quality':quality,'input_frames':len(joined),
                'captured_frames':len(frames),'capture_seconds':(timestamps[-1]-timestamps[0])/1000,
                'tracking_summary':hand_visibility_summary(arrays['hand_present']),'processing_seconds':time.perf_counter()-started}
