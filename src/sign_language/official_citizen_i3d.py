"""Read-only adapter for Microsoft's released ASL Citizen I3D baseline."""
import hashlib
import csv
import math
import importlib.util
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import numpy as np
import cv2


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def aggregate_frame_logits(logits,frames=64):
    """Exact official test readout: interpolate per-frame logits, then max."""
    if logits.ndim!=3 or not torch.isfinite(logits).all():raise ValueError('Invalid per-frame logits')
    return F.interpolate(logits,size=frames,mode='linear',align_corners=False).max(dim=2).values


def preprocess_captured_frames(frames,dataset_module,transforms,seed=0):
    """RAM equivalent of upstream video loader/pad/crop; validated against it."""
    total=len(frames)
    if total<8:raise ValueError('Need at least eight captured frames')
    skip=3 if total>=160 else 2 if total>=96 else 1
    start=int(np.clip((total-192)//2,0,160)) if skip==3 else int(np.clip((total-128)//2,0,96)) if skip==2 else int(np.clip((total-64)//2,0,64))
    selected=[]
    for i in range(start,min(start+64*skip,total),skip):
        image=frames[i];w,h,_=image.shape  # Upstream names height w and width h.
        if w<226 or h<226:
            scale=1+(226.-min(w,h))/min(w,h);image=cv2.resize(image,dsize=(0,0),fx=scale,fy=scale)
        if w>256 or h>256:image=cv2.resize(image,(math.ceil(w*(256/w)),math.ceil(h*(256/h))))
        selected.append((image/255.)*2-1)
    array=np.asarray(selected,np.float32)
    state=np.random.get_state()
    try:
        np.random.seed(seed);array=dataset_module.ASLCitizen.pad(None,array,64)
    finally:np.random.set_state(state)
    array=transforms.CenterCrop(224)(array)
    tensor=dataset_module.video_to_tensor(array)
    return tensor,{'captured_frames':total,'selected_before_padding':len(selected),'source_start_frame':start,
                   'frame_skip':skip,'model_frames':64,'crop_size':224,'padding_seed':seed,'channel_order':'BGR per upstream'}


class OfficialCitizenI3D:
    def __init__(self,root,device='cuda'):
        self.root=Path(root);self.folder=self.root/'models/asl-citizen-official-i3d-v1'
        self.config=json.loads((self.folder/'config.json').read_text())
        self.checkpoint=self.folder/'ASL_citizen_I3D_weights.pt'
        if sha(self.checkpoint)!=self.config['checkpoint_sha256']:raise ValueError('Official checkpoint hash differs')
        upstream=self.root/self.config['upstream_directory']
        for name,h in self.config['source_sha256'].items():
            if sha(upstream/name)!=h:raise ValueError('Upstream source changed: '+name)
        self.arch=source_module('_citizen_official_i3d_arch',upstream/'I3D/pytorch_i3d.py')
        self.dataset_module=source_module('_citizen_official_i3d_dataset',upstream/'I3D/aslcitizen_dataset.py')
        self.transforms=source_module('_citizen_official_i3d_transforms',upstream/'I3D/videotransforms.py')
        self.labels=json.loads((self.folder/'labels.json').read_text())
        train=self.root/'data/asl_citizen/ASL_Citizen/splits/train.csv'
        if sha(train)!=self.config['train_csv_sha256']:raise ValueError('Official training CSV changed')
        with train.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.reader(stream);next(reader);expected={g:i for i,g in enumerate(sorted({row[2].strip() for row in reader}))}
        if self.labels!=expected:raise ValueError('Official gloss-to-index mapping differs')
        if sorted(self.labels.values())!=list(range(2731)):raise ValueError('Unexpected official label map')
        self.names=[g for g,i in sorted(self.labels.items(),key=lambda item:item[1])]
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required; no silent fallback')
        self.device=torch.device(device)
        torch.backends.cudnn.enabled=True;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        self.model=self.arch.InceptionI3d(400,in_channels=3)
        self.model.replace_logits(2731)
        self.model.load_state_dict(torch.load(self.checkpoint,map_location='cpu',weights_only=True),strict=True)
        self.model.to(self.device).eval()

    def dataset(self,csv_path):
        videos=str(self.root/'data/asl_citizen/ASL_Citizen/videos')+'/'
        return self.dataset_module.ASLCitizen(videos,self.transforms.CenterCrop(224),str(csv_path),gloss_dict=self.labels)

    def logits(self,frames):
        if frames.ndim!=5 or tuple(frames.shape[1:])!=(3,64,224,224):raise ValueError('Expected [B,3,64,224,224] official input')
        with torch.inference_mode():
            # Official test uses float32, without AMP, and pretrained=False on the fully trained recognizer.
            return aggregate_frame_logits(self.model(frames.to(self.device,dtype=torch.float32),pretrained=False),64).cpu().numpy()
