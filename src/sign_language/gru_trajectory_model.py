"""Mean-pool GRU with 17 hand and six wrist-trajectory summary features."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

from sign_language.gru_geometry_model import SignGeometryMeanPoolGRU
from sign_language.gru_mean_pool_model import masked_temporal_mean
from sign_language.trajectory_geometry import FEATURES,trajectory_batch


class SignTrajectoryMeanPoolGRU(SignGeometryMeanPoolGRU):
    def __init__(self,input_size=254,hidden_size=256,layers=2,classes=15,dropout=.5):
        super().__init__(input_size,hidden_size,layers,classes,dropout)
        original = self.classifier
        with torch.random.fork_rng(devices=[]):
            self.classifier = nn.Linear(hidden_size+len(FEATURES),classes)
        with torch.no_grad():
            self.classifier.weight.zero_()
            self.classifier.weight[:,:original.in_features].copy_(original.weight)
            self.classifier.bias.copy_(original.bias)
        self.geometry_median = torch.zeros(len(FEATURES))
        self.geometry_mean = torch.zeros(len(FEATURES))
        self.geometry_scale = torch.ones(len(FEATURES))

    def set_geometry_statistics(self,statistics):
        values = {n:torch.as_tensor(statistics[n],dtype=torch.float32) for n in ('median','mean','scale')}
        if any(v.shape!=(len(FEATURES),) or not torch.isfinite(v).all() for v in values.values()) or torch.any(values['scale']<=0):
            raise ValueError('Invalid trajectory geometry statistics')
        for name,value in values.items():
            getattr(self,'geometry_'+name).copy_(value)

    def standardize_geometry(self,geometry):
        if geometry.ndim!=2 or geometry.shape[1]!=len(FEATURES) or torch.isinf(geometry).any():
            raise ValueError('Invalid 23-feature geometry matrix')
        return (torch.where(torch.isnan(geometry),self.geometry_median,geometry)-self.geometry_mean)/self.geometry_scale

    def forward(self,frames,lengths,geometry=None,zero_geometry=False):
        packed=pack_padded_sequence(frames,lengths.cpu(),batch_first=True,enforce_sorted=False)
        packed_outputs,_=self.gru(packed)
        outputs,actual_lengths=pad_packed_sequence(packed_outputs,batch_first=True)
        pooled=self.dropout(masked_temporal_mean(outputs,actual_lengths))
        if zero_geometry:
            explicit=torch.zeros((len(frames),len(FEATURES)),device=frames.device)
        else:
            if geometry is None:geometry=trajectory_batch(frames,lengths)
            raw=torch.as_tensor(geometry,dtype=torch.float32,device=frames.device)
            if raw.shape[0]!=len(frames):raise ValueError('Geometry batch differs')
            explicit=self.standardize_geometry(raw)
        return self.classifier(torch.cat([pooled,explicit],dim=1))
