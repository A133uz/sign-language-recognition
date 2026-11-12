"""Frozen recurrent backbone with a refitted, float32 linear 15-class readout."""
import torch
from torch import nn
from sign_language.gru_trajectory_model import SignTrajectoryMeanPoolGRU


class Float32Linear(nn.Linear):
    def forward(self,values):
        with torch.autocast(device_type=values.device.type,enabled=False):
            return nn.functional.linear(values.float(),self.weight.float(),self.bias.float())


class SignTrajectoryReadoutGRU(SignTrajectoryMeanPoolGRU):
    def __init__(self,input_size=254,hidden_size=256,layers=2,classes=15,dropout=.5):
        super().__init__(input_size,hidden_size,layers,classes,dropout)
        original=self.classifier
        with torch.random.fork_rng(devices=[]):
            self.classifier=Float32Linear(original.in_features,original.out_features)
        self.classifier.load_state_dict(original.state_dict())
        self.gru.requires_grad_(False)
