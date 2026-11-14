"""Inference adapter for the notebook-trained 700-class frozen-GRU linear head."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sign_language.gru_trajectory_model import SignTrajectoryMeanPoolGRU
from sign_language.preprocessing import CONFIG, normalize_sequence
from sign_language.recognition import SignRecognizer
from sign_language.trajectory_geometry import FEATURES, trajectory_batch


def _sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def limit_frames(features, maximum=83):
    """Match the notebook's second uniform sampling after float16 cache rounding."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 254 or len(features) < 1:
        raise ValueError('Invalid normalized landmark sequence')
    if len(features) > maximum:
        indices = torch.linspace(0, len(features)-1, maximum).round().long().numpy()
        features = features[indices]
    return features


class Fitted700Recognizer(SignRecognizer):
    def __init__(self, backbone, head, models, device='cuda'):
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        self.checkpoint = Path(backbone); self.head_checkpoint = Path(head)
        self.checkpoint_sha256 = _sha(self.checkpoint); self.head_sha256 = _sha(self.head_checkpoint)
        state = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        classifier = torch.load(self.head_checkpoint, map_location='cpu', weights_only=True)
        self.config = state['config']; self.labels = state['labels']; self.epoch = state['epoch']
        self.head_epoch = classifier['selected_epoch']
        if classifier['architecture'] != 'linear' or classifier['backbone_sha256'] != self.checkpoint_sha256:
            raise ValueError('Classifier and backbone do not match')
        if classifier['labels'] != self.labels or sorted(self.labels.values()) != list(range(700)):
            raise ValueError('Classifier label order differs')
        if self.config['model'] != {'input_size':254,'hidden_size':256,'layers':2,'classes':700,'dropout':.5}:
            raise ValueError('Unexpected backbone architecture')
        if self.config['geometry_features'] != list(FEATURES):
            raise ValueError('Unexpected geometry schema')
        pre = dict(self.config['preprocessing']); sampling = pre.pop('sampling'); maximum = pre.pop('max_frames')
        expected = dict(CONFIG); expected.pop('sampling'); expected.pop('max_frames')
        if pre != expected or maximum != 83 or self.config.get('source_cache_max_frames') != 128:
            raise ValueError('Unexpected landmark preprocessing configuration')
        self.names = [name for name, _ in sorted(self.labels.items(), key=lambda kv: kv[1])]
        self.hand_model = Path(models)/'hand_landmarker.task'; self.pose_model = Path(models)/'pose_landmarker_lite.task'
        if not self.hand_model.is_file() or not self.pose_model.is_file(): raise FileNotFoundError('Missing MediaPipe task')
        torch.backends.cudnn.enabled = False
        self.model = SignTrajectoryMeanPoolGRU(**self.config['model']).to(self.device).eval()
        self.model.load_state_dict(state['model'], strict=True)
        for key in ('mean','median','scale'):
            actual = getattr(self.model,'geometry_'+key)
            expected = torch.tensor(self.config['geometry_statistics'][key],device=self.device,dtype=torch.float32)
            if not torch.equal(actual,expected): raise ValueError('Geometry normalization mismatch')
        self.head = nn.Linear(279,700).to(self.device).eval()
        self.head.load_state_dict({k.removeprefix('output.'):v for k,v in classifier['head'].items()},strict=True)

    def score_prepared(self, features):
        """Features are an unpadded, already cache-rounded and limited sequence."""
        x = torch.as_tensor(features,dtype=torch.float32,device=self.device).unsqueeze(0)
        return self.score_batch(x, torch.tensor([len(features)],dtype=torch.long))

    def score_batch(self, x, lengths):
        x = x.to(self.device,dtype=torch.float32)
        lengths = lengths.cpu().long()
        if x.ndim != 3 or x.shape[2] != 254 or x.shape[1] > 83 or len(x) != len(lengths):
            raise ValueError('Invalid prepared batch')
        with torch.inference_mode():
            with torch.autocast('cuda',dtype=torch.float16,enabled=self.device.type=='cuda'):
                packed = pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
                output,_ = self.model.gru(packed)
                values,actual = pad_packed_sequence(output,batch_first=True)
                mask = (torch.arange(values.shape[1],device=values.device)[None,:] < actual.to(values.device)[:,None]).unsqueeze(-1)
                pooled = ((values*mask).sum(1)/actual.to(values.device).clamp_min(1)[:,None]).float()
            geometry = torch.as_tensor(trajectory_batch(x,lengths),device=self.device,dtype=torch.float32)
            inputs = torch.cat((pooled,self.model.standardize_geometry(geometry)),dim=1)
            logits = self.head(inputs)
        if not torch.isfinite(logits).all(): raise ValueError('Nonfinite prediction')
        return logits

    def predict_observations(self, arrays, frame_shape):
        features, quality = normalize_sequence(arrays,frame_shape)
        if not quality['frames_with_hands'] or not quality['frames_with_body_anchor']:
            raise ValueError('Hands and shoulders must be visible')
        features = limit_frames(features.astype(np.float16).astype(np.float32))
        scores = self.score_prepared(features).softmax(-1)
        index = int(scores.argmax(-1).item())
        return {'top1': {'label':self.names[index], 'score':float(scores[0,index])},
                'quality':quality, 'input_frames':len(features)}
