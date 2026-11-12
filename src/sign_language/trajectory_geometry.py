"""Six label-independent wrist-path descriptors appended to the original 17."""
import numpy as np
import torch

from sign_language.gru_geometry_model import FEATURES as HAND_FEATURES, geometry_batch
from sign_language.wrist_trajectory import contiguous_spans

TRAJECTORY_FEATURES = ('wrist_loop_strength', 'wrist_loop_turn_coherence',
                       'wrist_loop_abs_turns', 'wrist_loop_axis_ratio',
                       'wrist_loop_endpoint_over_path', 'wrist_loop_path_length')
FEATURES = HAND_FEATURES + TRAJECTORY_FEATURES
CONFIG = {'version': 'wrist-path-window-v1', 'windows_cached_steps': [12,24,36],
          'slot': 'most body-anchored hand-present frames; deterministic first-slot tie',
          'window_selection': 'maximum circularity times (1-endpoint/path); no label or score input',
          'gaps': 'never bridge absent hand/anchor frames', 'smoothing': None,
          'units': 'body-relative XY; original shoulder-width normalization',
          'limits': 'heuristic estimated projected motion, not anatomical or semantic ground truth'}


def best_window_features(points, valid):
    p = np.asarray(points, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if p.shape != (len(mask),2) or not np.isfinite(p).all():
        raise ValueError('Invalid aligned wrist trajectory')
    best_score, best = -1., None
    for ids in contiguous_spans(mask):
        segment = p[ids]
        for count in CONFIG['windows_cached_steps']:
            if len(segment) < count:
                continue
            w = np.lib.stride_tricks.sliding_window_view(segment,count,axis=0).transpose(0,2,1)
            steps = np.diff(w,axis=1)
            distances = np.linalg.norm(steps,axis=2)
            length = distances.sum(1)
            gap = np.linalg.norm(w[:,-1]-w[:,0],axis=1)
            closure = np.divide(gap,length,out=np.ones_like(length),where=length>1.e-8).clip(0,1)
            area = .5*np.abs((w[:,:,0]*np.roll(w[:,:,1],-1,axis=1)-w[:,:,1]*np.roll(w[:,:,0],-1,axis=1)).sum(1))
            perimeter = length+gap
            circularity = np.divide(4*np.pi*area,perimeter**2,out=np.zeros_like(length),where=perimeter>1.e-8).clip(0,1)
            score = circularity*(1-closure)
            index = int(score.argmax())
            if score[index] <= best_score:
                continue
            selected = w[index];d = steps[index];good = distances[index]>1.e-5
            turns = np.arctan2(d[:-1,0]*d[1:,1]-d[:-1,1]*d[1:,0],(d[:-1]*d[1:]).sum(1))
            turns = turns[good[:-1]&good[1:]]
            total = np.abs(turns).sum()
            centered = selected-selected.mean(0)
            xx, yy = (centered**2).sum(0);xy = (centered[:,0]*centered[:,1]).sum()
            delta = np.hypot(xx-yy,2*xy)
            major, minor = (xx+yy+delta)/2, max(0.,(xx+yy-delta)/2)
            ratio = np.sqrt(minor/major) if major>1.e-8 else 0.
            best_score = float(score[index])
            best = np.asarray([best_score,abs(turns.sum())/total if total>1.e-8 else 0.,
                               abs(turns.sum())/(2*np.pi),ratio,closure[index],length[index]],dtype=np.float32)
    return best if best is not None else np.full(6,np.nan,dtype=np.float32)


def trajectory_batch(frames,lengths):
    x = frames.detach().float().cpu().numpy()
    counts = lengths.detach().cpu().numpy()
    if x.ndim!=3 or x.shape[2]!=254 or counts.shape!=(len(x),):
        raise ValueError('Expected [batch,time,254] and lengths')
    rows = []
    for padded,count in zip(x,counts,strict=True):
        if not np.isfinite(count) or count!=int(count) or not 1<=count<=len(padded):
            raise ValueError('Invalid real sequence length')
        sequence = padded[:int(count)]
        valid = (sequence[:,249:251]>0)&(sequence[:,253,None]>0)
        slot = int(valid.sum(0).argmax())
        wrist = sequence[:,39:123].reshape(-1,2,21,2)[:,slot,0]
        rows.append(best_window_features(wrist,valid[:,slot]))
    return np.concatenate([geometry_batch(frames,lengths),np.stack(rows)],axis=1)


def fit_statistics(training):
    x = np.asarray(training,dtype=float)
    if x.ndim!=2 or x.shape[1]!=len(FEATURES) or not len(x) or np.isinf(x).any():
        raise ValueError('Invalid training feature matrix')
    median = np.asarray([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in x.T])
    values = np.where(np.isnan(x),median,x)
    scale = values.std(0);scale[scale<1.e-8]=1.
    return {'median':median.tolist(),'mean':values.mean(0).tolist(),'scale':scale.tolist()}
