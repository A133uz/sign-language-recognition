"""Diagnostic-only timing and full-context preprocessing; no production changes.

The unsampled adapter mirrors the frozen preprocessor and is checked byte for
byte at float16 precision before use. Full-capture context includes future
frames relative to a prefix: this is a retrospective control, not a streaming
normalizer or a proposed deployed preprocessing change.
"""
import hashlib
from pathlib import Path

import numpy as np

from sign_language import preprocessing
from sign_language.frame_rate import FrameRateLimiter

REFERENCE_SHA='51deb87625ecb587372d7439792f2892f729b3a29e03f93348f5290562a53988'
CONFIG={'version':'capture-timing-normalization-v1','target_fps':30.,
        'stage_units':'milliseconds','normalization_context':'full capture before cropping and 128-frame sampling',
        'limits':'Arrival times are application delivery, not hardware exposure timestamps. No synthetic frames or limiter changes.'}


def sample_features(features):
    x=np.asarray(features,dtype=np.float32)
    if len(x)>128:x=x[np.linspace(0,len(x)-1,128).round().astype(int)]
    return x.astype(np.float16).astype(np.float32)


def normalize_all_frames(arrays,shape):
    """Mirror frozen normalization, omitting only its final <=128 selection."""
    if hashlib.sha256(Path(preprocessing.__file__).read_bytes()).hexdigest()!=REFERENCE_SHA:
        raise ValueError('Frozen preprocessing changed; revalidate the diagnostic adapter')
    pose=np.asarray(arrays['pose_xyz'],dtype=np.float32)[...,:2].copy()
    mask=np.asarray(arrays['pose_valid_mask'],dtype=bool).copy()
    if not len(pose) or len(shape)!=3 or min(shape[:2])<=0:raise ValueError('Invalid observation dimensions')
    aspect=shape[0]/shape[1];pose[...,1]*=aspect
    if not np.isfinite(pose).all() or not np.isfinite(arrays['hands_xyz']).all():raise ValueError('Nonfinite coordinates')
    hands,present=preprocessing.canonical_hands(arrays,pose,mask,aspect)
    center=(pose[:,11]+pose[:,12])/2
    widths=np.linalg.norm(pose[:,11]-pose[:,12],axis=1)
    anchors=mask[:,11]&mask[:,12]&(widths>=.02)
    scale=float(np.median(widths[anchors])) if anchors.any() else 1.
    anchor_valid=anchors.copy();good=np.flatnonzero(anchors)
    for left,right in zip(good[:-1],good[1:]):
        if 1<right-left<=preprocessing.CONFIG['max_anchor_gap']+1:
            alpha=np.arange(1,right-left,dtype=np.float32)/(right-left)
            center[left+1:right]=center[left]*(1-alpha[:,None])+center[right]*alpha[:,None]
            anchor_valid[left+1:right]=True
    body_pose=(pose[:,preprocessing.POSE_POINTS]-center[:,None])/scale
    pose_valid=mask[:,preprocessing.POSE_POINTS]&anchor_valid[:,None];body_pose[~pose_valid]=0
    body_hands=(hands[...,:2]-center[:,None,None])/scale
    body_hands[~(present&anchor_valid[:,None])]=0
    local=hands-hands[:,:,:1]
    palm=np.linalg.norm(local[:,:,[5,9,13,17],:2],axis=-1).mean(axis=-1)
    local_valid=np.zeros_like(present);palm_scales=[]
    for slot in (0,1):
        usable=present[:,slot]&(palm[:,slot]>=.005)
        palm_scale=float(np.median(palm[usable,slot])) if usable.any() else None
        palm_scales.append(palm_scale)
        if usable.any():
            local[:,slot]/=palm_scale;local_valid[:,slot]=usable
        local[~usable,slot]=0
    x=np.concatenate([body_pose.reshape(len(pose),-1),pose_valid.astype(np.float32),
        body_hands.reshape(len(pose),-1),local.reshape(len(pose),-1),present.astype(np.float32),
        local_valid.astype(np.float32),anchor_valid[:,None].astype(np.float32)],axis=1)
    x=np.clip(x,-preprocessing.CONFIG['clip_features'],preprocessing.CONFIG['clip_features']).astype(np.float32)
    if x.shape!=(len(pose),254) or not np.isfinite(x).all():raise ValueError('Invalid unsampled normalized features')
    quality={'frames':len(pose),'frames_with_hands':int(present.any(1).sum()),'frames_with_body_anchor':int(anchor_valid.sum())}
    return x,{'shoulder_median_width':scale,'palm_median_sizes':palm_scales,'quality':quality}


def verify_adapter(arrays,shape):
    all_frames,context=normalize_all_frames(arrays,shape)
    expected,quality=preprocessing.normalize_sequence(arrays,shape)
    if not np.array_equal(sample_features(all_frames),sample_features(expected)) or context['quality']!=quality:
        raise AssertionError('Diagnostic adapter does not reproduce the frozen preprocessor')
    return all_frames,context


def feature_difference(fixed,refitted):
    a=np.asarray(fixed);b=np.asarray(refitted)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=254:raise ValueError('Windows must contain the same sampled frames')
    result={}
    for name,first,last in (('body_pose',0,26),('body_hands',39,123),('local_hands',123,249)):
        delta=np.asarray(a[:,first:last],float)-b[:,first:last]
        result[name]={'rms_difference':float(np.sqrt(np.mean(delta**2))),'max_abs_difference':float(np.max(np.abs(delta)))}
    ids=list(range(26,39))+list(range(249,254))
    result['mask_disagreement_fraction']=float(np.mean(a[:,ids]!=b[:,ids]))
    return result


def distribution(values):
    x=np.asarray(values,dtype=float)
    if not len(x):return {'count':0}
    if not np.isfinite(x).all():raise ValueError('Nonfinite timing values')
    return {'count':len(x),'min':float(x.min()),'p50':float(np.median(x)),
            'p95':float(np.quantile(x,.95)),'max':float(x.max()),'mean':float(x.mean())}


class CaptureTiming:
    """Keep a brief timing trace in RAM; serialize only aggregate statistics."""
    def __init__(self,fps=30.):
        self.fps=fps;self.arrivals=[];self.kept=[];self.stages=[];self.credit=[]

    def record(self,seconds,kept,stages,credit_after):
        if not np.isfinite(seconds) or seconds<0 or (self.arrivals and seconds<self.arrivals[-1]):
            raise ValueError('Invalid arrival order')
        if any(not np.isfinite(v) or v<0 for v in stages.values()) or not np.isfinite(credit_after):
            raise ValueError('Invalid stage timing or limiter credit')
        self.arrivals.append(float(seconds));self.kept.append(bool(kept));self.stages.append(dict(stages));self.credit.append(float(credit_after))

    def summary(self):
        arrivals=np.asarray(self.arrivals);kept=np.asarray(self.kept,bool)
        intervals=np.diff(arrivals)*1000.;retained=arrivals[kept]
        reset=np.r_[False,np.diff(arrivals)>2/self.fps+1.e-9] if len(arrivals) else np.array([],bool)
        replay=FrameRateLimiter(self.fps);decisions=np.array([replay.accept(float(t)) for t in arrivals],bool)
        since_reset=float('inf');near_reset_drops=0
        for i,t in enumerate(arrivals):
            if reset[i]:since_reset=t
            if not kept[i] and 0<=t-since_reset<=.1:near_reset_drops+=1
        edges=np.array([0,5,10,20,30,40,50,67,100,200,500,np.inf])
        histogram=np.histogram(intervals,bins=edges)[0]
        return {'arrival_count':len(arrivals),'kept_count':int(kept.sum()),'dropped_count':int((~kept).sum()),
                'arrival_interval_ms':distribution(intervals),'kept_interval_ms':distribution(np.diff(retained)*1000.),
                'drop_preceding_interval_ms':distribution(intervals[~kept[1:]]) if len(kept)>1 else {'count':0},
                'arrival_interval_histogram_ms':{f'{edges[i]:g}..{edges[i+1]:g}':int(v) for i,v in enumerate(histogram)},
                'stages_ms':{k:distribution([s[k] for s in self.stages if k in s]) for k in sorted({k for s in self.stages for k in s})},
                'limiter_long_gap_reset_count':int(reset.sum()),'drops_within_100ms_after_long_gap':near_reset_drops,
                'limiter_credit_after_drop':distribution(np.asarray(self.credit)[~kept]),
                'limiter_replay_mismatches':int((decisions!=kept).sum()),
                'raw_timing_trace_saved':False,'hardware_frame_timestamps_available':False}


def diagnose_timing_capture(current,baseline,frames,timestamps):
    from sign_language.pronoun_diagnostics import diagnose_observations
    arrays,shape,metadata=current.observe_frames(frames,timestamps)
    return {**metadata,**diagnose_observations(current,baseline,arrays,shape,timestamps,fixed_normalization=True),
            'recognition_tracking_passes':1,'media_saved':False,'coordinates_saved':False,'embeddings_saved':False}
