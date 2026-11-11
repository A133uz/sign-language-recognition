"""Frozen-model controls. Diagnostics never alter the production checkpoint/cache.

Most interventions are destructive: a changed prediction establishes sensitivity,
not correctness or a recommended preprocessing change. Live arrays stay in memory.
"""
import numpy as np
import torch

from sign_language.preprocessing import POSE_POINTS

VARIANTS = ('original', 'reverse_time', 'shuffle_time', 'static_active_frame',
            'last_8', 'last_16', 'last_32', 'first_80pct', 'middle_60pct', 'last_80pct',
            'resample_64', 'resample_128', 'swap_hand_slots', 'reflect_anatomy',
            'zero_pose', 'zero_face', 'zero_pose_arms', 'zero_local_hands',
            'zero_hand_depth', 'zero_body_hands', 'training_rest_pose_tail',
            'training_rest_pose_all', 'training_rest_tail_16', 'training_rest_tail_32')


def phase_lengths(clip):
    ids=np.flatnonzero(clip[:,249:251].any(1))
    return [int(ids[0]),int(ids[-1]-ids[0]+1),int(len(clip)-ids[-1]-1)] if len(ids) else None


def match_training_phases(clip,desired,motion_only=False):
    """Resample real pre/motion/post spans; never invent absent neutral poses."""
    lengths=phase_lengths(clip)
    if lengths is None or min(lengths)==0:return clip.copy(),False
    a,b,c=lengths;parts=[clip[:a],clip[a:a+b],clip[a+b:]]
    targets=[a,int(desired[1]),c] if motion_only else desired
    pieces=[part[np.linspace(0,len(part)-1,int(n)).round().astype(int)] for part,n in zip(parts,targets)]
    x=np.concatenate(pieces)
    if len(x)>128:x=x[np.linspace(0,len(x)-1,128).round().astype(int)]
    return x.copy(),True


def feature_summary(clip):
    """Aggregate comparisons only; no per-frame positions returned."""
    present=clip[:,249:251]>0
    seen=present.any(1); ids=np.flatnonzero(seen)
    single=present.sum(1)==1
    switched=single[1:] & single[:-1] & (present[1:].argmax(1)!=present[:-1].argmax(1))
    pose_mask=clip[:,26:39]>0
    return {'frames':len(clip), 'hand_frame_fraction':float(seen.mean()),
            'left_hand_fraction':float(present[:,0].mean()),
            'right_hand_fraction':float(present[:,1].mean()),
            'first_hand_fraction':float(ids[0]/len(clip)) if len(ids) else None,
            'last_hand_fraction':float(ids[-1]/len(clip)) if len(ids) else None,
            'hand_fraction_by_quarter':[float(x.mean()) for x in np.array_split(seen,4)],
            'single_hand_slot_changes':int(switched.sum()),
            'valid_pose_fraction':float(pose_mask.mean()),
            'tail_pose_rms':float(np.sqrt(np.mean(clip[-min(16,len(clip)):,:26]**2))),
            'clipped_fraction':float((np.abs(clip)>=7.999).mean())}


def summary_vector(clip, kind='all'):
    """Fixed aggregates for independent diagnostic classifiers."""
    if kind=='timing':
        mask=clip[:,249:251]
        values=[len(clip)/128, *mask.mean(0), *mask.std(0)]
        for quarter in np.array_split(mask,4): values.extend(quarter.mean(0))
        return np.asarray(values,np.float32)
    if kind=='pose': values=clip[:,:39]
    elif kind=='hands':
        ids=np.flatnonzero(clip[:,249:251].any(1))
        values=clip[ids,39:253] if len(ids) else np.zeros((1,214),np.float32)
    else: raise ValueError(kind)
    return np.concatenate([values.mean(0),values.std(0)]).astype(np.float32)


def make_variant(clip,name,rest):
    """Return a private array, never modify the source; rest comes from train only."""
    x=clip.copy(); n=len(x)
    if name=='original': pass
    elif name=='reverse_time': x=x[::-1].copy()
    elif name=='shuffle_time': x=x[np.random.default_rng(42).permutation(n)]
    elif name=='static_active_frame':
        ids=np.flatnonzero(x[:,249:251].any(1)); i=int(ids[len(ids)//2]) if len(ids) else n//2
        x[:]=x[i].copy()
    elif name.startswith('last_') and name[5:].isdigit(): x=x[-int(name[5:]):].copy()
    elif name in ('first_80pct','middle_60pct','last_80pct'):
        a,b={'first_80pct':(0,.8),'middle_60pct':(.2,.8),'last_80pct':(.2,1)}[name]
        x=x[round(a*n):max(round(a*n)+1,round(b*n))].copy()
    elif name.startswith('resample_'):
        x=x[np.linspace(0,n-1,int(name[9:])).round().astype(int)]
    elif name in ('swap_hand_slots','reflect_anatomy'):
        x[:,39:123]=x[:,39:123].reshape(n,2,21,2)[:,::-1].reshape(n,84)
        x[:,123:249]=x[:,123:249].reshape(n,2,21,3)[:,::-1].reshape(n,126)
        x[:,249:251]=x[:,249:251][:,::-1]; x[:,251:253]=x[:,251:253][:,::-1]
        if name=='reflect_anatomy':
            pairs={0:0,2:5,5:2,7:8,8:7,9:10,10:9,11:12,12:11,13:14,14:13,15:16,16:15}
            order=[POSE_POINTS.index(pairs[p]) for p in POSE_POINTS]
            p=x[:,:26].reshape(n,13,2)[:,order].copy(); p[...,0]*=-1
            x[:,:26]=p.reshape(n,26);x[:,26:39]=x[:,26:39][:,order]
            x[:,39:123:2]*=-1;x[:,123:249:3]*=-1
    elif name=='zero_pose': x[:,:39]=0
    elif name=='zero_face': x[:,:14]=0;x[:,26:33]=0
    elif name=='zero_pose_arms': x[:,18:26]=0;x[:,35:39]=0
    elif name=='zero_local_hands': x[:,123:249]=0;x[:,251:253]=0
    elif name=='zero_hand_depth': x[:,125:249:3]=0
    elif name=='zero_body_hands': x[:,39:123]=0
    elif name in ('training_rest_pose_tail','training_rest_pose_all'):
        ids=np.flatnonzero(x[:,249:251].any(1))
        start=int(ids[-1]+1) if len(ids) else 0
        if name.endswith('_all'): start=0
        x[start:,:39]=rest[:39]; x[start:,253]=rest[253]
    elif name.startswith('training_rest_tail_'):
        count=int(name.rsplit('_',1)[1]);x=np.concatenate([x,np.repeat(rest[None],count,axis=0)])
        if len(x)>128:x=x[np.linspace(0,len(x)-1,128).round().astype(int)]
    else: raise ValueError(name)
    return np.ascontiguousarray(x,dtype=np.float32)


def infer_clips(recognizer,clips):
    """Return logits/hidden arrays in memory, using the actual frozen model."""
    all_logits=[];all_hidden=[]
    for start in range(0,len(clips),16):
        subset=clips[start:start+16];batch=np.zeros((len(subset),128,254),np.float32)
        lengths=torch.tensor([len(x) for x in subset],dtype=torch.long)
        for i,x in enumerate(subset):batch[i,:len(x)]=x
        captured=[]
        hook=recognizer.model.classifier.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().float().cpu().numpy()))
        try:
            with torch.inference_mode(),torch.autocast('cuda',enabled=recognizer.device.type=='cuda'):
                logits=recognizer.model(torch.from_numpy(batch).to(recognizer.device),lengths).float()
            all_logits.append(logits.cpu().numpy());all_hidden.append(captured[0])
        finally:hook.remove()
    return np.concatenate(all_logits),np.concatenate(all_hidden)


def scores(logits,names,expected=None):
    z=logits-np.max(logits);p=np.exp(z);p/=p.sum();order=np.argsort(-p)
    result={'top5':[{'label':names[int(i)],'score':float(p[i])} for i in order]}
    if expected is not None:
        result.update(expected_score=float(p[expected]),expected_rank=int(np.flatnonzero(order==expected)[0]+1),
                      correct=bool(order[0]==expected))
    return result


def metrics(logits,targets,names):
    pred=np.argmax(logits,axis=1)
    return {'correct':int((pred==targets).sum()),'clips':len(targets),
            'predicted_counts':{n:int((pred==i).sum()) for i,n in enumerate(names)},
            'per_class_correct':{n:int(((pred==targets)&(targets==i)).sum()) for i,n in enumerate(names)}}
