"""Frozen ME/WE controls. Persist only scores and aggregate measurements.

An intervention establishes sensitivity, not tracking ground truth or a fix.
Expected labels never enter these inference functions. All 15 classes compete.
"""
import numpy as np
import torch

from sign_language.preprocessing import normalize_sequence,canonical_hands
from sign_language.recognition_forensics import infer_clips,make_variant,feature_summary
from sign_language.recognition_diagnostics import observation_statistics,hand_visibility_summary
from sign_language.trajectory_geometry import FEATURES

CURRENT_SHA='34b766c393cf20a77b001004376195e60ffb58943d7b1f9661bd23edbab81c91'
BASELINE_SHA='0921e10a193044e5cbc92e29b386b370b827431b6850f0bc49f6e64ea4441843'
VARIANTS=('original','reverse_time','shuffle_time','static_active_frame','reflect_anatomy','swap_hand_slots')
PROMPTS=('ME','WE','PLEASE')*3


def check_models(current,baseline):
    if current.checkpoint_sha256!=CURRENT_SHA or baseline.checkpoint_sha256!=BASELINE_SHA:
        raise ValueError('Use the declared frozen checkpoints')
    if current.labels!=baseline.labels or len(current.names)!=15 or current.config['cache_id']!=baseline.config['cache_id']:
        raise ValueError('Both models must share the official 15-label cache')
    if current.model.classifier.in_features!=279 or baseline.model.classifier.in_features!=273:
        raise ValueError('Expected 256 pooled + 23/17 geometry features')
    if not all(torch.equal(p.cpu(),baseline.model.gru.state_dict()[n].cpu())
               for n,p in current.model.gru.state_dict().items()):
        raise ValueError('Comparison requires the identical recurrent backbone')


def score(logits,names):
    z=np.asarray(logits,dtype=np.float64)
    if z.shape!=(len(names),) or not np.isfinite(z).all():raise ValueError('Invalid classifier logits')
    p=np.exp(z-z.max());p/=p.sum();ids=np.argsort(-p,kind='stable')[:5]
    return {'top5':[{'label':names[i],'score':float(p[i])} for i in ids],
            'scores':p.tolist(),'we_minus_me_margin':float(z[names.index('WE')]-z[names.index('ME')])}


def explain_head(model,joined,logits,names):
    width=model.gru.hidden_size
    x=np.asarray(joined,dtype=np.float64)
    w=model.classifier.weight.detach().float().cpu().numpy().astype(np.float64)
    b=model.classifier.bias.detach().float().cpu().numpy().astype(np.float64)
    if width!=256 or x.shape!=(279,) or w.shape!=(15,279):raise ValueError('Wrong diagnostic feature groups')
    we,me=names.index('WE'),names.index('ME');delta=w[we]-w[me]
    values=delta*x
    parts={'bias':float(b[we]-b[me]),'pooled_gru':float(values[:width].sum()),
           'original_17_geometry':float(values[width:width+17].sum()),
           'six_path_geometry':float(values[width+17:].sum())}
    actual=float(logits[we]-logits[me]);residual=actual-sum(parts.values())
    if abs(residual)>1.e-4*max(1.,abs(actual)):raise AssertionError('Head decomposition does not reconstruct logits')
    return {'positive_favors':'WE','negative_favors':'ME','actual_margin':actual,'components':parts,
            'reconstruction_residual':residual,
            'geometry_contributions':{name:float(value) for name,value in zip(FEATURES,values[width:],strict=True)}}


def recover_competitor_margin(explanation,probabilities,weights,bias,names,target,competitor):
    """Explain another class pair from saved scalar contributions, without embeddings.

    Each saved geometry contribution is (WE_weight-ME_weight)*standardized_value.
    Nonzero coefficients recover clip summaries, NOT coordinates or the GRU
    vector. The aggregate GRU contribution is the remaining logit margin. If a
    required geometry value is not identifiable, report unavailable explicitly.
    """
    w=np.asarray(weights,dtype=np.float64);b=np.asarray(bias,dtype=np.float64)
    p=np.asarray(probabilities,dtype=np.float64)
    a,c=names.index(target),names.index(competitor)
    original=w[names.index('WE'),256:]-w[names.index('ME'),256:]
    delta=w[c]-w[a];geometry=[]
    for i,name in enumerate(FEATURES):
        if original[i]==0:
            if delta[256+i]!=0:return {'available':False,'reason':'Geometry summary not identifiable from saved pair contribution','feature':name}
            geometry.append(0.)
        else:geometry.append(explanation['geometry_contributions'][name]/original[i]*delta[256+i])
    if min(p[a],p[c])<=0:return {'available':False,'reason':'Probability underflow prevents recovering a logit difference'}
    total=float(np.log(p[c])-np.log(p[a]));bias_part=float(b[c]-b[a])
    hand=float(sum(geometry[:17]));path=float(sum(geometry[17:]))
    return {'available':True,'positive_favors':competitor,'negative_favors':target,'actual_margin':total,
            'components':{'bias':bias_part,'pooled_gru':total-bias_part-hand-path,
                          'original_17_geometry':hand,'six_path_geometry':path},
            'derivation':'Geometry recovered from saved scalar pair products; aggregate GRU term is log-probability margin minus geometry and bias. No hidden vectors reconstructed.'}


def diagnose_features(current,baseline,features):
    x=np.asarray(features,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=254 or not 8<=len(x)<=128 or not np.isfinite(x).all():
        raise ValueError('Expected unpadded finite [8..128,254] input')
    before=x.copy()
    clips=[make_variant(x,name,None) for name in VARIANTS]
    logits,joined=infer_clips(current,clips)
    old_logits,_=infer_clips(baseline,clips)
    results={}
    for i,name in enumerate(VARIANTS):
        entry={'current':score(logits[i],current.names),'original_temporal_model':score(old_logits[i],baseline.names),
               'margin_explanation':explain_head(current.model,joined[i],logits[i],current.names)}
        # These are interventions on the SAME readout input, not refitted heads.
        interventions={}
        for key,start,end in (('original_geometry_off',256,273),('path_geometry_off',273,279),('all_geometry_off',256,279)):
            altered=joined[i].copy();altered[start:end]=0.
            with torch.inference_mode():
                z=current.model.classifier(torch.from_numpy(altered).to(current.device)).float().cpu().numpy()
            interventions[key]=score(z,current.names)
        entry['head_interventions']=interventions
        results[name]=entry
    if not np.array_equal(x,before):raise AssertionError('Input features changed')
    return {'variants':results,'feature_summary':feature_summary(x),
            'interpretation':'Destructive controls measure sensitivity, not correctness or a recommended preprocessing change.'}


def signed_motion(points,valid,timestamps):
    """Aggregate projected movement without joining missing frames or slots."""
    p=np.asarray(points,dtype=float);valid=np.asarray(valid,bool);t=np.asarray(timestamps,dtype=float)/1000.
    if p.shape!=(len(valid),2) or t.shape!=(len(valid),) or not np.isfinite(p).all() or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Invalid aligned motion/timestamps')
    ids=np.flatnonzero(valid)
    if not len(ids):return {'tracked_frames':0}
    step=np.diff(p,axis=0);dt=np.diff(t);good=valid[:-1]&valid[1:]
    kept=step[good];speed=np.linalg.norm(kept,axis=1)/dt[good]
    gaps=np.diff(np.r_[False,~valid,False].astype(int));starts=np.flatnonzero(gaps==1);ends=np.flatnonzero(gaps==-1)
    return {'tracked_frames':len(ids),'median_x':float(np.median(p[ids,0])), 'median_y':float(np.median(p[ids,1])),
            'range_x':float(np.ptp(p[ids,0])),'range_y':float(np.ptp(p[ids,1])),
            'signed_dx_sum':float(kept[:,0].sum()),'signed_dy_sum':float(kept[:,1].sum()),
            'path_length':float(np.linalg.norm(kept,axis=1).sum()),
            'median_speed_per_second':float(np.median(speed)) if len(speed) else None,
            'longest_missing_span_frames':int((ends-starts).max()) if len(starts) else 0,
            'consecutive_tracked_pairs':int(good.sum())}


def observation_summary(arrays,shape,timestamps):
    pose=arrays['pose_xyz'][...,:2].copy();aspect=shape[0]/shape[1];pose[...,1]*=aspect
    mask=arrays['pose_valid_mask'];hands,present=canonical_hands(arrays,pose,mask,aspect)
    widths=np.linalg.norm(pose[:,11]-pose[:,12],axis=1)
    anchors=mask[:,11]&mask[:,12]&(widths>=.02)
    scale=float(np.median(widths[anchors])) if anchors.any() else 1.
    centered=(hands[...,:2]-(pose[:,11]+pose[:,12])[:,None,None]/2)/scale
    per_slot=[]
    for slot in (0,1):
        valid=present[:,slot]&anchors
        per_slot.append({'slot':slot,'wrist':signed_motion(centered[:,slot,0],valid,timestamps),
                         'index_tip':signed_motion(centered[:,slot,8],valid,timestamps)})
    return {'tracking':hand_visibility_summary(arrays['hand_present']),
            'input_statistics':observation_statistics(arrays,shape),'per_slot_motion':per_slot,
            'units':'Shoulder-width projected XY; speeds use actual timestamps; gaps never joined. No shared hand/body depth.'}


def diagnose_observations(current,baseline,arrays,shape,timestamps,*,fixed_normalization=False,additional_windows=None):
    stamps=np.asarray(timestamps)
    if len(stamps)!=len(arrays['hand_present']) or len(stamps)<2 or np.any(np.diff(stamps)<=0):
        raise ValueError('Need aligned increasing timestamps')
    seen=np.flatnonzero(arrays['hand_present'].any(1))
    windows={'full':np.arange(len(stamps)),'first_two_seconds':np.flatnonzero(stamps-stamps[0]<2000)}
    windows['detected_hand_envelope']=np.arange(seen[0],seen[-1]+1) if len(seen) else np.array([],int)
    for name,(start_ms,end_ms) in (additional_windows or {}).items():
        if name in windows or not 0<=start_ms<end_ms:raise ValueError('Invalid additional diagnostic window')
        relative=stamps-stamps[0]
        windows[name]=np.flatnonzero((relative>=start_ms)&(relative<end_ms))
    results={};window_features={}
    for name,ids in windows.items():
        if len(ids)<8:
            results[name]={'status':'unavailable','reason':'Fewer than eight observed frames'};continue
        subset={key:value[ids] for key,value in arrays.items()}
        features,quality=normalize_sequence(subset,shape)
        if not quality['frames_with_hands'] or not quality['frames_with_body_anchor']:
            results[name]={'status':'unavailable','reason':'No detected hands or body anchor','quality':quality};continue
        features=features.astype(np.float16).astype(np.float32)
        window_features[name]=features
        results[name]={'status':'scored','observed_frames':len(ids),'input_frames':len(features),'quality':quality,
                       'duration_seconds':float((stamps[ids[-1]]-stamps[ids[0]])/1000.),
                       **diagnose_features(current,baseline,features)}
    normalization={}
    if fixed_normalization:
        from sign_language.capture_timing_diagnostics import verify_adapter,sample_features,feature_difference
        normalized_all,context=verify_adapter(arrays,shape)
        normalization={'full_context':context,'full_adapter_matches_production_at_float16':True,
                       'limits':'Retrospective full-capture preprocessing includes future frames for a prefix; not deployable online. Window refitting may also affect anchor interpolation/hand-slot initialization.'}
        for name,ids in windows.items():
            if name=='full':continue
            key=name+'_fixed_normalization'
            if results[name]['status']!='scored':results[key]=dict(results[name]);continue
            fixed=sample_features(normalized_all[ids])
            subset={k:v[ids] for k,v in arrays.items()}
            _,refitted_context=verify_adapter(subset,shape)
            normalization[name]={'refitted_context':refitted_context,
                                  'matched_frame_feature_differences':feature_difference(fixed,window_features[name])}
            results[key]={'status':'scored','observed_frames':len(ids),'input_frames':len(fixed),
                          'duration_seconds':results[name]['duration_seconds'],'quality':results[name]['quality'],
                          'preprocessing_context':'All frame features/identities/anchors and scale medians computed on full capture, then cropped before 128-frame sampling.',
                          **diagnose_features(current,baseline,fixed)}
    return {'windows':results,'normalization_comparison':normalization,'measurements':observation_summary(arrays,shape,stamps),
            'window_limit':'Prefix can truncate a sign; detected-hand envelope is a tracking proxy, not a verified sign boundary.'}


def diagnose_capture(current,baseline,frames,timestamps):
    arrays,shape,metadata=current.observe_frames(frames,timestamps)
    return {**metadata,**diagnose_observations(current,baseline,arrays,shape,timestamps),
            'recognition_tracking_passes':1,'media_saved':False,'coordinates_saved':False,'embeddings_saved':False}
