"""Scalar-only articulation diagnostics for the exact normalized model input.

Values are estimated geometry, not anatomical ground truth. No joint arrays are
returned. Dominant-slot comparisons include detected resting hand frames.
"""
import numpy as np


def quantiles(values):
    a=np.asarray(values);a=a[np.isfinite(a)]
    return {'p10':float(np.quantile(a,.1)),'median':float(np.median(a)),
            'p90':float(np.quantile(a,.9))} if len(a) else None


def angle_values(points,joints):
    a,b,c=joints;u=points[:,a]-points[:,b];v=points[:,c]-points[:,b]
    denominator=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
    good=denominator>1e-6
    return np.degrees(np.arccos(np.clip((u[good]*v[good]).sum(1)/denominator[good],-1,1)))


def summarize_articulation(features):
    x=np.asarray(features,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=254 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite [time,254] features')
    valid=(x[:,249:251]>0)&(x[:,251:253]>0)
    slot=int(valid.sum(0).argmax());ids=np.flatnonzero(valid[:,slot])
    presence=x[:,249:251]>0;single=presence.sum(1)==1
    switched=single[1:]&single[:-1]&(presence[1:].argmax(1)!=presence[:-1].argmax(1))
    result={'frames':len(x),'dominant_slot':slot,'usable_frames':len(ids),
            'usable_fraction':float(len(ids)/len(x)),'single_hand_slot_changes':int(switched.sum()),
            'insufficient':len(ids)<2,'metrics':{},'quarters':[]}
    if len(ids)<2:return result
    points=x[:,123:249].reshape(-1,2,21,3)[ids,slot]
    for name,tip in [('thumb_index',8),('thumb_middle',12)]:
        delta=points[:,tip]-points[:,4]
        for dim,suffix in [(2,'xy'),(3,'xyz')]:
            d=np.linalg.norm(delta[:,:dim],axis=1);q=quantiles(d)
            result['metrics'][name+'_'+suffix+'_median']=q['median']
            result['metrics'][name+'_'+suffix+'_range']=q['p90']-q['p10']
    for name,tip in [('index',8),('middle',12)]:
        result['metrics'][name+'_tip_wrist_median']=quantiles(np.linalg.norm(points[:,tip],axis=1))['median']
    for name,joints in [('index_pip',(5,6,7)),('middle_pip',(9,10,11))]:
        q=quantiles(angle_values(points,joints))
        result['metrics'][name+'_angle_median_deg']=q['median'] if q else None
        result['metrics'][name+'_angle_range_deg']=q['p90']-q['p10'] if q else None
    # Orientation proxy: wrist -> middle MCP direction; do not bridge tracking gaps.
    axis=points[:,9]-points[:,0];usable=np.linalg.norm(axis[:,:2],axis=1)>1e-6
    ranges=[];eligible_pairs=0
    segments=np.split(np.arange(len(ids)),np.flatnonzero((np.diff(ids)!=1)|(~usable[1:])|(~usable[:-1]))+1)
    for group in segments:
        group=group[usable[group]]
        if len(group)<2:continue
        angle=np.degrees(np.unwrap(np.arctan2(axis[group,1],axis[group,0])))
        q=quantiles(angle);ranges.append(q['p90']-q['p10']);eligible_pairs+=len(group)-1
    result['metrics']['palm_axis_xy_excursion_deg']=float(max(ranges)) if ranges else None
    result['orientation_contiguous_pairs']=eligible_pairs
    norm=np.linalg.norm(axis,axis=1);good3=norm>1e-6;ranges3=[]
    segments3=np.split(np.arange(len(ids)),np.flatnonzero((np.diff(ids)!=1)|(~good3[1:])|(~good3[:-1]))+1)
    for group in segments3:
        group=group[good3[group]]
        if len(group)<2:continue
        unit=axis[group]/norm[group,None]
        angle=np.degrees(np.arccos(np.clip(unit@unit[0],-1,1)))
        q=quantiles(angle);ranges3.append(q['p90']-q['p10'])
    result['metrics']['palm_axis_xyz_excursion_deg']=float(max(ranges3)) if ranges3 else None
    # Body-normalized wrist displacement, not a physical wrist-joint angle.
    wrists=x[:,39:123].reshape(-1,2,21,2)[ids,slot,0]
    result['metrics']['wrist_body_xy_excursion']=float(np.linalg.norm(np.quantile(wrists,.9,axis=0)-np.quantile(wrists,.1,axis=0)))
    for quarter,group in enumerate(np.array_split(np.arange(len(x)),4),1):
        mask=np.isin(ids,group);q={'quarter':quarter,'usable_frames':int(mask.sum())}
        q['thumb_index_xy_median']=float(np.median(np.linalg.norm((points[mask,8]-points[mask,4])[:,:2],axis=1))) if mask.any() else None
        result['quarters'].append(q)
    return result


def compare_training(articulation,reference):
    """Percentile locations describe similarity, never recognition confidence."""
    result={}
    for gloss in ('NO','YES','HELLO'):
        rows=[r for r in reference['records'] if r['split']=='train' and r['gloss']==gloss and not r['articulation']['insufficient']]
        comparison={}
        for key,value in articulation['metrics'].items():
            values=[r['articulation']['metrics'].get(key) for r in rows]
            values=np.asarray([v for v in values if v is not None and np.isfinite(v)],dtype=float)
            if value is not None and len(values):
                comparison[key]={'training_clips':len(values),'training_p10':float(np.quantile(values,.1)),
                    'training_median':float(np.median(values)),'training_p90':float(np.quantile(values,.9)),
                    'live_percentile':float((values<=value).mean()*100)}
        result[gloss]=comparison
    return result
