"""Same-capture diagnostic controls; live media/coordinate arrays remain in RAM."""
import numpy as np

from sign_language.preprocessing import normalize_sequence
from sign_language.recognition_diagnostics import observation_statistics,hand_visibility_summary
from sign_language.recognition_forensics import (VARIANTS,make_variant,summary_vector,
                                                feature_summary,infer_clips,scores,phase_lengths,match_training_phases)


def analyze_features(recognizer,bank,features,expected):
    target=recognizer.labels[expected]
    transformed=[make_variant(features,n,bank['rest']) for n in VARIANTS]
    logits,hidden=infer_clips(recognizer,transformed)
    result={'feature_summary':feature_summary(features),
            'variants':{n:scores(z,recognizer.names,target) for n,z in zip(VARIANTS,logits)},
            'diagnostic_heads':{}}
    desired=np.median([phase_lengths(x) for x in bank['train_clips'] if phase_lengths(x) is not None],axis=0).astype(int)
    phased=[];phase_info={}
    for name,motion_only in [('match_training_phases',False),('match_training_motion_span',True)]:
        x,applied=match_training_phases(features,desired,motion_only)
        phased.append(x);phase_info[name]={'applied':applied,'output_frames':len(x)}
    phase_logits,_=infer_clips(recognizer,phased)
    for (name,info),z in zip(phase_info.items(),phase_logits):
        result['variants'][name]={**scores(z,recognizer.names,target),**info}
    result['phase_comparison']={'source_phase_lengths':phase_lengths(features),'training_median_phase_lengths':desired.tolist()}
    for name,probe in bank['probes'].items():
        kind=name.split('_')[0]
        vector=hidden[0] if kind=='hidden' else summary_vector(features,kind)
        p=probe.predict_proba(vector[None])[0]
        # These estimators' class indices were validated when the bank was built.
        result['diagnostic_heads'][name]=scores(np.log(np.maximum(p,1e-30)),recognizer.names,target)
    h=hidden[0];training=bank['train_hidden'];train_y=bank['train_targets']
    cosine=1-(training@h)/(np.maximum(np.linalg.norm(training,axis=1)*np.linalg.norm(h),1e-8))
    near=np.argsort(cosine)[:5]
    result['hidden_comparison']={'abs_mean':float(np.abs(h).mean()),
        'saturated_fraction':float((np.abs(h)>.99).mean()),
        'nearest_training_examples':[{'video_file':bank['train_rows'][i]['video_file'],
                 'label':recognizer.names[int(train_y[i])],'cosine_distance':float(cosine[i])} for i in near],
        'median_distance_by_class':{n:float(np.median(cosine[train_y==i])) for i,n in enumerate(recognizer.names)}}
    if expected!='NO':
        weights=recognizer.model.classifier.weight.detach().float().cpu().numpy()
        bias=recognizer.model.classifier.bias.detach().float().cpu().numpy();no=recognizer.labels['NO']
        result['no_vs_target_logit_margin']={'total':float(logits[0,no]-logits[0,target]),
            'bias_contribution':float(bias[no]-bias[target]),
            'representation_contribution':float(np.dot(weights[no]-weights[target],h))}
    # Coarse progression scores only; never save hidden vectors or frame coordinates.
    prefixes=[features[:max(8,round(len(features)*f))] for f in (.25,.5,.75,1.)]
    prefix_logits,_=infer_clips(recognizer,prefixes)
    result['prefix_progression']=[{'fraction':f,**scores(z,recognizer.names,target)} for f,z in zip((.25,.5,.75,1.),prefix_logits)]
    return result


def analyze_capture(recognizer,bank,frames,timestamps,expected):
    arrays,shape,metadata=recognizer.observe_frames(frames,timestamps)
    features,_=normalize_sequence(arrays,shape);features=features.astype(np.float16).astype(np.float32)
    result=analyze_features(recognizer,bank,features,expected)
    result.update(expected=expected,**metadata,input_statistics=observation_statistics(arrays,shape),
                  tracking_summary=hand_visibility_summary(arrays['hand_present']))
    mirrored=[np.ascontiguousarray(f[:,::-1]) for f in frames]
    reflected,ref_shape,_=recognizer.observe_frames(mirrored,timestamps)
    rf,_=normalize_sequence(reflected,ref_shape);rf=rf.astype(np.float16).astype(np.float32)
    z,_=infer_clips(recognizer,[rf])
    result['variants']['mirrored_pixels']=scores(z[0],recognizer.names,recognizer.labels[expected])
    result['mirrored_input_statistics']=observation_statistics(reflected,ref_shape)
    return result
