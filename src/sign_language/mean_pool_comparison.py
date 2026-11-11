"""Paired inference on identical transient landmarks; save scores, never inputs."""
import numpy as np

from sign_language.preprocessing import normalize_sequence
from sign_language.recognition_diagnostics import observation_statistics,hand_visibility_summary
from sign_language.recognition_forensics import infer_clips,make_variant,scores


def compare_pooling_capture(baseline,candidate,bank,frames,timestamps,expected,feature_diagnostics=None):
    if baseline.labels!=candidate.labels or baseline.config['cache_id']!=candidate.config['cache_id']:
        raise ValueError('Comparison requires identical labels and training cache')
    arrays,shape,metadata=baseline.observe_frames(frames,timestamps)
    features,quality=normalize_sequence(arrays,shape)
    if not quality['frames_with_hands'] or not quality['frames_with_body_anchor']:
        raise ValueError('Capture must contain detected hands and a valid shoulder anchor')
    features=features.astype(np.float16).astype(np.float32);target=baseline.labels[expected]
    variants={};prefix_results={}
    base_controls=[features,make_variant(features,'training_rest_tail_16',bank['rest']),
                   make_variant(features,'training_rest_tail_32',bank['rest'])]
    prefixes=[features[:max(8,round(len(features)*f))] for f in (.25,.5,.75,1.)]
    for name,recognizer in [('baseline',baseline),('mean_pool',candidate)]:
        z,_=infer_clips(recognizer,base_controls+prefixes)
        keys=['original' if name=='baseline' else 'mean_pool',name+'_rest16',name+'_rest32']
        for key,logits in zip(keys,z[:3]):variants[key]=scores(logits,recognizer.names,target)
        prefix_results[name]=[{'fraction':fraction,**scores(logits,recognizer.names,target)}
                              for fraction,logits in zip((.25,.5,.75,1.),z[3:])]
    result={'expected':expected,**metadata,'variants':variants,'prefix_progression':prefix_results,
            'input_statistics':observation_statistics(arrays,shape),
            'tracking_summary':hand_visibility_summary(arrays['hand_present']),
            'rest_controls_are_artificial_diagnostics':True,
            'identical_observations_for_both_models':True}
    if feature_diagnostics is not None:result.update(feature_diagnostics(features))
    return result


def compare_compact_capture(baseline,candidate,bank,frames,timestamps,expected,reference):
    from sign_language.hand_articulation import summarize_articulation,compare_training
    def measure(features):
        articulation=summarize_articulation(features)
        return {'articulation':articulation,'training_reference_comparison':compare_training(articulation,reference)}
    return compare_pooling_capture(baseline,candidate,bank,frames,timestamps,expected,feature_diagnostics=measure)
