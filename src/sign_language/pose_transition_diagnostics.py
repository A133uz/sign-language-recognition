"""Frozen pose-versus-entry/exit controls. Live observations stay in memory."""
import numpy as np
from sign_language.pronoun_diagnostics import diagnose_observations,observation_summary

TRIALS=tuple(
    {'label':label,'condition':condition,'repetition':repetition}
    for repetition in (1,2)
    for label,condition in (('ME','hold_pose'),('ME','natural_entry_exit'),
                            ('YOU','natural_entry_exit'),('YOU','hold_pose')))

WINDOWS={'first_second':(0,1000),'middle_second':(1000,2000),
         'last_second':(2000,3000),'last_two_seconds':(1000,3000)}


def diagnose_pose_capture(current,baseline,frames,timestamps):
    arrays,shape,metadata=current.observe_frames(frames,timestamps)
    result=diagnose_observations(current,baseline,arrays,shape,timestamps,
                                 fixed_normalization=True,additional_windows=WINDOWS)
    relative=np.asarray(timestamps)-timestamps[0]
    result['phase_measurements']={}
    for name,(start,end) in WINDOWS.items():
        ids=np.flatnonzero((relative>=start)&(relative<end))
        result['phase_measurements'][name]=(observation_summary({k:v[ids] for k,v in arrays.items()},shape,np.asarray(timestamps)[ids])
                                             if len(ids)>=2 else {'status':'unavailable'})
    return {**metadata,**result,'recognition_tracking_passes':1,
            'media_saved':False,'coordinates_saved':False,'embeddings_saved':False,
            'interpretation':'Held-posture trials and timed windows are diagnostic manipulations, not verified sign boundaries or natural sign-accuracy estimates.'}
