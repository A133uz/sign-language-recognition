import importlib.util
import json
import unittest
from unittest.mock import patch

import numpy as np
from sign_language.capture_timing_diagnostics import (CaptureTiming,normalize_all_frames,verify_adapter,
    sample_features,feature_difference)
from sign_language.frame_rate import FrameRateLimiter
from sign_language.preprocessing import normalize_sequence


def observations(n):
    phase=np.linspace(0,1,n,dtype=np.float32)
    pose=np.zeros((n,33,3),np.float32)
    pose[:,11,:2]=np.c_[.3-.08*phase,.35+0*phase]
    pose[:,12,:2]=np.c_[.7+.08*phase,.35+0*phase]
    pose[:,15,:2]=[.4,.6];pose[:,16,:2]=[.8,.6]
    hands=np.zeros((n,2,21,3),np.float32)
    for joint in range(21):
        hands[:,0,joint,0]=.4+joint*.003*(1+phase)
        hands[:,0,joint,1]=.6+joint*.002*(1+phase)
        hands[:,0,joint,2]=joint*.001
    present=np.zeros((n,2),bool);present[:,0]=True
    mask=np.ones((n,33),bool)
    return {'pose_xyz':pose,'pose_valid_mask':mask,'hands_xyz':hands,'hand_present':present,
            'handedness':np.tile([0,-1],(n,1)),'handedness_score':np.tile([1.,0.],(n,1)),
            'frame_index':np.arange(n)}


class CaptureTimingTests(unittest.TestCase):
    def test_normalizer_matches_frozen_code_before_and_after_128_sampling(self):
        for n in (24,128,241):
            with self.subTest(n=n):
                arrays=observations(n);arrays['pose_valid_mask'][5:8,11:13]=False
                arrays['hand_present'][12:16,0]=False
                originals={k:v.copy() for k,v in arrays.items()}
                x,context=verify_adapter(arrays,(480,640,3))
                expected,quality=normalize_sequence(arrays,(480,640,3))
                np.testing.assert_array_equal(sample_features(x),expected.astype(np.float16).astype(np.float32))
                self.assertEqual(context['quality'],quality)
                for k,v in originals.items():np.testing.assert_array_equal(arrays[k],v)

    def test_fixed_context_preserves_prefix_values_and_refitting_changes_scales(self):
        arrays=observations(240);full,stats=verify_adapter(arrays,(480,640,3))
        prefix={k:v[:60] for k,v in arrays.items()};refit,part=verify_adapter(prefix,(480,640,3))
        self.assertNotEqual(stats['shoulder_median_width'],part['shoulder_median_width'])
        self.assertNotEqual(stats['palm_median_sizes'][0],part['palm_median_sizes'][0])
        difference=feature_difference(sample_features(full[:60]),sample_features(refit))
        self.assertGreater(difference['body_hands']['rms_difference'],0)
        self.assertEqual(difference['mask_disagreement_fraction'],0)
        self.assertEqual(sample_features(full[:60]).shape,(60,254))

    def test_interval_summary_replays_exact_limiter_decisions_without_serializing_trace(self):
        limiter=FrameRateLimiter(30);timing=CaptureTiming(30)
        times=[0.,.1,.101,.102,.14,.18,.28,.281,.282,.32]
        decisions=[]
        for t in times:
            keep=limiter.accept(t);decisions.append(keep)
            timing.record(t,keep,{'camera_read':3.,'wait_key':15.},limiter.credit)
        result=timing.summary()
        self.assertEqual(result['limiter_replay_mismatches'],0)
        self.assertEqual(result['kept_count'],sum(decisions))
        self.assertEqual(result['dropped_count'],len(times)-sum(decisions))
        self.assertGreater(result['limiter_long_gap_reset_count'],0)
        self.assertGreater(result['drops_within_100ms_after_long_gap'],0)
        self.assertEqual(result['arrival_interval_ms']['count'],len(times)-1)
        self.assertEqual(sum(result['arrival_interval_histogram_ms'].values()),len(times)-1)
        self.assertEqual(result['stages_ms']['wait_key']['p50'],15.)
        self.assertNotIn('arrivals',result);self.assertNotIn('kept',result)
        self.assertFalse(result['raw_timing_trace_saved']);json.dumps(result,allow_nan=False)

    def test_invalid_arrivals_and_stages_are_rejected(self):
        timing=CaptureTiming();timing.record(.2,True,{'render':1},0.)
        for t in (.1,float('nan'),-1):
            with self.subTest(t=t),self.assertRaises(ValueError):timing.record(t,True,{},0.)
        with self.assertRaises(ValueError):timing.record(.3,True,{'render':float('inf')},0.)

    def test_empty_summary_and_absent_hands_remain_finite(self):
        json.dumps(CaptureTiming().summary(),allow_nan=False)
        arrays=observations(20);arrays['hand_present'][:]=False
        _,context=verify_adapter(arrays,(480,640,3))
        self.assertEqual(context['palm_median_sizes'],[None,None])


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'Optional PyTorch unavailable')
class FixedWindowTests(unittest.TestCase):
    def test_fixed_windows_crop_full_resolution_features_before_sampling(self):
        import torch
        from sign_language import pronoun_diagnostics as d
        arrays=observations(240);stamps=np.arange(240)*33
        full,_=normalize_all_frames(arrays,(480,640,3));seen=[]
        def analyze(a,b,x):seen.append(x.copy());return {'test_result':True}
        with patch.object(d,'diagnose_features',side_effect=analyze):
            result=d.diagnose_observations(None,None,arrays,(480,640,3),stamps,fixed_normalization=True)
        fixed=result['windows']['first_two_seconds_fixed_normalization']
        self.assertEqual(fixed['observed_frames'],61);self.assertEqual(fixed['input_frames'],61)
        np.testing.assert_array_equal(seen[3],sample_features(full[:61]))
        self.assertEqual(result['windows']['full']['input_frames'],128)
        self.assertTrue(result['normalization_comparison']['full_adapter_matches_production_at_float16'])
        json.dumps(result,allow_nan=False)
