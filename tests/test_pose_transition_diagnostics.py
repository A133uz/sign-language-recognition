import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from sign_language import pronoun_diagnostics as pronouns
from sign_language import pose_transition_diagnostics as poses
from test_capture_timing_diagnostics import observations


class PoseTransitionTests(unittest.TestCase):
    def test_phase_windows_use_real_time_raw_frames_and_fixed_context_on_same_tracking_pass(self):
        arrays=observations(90);stamps=np.arange(90)*33
        current=SimpleNamespace(observe_frames=lambda frames,times:(arrays,(480,640,3),{'captured_frames':90}))
        before={k:v.copy() for k,v in arrays.items()};seen=[]
        def score(a,b,features):
            seen.append(features.copy())
            return {'aggregate_fixture':True}
        with patch.object(current,'observe_frames',wraps=current.observe_frames) as observe,\
             patch.object(pronouns,'diagnose_features',side_effect=score):
            result=poses.diagnose_pose_capture(current,None,[object()]*90,stamps)
        observe.assert_called_once()
        self.assertEqual(result['recognition_tracking_passes'],1)
        self.assertEqual(result['windows']['first_second']['observed_frames'],31)
        self.assertEqual(result['windows']['middle_second']['observed_frames'],30)
        self.assertEqual(result['windows']['last_second']['observed_frames'],29)
        self.assertEqual(result['windows']['last_two_seconds']['observed_frames'],59)
        self.assertEqual(result['windows']['middle_second_fixed_normalization']['observed_frames'],30)
        self.assertEqual(set(result['phase_measurements']),set(poses.WINDOWS))
        self.assertTrue(result['normalization_comparison']['full_adapter_matches_production_at_float16'])
        self.assertFalse(result['coordinates_saved']);self.assertFalse(result['embeddings_saved'])
        json.dumps(result,allow_nan=False)
        for k,v in before.items():np.testing.assert_array_equal(arrays[k],v)

    def test_empty_detection_windows_are_explicit_not_guessed(self):
        arrays=observations(90);arrays['hand_present'][:]=False
        current=SimpleNamespace(observe_frames=lambda f,t:(arrays,(480,640,3),{}))
        with patch.object(pronouns,'diagnose_features') as score:
            result=poses.diagnose_pose_capture(current,None,[object()]*90,np.arange(90)*33)
        score.assert_not_called()
        self.assertTrue(all(w['status']=='unavailable' for w in result['windows'].values()))
