import importlib.util
import unittest

import numpy as np

HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from sign_language.recognition_forensics import make_variant,VARIANTS,feature_summary,match_training_phases


@unittest.skipUnless(HAS_TORCH,'Optional PyTorch extra unavailable')
class ForensicControlTests(unittest.TestCase):
    def test_phase_matching_preserves_order_and_never_invents_missing_rest(self):
        x=np.zeros((100,254),np.float32);x[:,0]=np.arange(100);x[10:90,249]=1
        before=x.copy();changed,applied=match_training_phases(x,[12,31,26])
        self.assertTrue(applied);self.assertEqual(len(changed),69)
        self.assertTrue(np.all(np.diff(changed[:,0])>=0))
        self.assertFalse(changed[:12,249].any());self.assertTrue(changed[12:43,249].all())
        self.assertFalse(changed[43:,249].any());np.testing.assert_array_equal(x,before)
        original,applied=match_training_phases(x[10:],[12,31,26])
        self.assertFalse(applied);np.testing.assert_array_equal(original,x[10:])

    def test_bilateral_reflection_is_involution_and_swaps_masks_with_geometry(self):
        x=np.random.default_rng(1).normal(size=(40,254)).astype(np.float32)
        x[:,249:251]=[1,0];x[:,251:253]=[1,0];before=x.copy();rest=x[-1].copy()
        reflected=make_variant(x,'reflect_anatomy',rest)
        np.testing.assert_array_equal(reflected[:,249:251],np.tile([0,1],(40,1)))
        np.testing.assert_array_equal(make_variant(reflected,'reflect_anatomy',rest),x)
        np.testing.assert_array_equal(x,before)

    def test_rest_pose_control_does_not_change_active_hands_or_local_geometry(self):
        x=np.zeros((50,254),np.float32);x[:30,249]=1;x[:30,39:249]=.2
        x[:,:39]=.6;rest=np.zeros(254,np.float32);rest[:39]=.1
        before=x.copy();changed=make_variant(x,'training_rest_pose_tail',rest)
        np.testing.assert_array_equal(changed[:30],x[:30])
        np.testing.assert_array_equal(changed[:,39:253],x[:,39:253])
        self.assertTrue(np.all(changed[30:,:39]==.1))
        np.testing.assert_array_equal(x,before)

    def test_controls_bound_lengths_preserve_source_and_summarize_slot_changes(self):
        x=np.zeros((128,254),np.float32);x[10:50,249]=1;x[50:90,250]=1
        before=x.copy();rest=np.zeros(254,np.float32)
        for name in VARIANTS:
            y=make_variant(x,name,rest)
            self.assertEqual(y.shape[1],254);self.assertTrue(1<=len(y)<=128)
            self.assertTrue(np.isfinite(y).all());np.testing.assert_array_equal(x,before)
        summary=feature_summary(x)
        self.assertEqual(summary['single_hand_slot_changes'],1)
        self.assertEqual(summary['hand_frame_fraction'],80/128)
