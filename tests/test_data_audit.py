"""Corrupt landmark fixtures must be detected by the independent auditor."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SPEC = importlib.util.spec_from_file_location('data_audit', Path(__file__).resolve().parents[1] / 'scripts/audit_video_features.py')
audit = importlib.util.module_from_spec(SPEC)
# These fixtures exercise NumPy validation only. Do not load OpenCV native DLLs
# during discovery before the project's PyTorch integration tests initialize.
with patch.dict(sys.modules, {'cv2': SimpleNamespace()}):
    SPEC.loader.exec_module(audit)
SLOT_SPEC = importlib.util.spec_from_file_location('audit_hand_slots', Path(__file__).resolve().parents[1] / 'scripts/audit_hand_slots.py')
slots = importlib.util.module_from_spec(SLOT_SPEC)
SLOT_SPEC.loader.exec_module(slots)


class AuditValidationTests(unittest.TestCase):
    def setUp(self):
        self.arrays = {key: np.zeros((3, *tail), dtype=dtype) for key,(dtype,tail) in audit.SCHEMA.items()}
        self.arrays['handedness'].fill(-1)
        self.arrays['frame_index'][:] = [0,1,2]
        self.arrays['timestamp_ms'][:] = [0,33,67]
        self.record = dict(frames=3, fps=30, frames_with_hands=0, frames_with_pose=0,
                           hand_detections=0, pose_valid_points=0)

    def test_explicit_missing_detections_are_valid_not_corrupt(self):
        self.assertEqual(audit.validate_arrays(self.arrays, self.record), [])

    def test_nonfinite_and_truncated_arrays_rejected(self):
        self.arrays['hands_xyz'][0,0,0,0] = np.nan
        self.arrays['pose_xyz'] = self.arrays['pose_xyz'][:2]
        errors = audit.validate_arrays(self.arrays, self.record)
        self.assertIn('nonfinite:hands_xyz', errors)
        self.assertIn('shape_or_dtype:pose_xyz', errors)

    def test_timestamp_mask_and_zero_fill_errors_rejected(self):
        self.arrays['timestamp_ms'][2] = 33
        self.arrays['pose_valid_mask'][0,0] = True
        self.arrays['hands_xyz'][0,0,0,0] = .2
        errors = audit.validate_arrays(self.arrays, self.record)
        for expected in ('timestamps', 'pose_mask', 'absent_hand_fill', 'record_count:pose_valid_points'):
            self.assertIn(expected, errors)

    def test_out_of_image_detected_landmarks_are_review_signals(self):
        self.arrays['hand_present'][0,0] = True
        self.arrays['handedness'][0,0] = 0
        self.arrays['hands_xyz'][0,0,:,0] = 1.1
        self.record.update(frames_with_hands=1, hand_detections=1)
        self.assertEqual(audit.validate_arrays(self.arrays, self.record), [])

    def test_slot_switch_heuristic_excludes_padding_and_missing_anchors(self):
        features = np.zeros((1,4,254),np.float16)
        features[0,:,253] = 1
        features[0,0,249] = 1
        features[0,1,250] = 1
        features[0,2,249] = 1
        changes,nearby,_ = slots.slot_changes(features,np.array([2]))
        self.assertEqual(np.flatnonzero(nearby).tolist(),[0])
        features[0,1,253] = 0
        self.assertFalse(slots.slot_changes(features,np.array([2]))[1].any())

    def test_distant_hand_change_is_not_flagged_nearby(self):
        features = np.zeros((1,2,254),np.float16)
        features[0,:,253] = 1
        features[0,0,249] = 1
        features[0,1,250] = 1
        features[0,1,81] = 2
        changes,nearby,_ = slots.slot_changes(features,np.array([2]))
        self.assertTrue(changes.any())
        self.assertFalse(nearby.any())


if __name__ == '__main__':
    unittest.main()
