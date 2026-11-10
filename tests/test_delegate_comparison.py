"""Verify comparison metrics independently of native GPU availability."""

import ast
from pathlib import Path
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / 'notebooks/colab_compare_delegates_cell.py'
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
worker = next(ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'comparison_worker' for target in node.targets))
namespace = {'__name__': 'comparison_metrics_test'}
exec(compile(worker, str(SOURCE), 'exec'), namespace)
compare_arrays = namespace['compare_arrays']

audit_source = SOURCE.with_name('colab_audit_comparison_cell.py')
audit_tree = ast.parse(audit_source.read_text(encoding='utf-8'))
audit_worker = next(ast.literal_eval(node.value) for node in audit_tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'comparison_audit_program' for target in node.targets))
audit_namespace = {'__name__': 'comparison_audit_test'}
exec(compile(audit_worker, str(audit_source), 'exec'), audit_namespace)


def observations():
    return {
        'hands_xyz': np.zeros((2,2,21,3), dtype=np.float32),
        'hand_present': np.ones((2,2), dtype=bool),
        'handedness': np.tile(np.array([0,1], dtype=np.int8), (2,1)),
        'pose_xyz': np.zeros((2,33,3), dtype=np.float32),
        'pose_valid_mask': np.ones((2,33), dtype=bool),
        'pose_detected': np.ones(2, dtype=bool),
        'frame_index': np.arange(2), 'timestamp_ms': np.array([0,33]),
    }


class DelegateComparisonTests(unittest.TestCase):
    def test_swapped_hand_slots_are_matched(self):
        cpu, gpu = observations(), observations()
        cpu['hands_xyz'][:,0,:,0] = .2
        cpu['hands_xyz'][:,1,:,0] = .8
        gpu['hands_xyz'] = cpu['hands_xyz'][:,::-1].copy()
        gpu['handedness'] = cpu['handedness'][:,::-1].copy()
        result = compare_arrays(cpu, gpu)
        self.assertEqual(result['hand_xy_distance']['mean'], 0)
        self.assertEqual(result['matched_handedness_agreement'], 1)

    def test_absent_detections_do_not_compare_zero_fills(self):
        cpu, gpu = observations(), observations()
        gpu['hand_present'][:] = False
        gpu['pose_detected'][:] = False
        gpu['pose_valid_mask'][:] = False
        result = compare_arrays(cpu, gpu)
        self.assertEqual(result['hand_count_agreement'], 0)
        self.assertEqual(result['hand_xy_distance']['points'], 0)
        self.assertIsNone(result['hand_xy_distance']['mean'])
        self.assertIsNone(result['pose_xy_distance']['mean'])

    def test_coordinate_distance_is_normalized_xy_on_common_points(self):
        cpu, gpu = observations(), observations()
        gpu['hand_present'][:,1] = False
        gpu['hands_xyz'][:,:,:,0] = .03
        gpu['hands_xyz'][:,:,:,1] = .04
        gpu['hands_xyz'][:,:,:,2] = 100
        gpu['pose_valid_mask'][:,1:] = False
        gpu['pose_xyz'][:,:,0] = .1
        result = compare_arrays(cpu, gpu)
        self.assertEqual(result['hand_xy_distance']['points'], 42)
        self.assertAlmostEqual(result['hand_xy_distance']['mean'], .05)
        self.assertEqual(result['pose_xy_distance']['points'], 2)
        self.assertAlmostEqual(result['pose_xy_distance']['mean'], .1)

    def test_nonfinite_or_different_timing_is_rejected(self):
        cpu, gpu = observations(), observations()
        gpu['timestamp_ms'][1] = 40
        with self.assertRaisesRegex(ValueError, 'timestamps differ'):
            compare_arrays(cpu, gpu)
        gpu = observations()
        gpu['hands_xyz'][0,0,0,0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            compare_arrays(cpu, gpu)


class ComparisonAuditTests(unittest.TestCase):
    def test_opposite_reported_hands_are_visible_in_audit(self):
        cpu, gpu = observations(), observations()
        cpu['hand_present'][:,1] = False
        gpu['hand_present'][:,0] = False
        gpu['hands_xyz'][:,1,:,0] = .4
        detail = audit_namespace['frame_audit'](cpu, gpu, 0)
        self.assertEqual(detail['cpu_reported_labels'], ['Left'])
        self.assertEqual(detail['gpu_reported_labels'], ['Right'])
        self.assertTrue(detail['matched_labels_disagree'])
        self.assertAlmostEqual(detail['xy_p95'], .4)

    def test_audit_selects_actual_worst_frame_and_counts_missing_groups(self):
        cpu, gpu = observations(), observations()
        cpu['hand_present'][0] = False
        gpu['hand_present'][0] = False
        gpu['hands_xyz'][1,:,:,0] = .2
        item = {'key':'000', 'video_file':'fixture.mp4', 'gloss':'TEST',
                'comparison':compare_arrays(cpu, gpu)}
        report = audit_namespace['audit_clip'](item, cpu, gpu)
        self.assertEqual(report['worst_frame']['source_frame'], 1)
        self.assertEqual(report['frames_both_no_hands'], 1)
        self.assertEqual(report['frames_xy_p95_over_0_10'], 1)

    def test_order_swaps_do_not_create_false_handedness_disagreement(self):
        cpu, gpu = observations(), observations()
        cpu['hands_xyz'][:,0,:,0] = .2
        cpu['hands_xyz'][:,1,:,0] = .8
        gpu['hands_xyz'] = cpu['hands_xyz'][:,::-1].copy()
        gpu['handedness'] = cpu['handedness'][:,::-1].copy()
        detail = audit_namespace['frame_audit'](cpu, gpu, 0)
        self.assertFalse(detail['matched_labels_disagree'])
        self.assertEqual(detail['xy_p95'], 0)


if __name__ == '__main__':
    unittest.main()
