import unittest
import numpy as np

from sign_language.wrist_trajectory import path_metrics, trajectory_summary


class WristTrajectoryTests(unittest.TestCase):
    def test_loop_descriptor_separates_circle_from_retraced_line_and_is_scale_invariant(self):
        angle = np.linspace(0, 2*np.pi, 121)
        circle = np.column_stack([np.cos(angle), np.sin(angle)])
        line = np.column_stack([np.cos(angle), np.cos(angle)])
        metrics = path_metrics(circle)
        self.assertGreater(metrics['loop_probe_score'], .98)
        self.assertAlmostEqual(path_metrics(line)['loop_probe_score'], 0.)
        transformed = path_metrics(circle*3 + [100, -20])
        self.assertAlmostEqual(metrics['loop_probe_score'], transformed['loop_probe_score'])
        self.assertAlmostEqual(metrics['path_length']*3, transformed['path_length'])

    def test_missing_segments_are_not_joined_and_no_window_is_fabricated(self):
        p = np.column_stack([np.arange(14), np.zeros(14)]).astype(float)
        p[9:, 0] += 1000
        mask = np.ones(14, bool)
        mask[5:9] = False
        result = trajectory_summary(p, mask, 30)
        self.assertEqual(result['contiguous_spans'], 2)
        self.assertEqual(result['longest_span_frames'], 5)
        self.assertEqual(result['longest_span']['path_length'], 4.)
        self.assertIsNone(result['best_loop_probe'])
        self.assertIsNone(trajectory_summary(p, np.zeros(14, bool), 30)['longest_span'])

    def test_stationary_path_and_invalid_input_are_explicit(self):
        self.assertEqual(path_metrics(np.zeros((12, 2)))['loop_probe_score'], 0.)
        with self.assertRaises(ValueError):
            trajectory_summary(np.zeros((10, 2)), np.ones(9), 30)
        with self.assertRaises(ValueError):
            trajectory_summary(np.zeros((10, 2)), np.ones(10), 0)
        with self.assertRaises(ValueError):
            path_metrics([[0, 0], [np.nan, 1]])
