import unittest
from pathlib import Path
import sys
import numpy as np

from sign_language.activity_metrics import motion_summary
from sign_language.streaming_face import RollingObservations
from test_streaming_face import sample
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from report_sliding_activity_diagnostic import summarize


class ActivityTests(unittest.TestCase):
    def test_static_and_moving_wrist(self):
        buffer = RollingObservations(5000)
        for i in range(12):
            buffer.append(i*100, *sample(0), (480, 640, 3))
        still = motion_summary(buffer.snapshot())
        self.assertAlmostEqual(still['wrist_speed_p75_shoulder_widths_per_second'], 0.)
        self.assertAlmostEqual(still['handshape_speed_p75_palm_widths_per_second'], 0.)
        moving = RollingObservations(5000)
        for i in range(12):
            body, face, present, blend = sample(0)
            body['hands_xyz'][0, :, 0] += i*.01
            moving.append(i*100, body, face, present, blend, (480, 640, 3))
        result = motion_summary(moving.snapshot())
        self.assertGreater(result['wrist_speed_p75_shoulder_widths_per_second'], .1)
        self.assertLess(result['handshape_speed_p75_palm_widths_per_second'], .0001)

    def test_missing_hands_do_not_turn_into_false_motion(self):
        buffer = RollingObservations(5000)
        for i in range(10):
            buffer.append(i*100, *sample(i, hand=False), (480, 640, 3))
        result = motion_summary(buffer.snapshot())
        self.assertIsNone(result['wrist_speed_p75_shoulder_widths_per_second'])
        self.assertIsNone(result['handshape_speed_p75_palm_widths_per_second'])
        self.assertEqual(result['valid_wrist_pairs'], 0)

    def test_missing_and_long_gaps_are_not_interpolated(self):
        buffer = RollingObservations(5000)
        for t in (0, 100, 200, 700, 800, 900, 1000, 1100):
            buffer.append(t, *sample(t//100, hand=t != 800), (480, 640, 3))
        result = motion_summary(buffer.snapshot())
        self.assertEqual(result['eligible_time_pairs'], 6)
        self.assertEqual(result['valid_wrist_pairs'], 4)

    def test_summary_separates_early_and_late_without_truth_claims(self):
        motion = {'wrist_speed_p75_shoulder_widths_per_second': .2,
                  'handshape_speed_p75_palm_widths_per_second': None}
        report = {'activity_measurements': True, 'status': 'complete',
                  'confirmation': 'pending_user_confirmation', 'trials': [
                  {'number': 1, 'prompt': 'HELLO', 'completed': True,
                   'duration_seconds': 7, 'windows': [
                       {'end_seconds': 2.5, 'motion': motion},
                       {'end_seconds': 6., 'motion': motion}]}]}
        item = summarize(report)['trials'][0]
        self.assertEqual(item['phases']['early_proxy']['windows'], 1)
        self.assertEqual(item['phases']['late_proxy']['windows'], 1)
        self.assertIsNone(item['phases']['all']['median']['handshape_speed_p75_palm_widths_per_second'])


if __name__ == '__main__': unittest.main()
