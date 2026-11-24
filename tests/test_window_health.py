import unittest
from pathlib import Path
import sys

from sign_language.streaming_face import RollingObservations
from sign_language.window_health import failure_category, window_health
from test_streaming_face import sample
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from report_sliding_health_diagnostic import summarize


class WindowHealthTests(unittest.TestCase):
    def make_snapshot(self, hands=True, shoulders=True, face=True):
        window = RollingObservations(2000)
        for i in range(10):
            body, xyz, present, blend = sample(i, hand=hands, face=face)
            if not shoulders:
                body['pose_valid_mask'][11:13] = False
            window.append(i*100, body, xyz, present, blend, (480, 640, 3))
        return window.snapshot()

    def test_missing_hands_and_shoulders_are_distinguished(self):
        for hands, shoulders, expected in [(False, True, 'no_hands'),
                                           (True, False, 'no_body_anchor'),
                                           (False, False, 'no_hands_and_no_body_anchor')]:
            health = window_health(self.make_snapshot(hands, shoulders))
            self.assertEqual(failure_category(ValueError('Keep hands and shoulders visible'), health), expected)
            self.assertEqual(health['raw_hand_frames'], 10 if hands else 0)
            self.assertEqual(health['raw_shoulder_frames'], 10 if shoulders else 0)
            self.assertTrue(all(isinstance(value, (int, str, type(None))) for value in health.values()))

    def test_missing_face_and_unrecognized_error(self):
        health = window_health(self.make_snapshot(face=False))
        self.assertEqual(health['model_face_frames'], 0)
        self.assertEqual(failure_category(ValueError('Keep face visible'), health), 'no_valid_face')
        self.assertEqual(failure_category(ValueError('arbitrary private text'), health), 'other_value_error')

    def test_summary_checks_failure_details(self):
        health = window_health(self.make_snapshot(hands=False))
        report = {'health_measurements':True, 'status':'complete', 'confirmation':'pending',
                  'trials':[{'number':1, 'prompt':'REST_BELOW', 'completed':True,
                             'windows':[], 'unscored_windows':1,
                             'unscored_details':[{'reason':'no_hands', 'health':health}]}]}
        result = summarize(report)['trials'][0]
        self.assertEqual(result['failure_reasons'], {'no_hands':1})
        self.assertEqual(result['aggregate']['raw_hand_frames_in_windows'], 0)
        report['trials'][0]['unscored_windows'] = 2
        with self.assertRaises(ValueError): summarize(report)


if __name__ == '__main__': unittest.main()
