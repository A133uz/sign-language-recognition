import unittest
from pathlib import Path
import sys

from sign_language.sliding_filter_diagnostics import CandidateFilters
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from report_sliding_filter_diagnostic import summarize


class FilterTests(unittest.TestCase):
    def test_matched_filters_and_switches(self):
        f = CandidateFilters(.6)
        self.assertEqual(f.update('HELLO', .59, 1),
                         {'raw': 'HELLO', 'threshold': None, 'threshold_and_stability': None})
        self.assertEqual(f.update('HELLO', .6, 1.5)['threshold_and_stability'], None)
        self.assertEqual(f.update('HELLO', .7, 2)['threshold_and_stability'], 'HELLO')
        self.assertIsNone(f.update('NO', .9, 2.5)['threshold_and_stability'])
        self.assertEqual(f.update('NO', .9, 3)['threshold_and_stability'], 'NO')

    def test_time_gap_and_unscored_reset_stability(self):
        f = CandidateFilters()
        f.update('HELLO', .9, 1)
        self.assertIsNone(f.update('HELLO', .9, 4)['threshold_and_stability'])
        f.reset()
        self.assertIsNone(f.update('HELLO', .9, 5)['threshold_and_stability'])

    def test_invalid_values(self):
        f = CandidateFilters()
        for label, score, timestamp in [('', .7, 1), ('X', float('nan'), 1), ('X', 1.1, 1)]:
            with self.assertRaises(ValueError): f.update(label, score, timestamp)
        f.update('X', .7, 1)
        with self.assertRaises(ValueError): f.update('X', .7, 1)

    def test_report_does_not_treat_missing_windows_as_success(self):
        report = {'status':'complete', 'confirmation':'pending_user_confirmation',
                  'threshold_experimental':.6, 'trials':[{'number':1,'prompt':'HELLO',
                  'completed':True,'windows':[],'duration_seconds':7}]}
        summary = summarize(report)
        self.assertEqual(summary['confirmation'], 'pending_user_confirmation')
        self.assertEqual(summary['trials'][0]['scored_windows'], 0)
        self.assertFalse(summary['trials'][0]['modes']['raw']['matched_prompt_at_least_once'])


if __name__ == '__main__': unittest.main()
