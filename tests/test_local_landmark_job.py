import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    'local_landmark_job', Path(__file__).resolve().parents[1]/'scripts/local_landmark_job.py')
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)
CRASH = 'concurrent.futures.process.BrokenProcessPool: fixture worker died'


class RecoveryTests(unittest.TestCase):
    def exercise(self, attempts, max_retries=2, stop_after=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = {'status': 'starting'}
            summary = {'status': 'partial', 'processed_clips': 100}
            calls = []

            def attempt(state, command, environment):
                code, count, status, error = attempts[len(calls)]
                calls.append(list(command))
                summary.update(status=status, processed_clips=count,
                               newly_processed_clips=count-summary['processed_clips'])
                log = root/f'{len(calls)}.native.log'
                log.write_text(error)
                state.update(status='running', native_log=str(log), stdout_log=str(root/'stdout.log'))
                if stop_after == len(calls):
                    (root/'stop').touch()
                return code

            with patch.object(job, 'STOP', root/'stop'), \
                 patch.object(job, 'read_summary', side_effect=lambda: dict(summary)), \
                 patch.object(job, 'run_attempt', side_effect=attempt), \
                 patch.object(job, 'save_state'), patch.object(job.time, 'sleep'):
                job.extraction_loop(state, ['extract', '--workers', '4', '--max-shards', '25'], {}, max_retries)
            return state, calls

    def test_refresh_crash_recovery_and_completion_keep_progress(self):
        state, calls = self.exercise([
            (0, 2600, 'partial', ''),
            (1, 2700, 'partial', CRASH),
            (0, 5200, 'partial', ''),
            (0, 5300, 'complete', ''),
        ])
        self.assertEqual(state['status'], 'completed')
        self.assertEqual([a['action'] for a in state['attempt_history']],
                         ['refreshing_workers', 'retrying_worker_crash', 'refreshing_workers', 'completed'])
        self.assertEqual([a['processed_before'] for a in state['attempt_history']], [100, 2600, 2700, 5200])
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(c == calls[0] for c in calls))

    def test_repeated_crash_at_same_point_stops_after_two_retries(self):
        state, calls = self.exercise([(1, 100, 'partial', CRASH)]*3)
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(len(calls), 3)
        self.assertEqual(state['consecutive_worker_crashes'], 3)

    def test_other_errors_do_not_retry_or_use_stale_partial_summary(self):
        for error in ('ValueError: Cache configuration differs', 'OSError: Disk full'):
            with self.subTest(error=error):
                state, calls = self.exercise([(1, 100, 'partial', error)])
                self.assertEqual(state['status'], 'failed')
                self.assertEqual(len(calls), 1)
        self.assertEqual(job.attempt_action(0, {'status': 'partial', 'newly_processed_clips': 0}, '', 0, 2), 'failed')

    def test_stop_after_crash_prevents_restart(self):
        state, calls = self.exercise([(1, 100, 'partial', CRASH)], stop_after=1)
        self.assertEqual(state['status'], 'stopped')
        self.assertEqual(len(calls), 1)

    def test_failed_clips_are_retained_in_final_status(self):
        state, _ = self.exercise([(2, 200, 'partial', ''), (2, 250, 'complete_with_failures', '')])
        self.assertEqual(state['status'], 'completed_with_failures')

    def test_stop_during_attempt_kills_child_and_preserves_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stop = root/'stop'
            stop.touch()
            state = {}
            process = SimpleNamespace(pid=123, poll=lambda: None, wait=lambda: 1)
            with patch.object(job, 'JOB', root), patch.object(job, 'STOP', stop), \
                 patch.object(job, 'save_state'), patch.object(job.subprocess, 'Popen', return_value=process), \
                 patch.object(job, 'kill_tree') as kill:
                code = job.run_attempt(state, ['extract'], {})
            kill.assert_called_once_with(process)
            self.assertEqual(code, 1)
            self.assertEqual(state['status'], 'stopped')
            self.assertTrue(Path(state['native_log']).exists())


if __name__ == '__main__':
    unittest.main()
