import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class QueueTests(unittest.TestCase):
    def test_waits_for_completed_cv_before_launching_64_unit_baseline(self):
        runner = script('run_small_gru')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); calls = []

            def run(command, **kwargs):
                calls.append(command)
                if command[-1] == 'status':
                    running = len(calls) == 1
                    return SimpleNamespace(returncode=0, stdout=json.dumps({
                        'status': 'running' if running else 'completed', 'supervisor_running': running}))
                return SimpleNamespace(returncode=0)

            with patch.object(runner, 'JOB', root), patch.object(runner, 'OUTPUT', root/'model'), \
                 patch.object(runner, 'verify_engine') as verify, patch.object(runner.time, 'sleep'), \
                 patch.object(runner.subprocess, 'run', side_effect=run):
                runner.main()
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[-1][calls[-1].index('--hidden-size')+1], '64')
            self.assertEqual(calls[-1][calls[-1].index('--profile')+1], 'baseline')
            self.assertEqual(json.loads((root/'queue.json').read_text())['status'], 'completed')
            self.assertEqual(verify.call_count, 2)

    def test_failed_cv_does_not_start_training(self):
        runner = script('run_small_gru')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runner, 'JOB', root), patch.object(runner, 'verify_engine'), \
                 patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(
                    stdout=json.dumps({'status': 'failed', 'supervisor_running': False}))) as run:
                with self.assertRaisesRegex(RuntimeError, 'did not complete'):
                    runner.main()
            self.assertEqual(run.call_count, 1)
            self.assertEqual(json.loads((root/'queue.json').read_text())['status'], 'failed')

    def test_existing_cv_output_uses_frozen_source(self):
        cli = script('train_local')
        study = cli.ROOT/'models/asl-gru-128-cv-regularized-v1'
        with patch.dict('os.environ', {}, clear=True), \
             patch.object(sys, 'argv', ['train_local.py', 'train', '--output', str(study/'fold-3')]), \
             patch.object(Path, 'exists', return_value=True):
            self.assertEqual(cli.source_directory(), study/'engine_snapshot/src')


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional PyTorch extra not installed')
class SmallModelTests(unittest.TestCase):
    def test_64_unit_model_size_and_output_classes(self):
        import torch
        from sign_language.gru_model import SignGRU
        model = SignGRU(hidden_size=64, layers=2, classes=2731)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 263915)
        result = model(torch.zeros(2, 5, 254), torch.tensor([5, 3]))
        self.assertEqual(tuple(result.shape), (2, 2731))
