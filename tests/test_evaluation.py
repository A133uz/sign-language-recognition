import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional PyTorch extra not installed')
class EvaluationTests(unittest.TestCase):
    def test_confusion_metrics_use_true_rows_and_handle_absent_classes(self):
        from sign_language.evaluation import classification_metrics
        metrics, precision, recall, f1 = classification_metrics(np.array([[3, 1, 0], [1, 1, 0], [0, 0, 0]]))
        self.assertAlmostEqual(metrics['top1'], 4/6)
        np.testing.assert_allclose(precision, [.75, .5, 0])
        np.testing.assert_allclose(recall, [.75, .5, 0])
        np.testing.assert_allclose(f1, [.75, .5, 0])
        self.assertAlmostEqual(metrics['macro_f1'], 1.25/3)
        self.assertAlmostEqual(metrics['weighted_f1'], 4/6)
        with self.assertRaises(ValueError):
            classification_metrics(np.zeros((3, 3), dtype=int))

    def test_evaluation_is_inference_only_ordered_and_preserves_checkpoint(self):
        for architecture in ('gru', 'stgcn27-v1'):
            with self.subTest(architecture=architecture):
                self.check_frozen_evaluation(architecture)

    def check_frozen_evaluation(self, architecture):
        import torch
        from sign_language.evaluation import evaluate

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.linear = torch.nn.Linear(4, 5, bias=False)
                with torch.no_grad():
                    self.linear.weight.zero_(); self.linear.weight[0, 0] = 3; self.linear.weight[1, 1] = 3

            def cuda(self):
                return self

            def forward(self, frames, lengths):
                assert not self.training and not torch.is_grad_enabled()
                return self.linear(frames[:, 0])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cache = root/'cache'; cache.mkdir(); output = root/'evaluation'
            np.save(cache/'test-targets.npy', np.array([0, 1]))
            model = TinyModel(); config = {'model': {}, 'cache_id': 'fixture', 'selection': 'validation loss',
                                         'cudnn_enabled': False, 'amp': False}
            if architecture != 'gru':
                config['architecture'] = architecture
            checkpoint = root/'best.pt'; (root/'config.json').write_text(json.dumps(config))
            torch.save({'config': config, 'model': model.state_dict(), 'labels': {str(i): i for i in range(5)},
                        'epoch': 7, 'metrics': {'val': {'top1': .5}}}, checkpoint)
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            index = [{'video_file': f'{i}.mp4', 'participant_id': str(i), 'class_id': i,
                      'split': 'test', 'gloss': str(i)} for i in range(2)]
            frames = torch.tensor([[[1., 0, 0, 0]], [[0., 1, 0, 0]]])
            batches = [(frames, torch.tensor([1, 1]), torch.tensor([0, 1]))]
            with patch('sign_language.evaluation.validate_cache', return_value=index), \
                 patch('sign_language.evaluation.SignGRU', return_value=model), \
                 patch('sign_language.stgcn_model.SignSTGCN', return_value=model), \
                 patch('sign_language.evaluation.DataLoader', return_value=batches), \
                 patch.object(torch.Tensor, 'cuda', lambda tensor, **kwargs: tensor), \
                 patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.synchronize'), \
                 patch('torch.cuda.reset_peak_memory_stats'), patch('torch.cuda.max_memory_allocated', return_value=0), \
                 patch('torch.cuda.get_device_name', return_value='fixture'):
                report = evaluate(checkpoint, cache, root/'dataset', output, workers=0)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['metrics']['samples'], 2)
            self.assertEqual(report['metrics']['top1'], 1)
            self.assertFalse(report['training_performed'])
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before)
            predictions = [json.loads(line) for line in (output/'predictions.jsonl').read_text().splitlines()]
            self.assertEqual([r['video_file'] for r in predictions], ['0.mp4', '1.mp4'])
            with np.load(output/'confusion.npz') as confusion:
                self.assertEqual(int(confusion['counts'].sum()), 2)
