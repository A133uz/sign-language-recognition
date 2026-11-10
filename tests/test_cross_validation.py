import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('sklearn'), 'Optional training extra not installed')
class GroupedFoldTests(unittest.TestCase):
    def fixture(self, root):
        rows = [{'video_file': f'{person}-{label}.mp4', 'participant_id': str(person),
                 'class_id': label, 'split': 'train', 'gloss': str(label)}
                for person in range(9) for label in range(3)]
        (root/'manifest.json').write_text(json.dumps({'cache_id': 'fixture', 'labels': {str(i): i for i in range(3)}}))
        (root/'train-index.json').write_text(json.dumps(rows))
        np.save(root/'train-targets.npy', [r['class_id'] for r in rows])
        return rows

    def test_folds_are_repeatable_complete_and_signer_disjoint(self):
        from sign_language.cross_validation import make_plan, load_fold
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); rows = self.fixture(root); path = root/'folds.json'
            plan = make_plan(root, path)
            self.assertEqual(make_plan(root, path), plan)
            held_out = []
            for fold in range(3):
                train, val, info = load_fold(root, path, fold)
                self.assertFalse({rows[i]['participant_id'] for i in train} & {rows[i]['participant_id'] for i in val})
                self.assertEqual(set(train)|set(val), set(range(len(rows))))
                self.assertFalse(info['official_validation_used'])
                held_out.extend(val)
            self.assertEqual(sorted(held_out), list(range(len(rows))))
            # No official val/test files exist in this fixture: planning reads neither.

    def test_tampered_plan_and_metadata_are_rejected(self):
        from sign_language.cross_validation import make_plan, load_fold
        from sign_language.extraction import fingerprint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root); path = root/'folds.json'
            plan = make_plan(root, path)
            train, val = plan['partitions'][0]['train'], plan['partitions'][0]['validation']
            train[0], val[0] = val[0], train[0]
            plan['plan_id'] = fingerprint({k: v for k, v in plan.items() if k != 'plan_id'})
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'Signer leakage'):
                load_fold(root, path, 0)
            (root/'train-index.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'labels differ'):
                load_fold(root, path, 0)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional PyTorch extra not installed')
class RegularizationTests(unittest.TestCase):
    def test_resume_recovers_selected_weights_and_metrics_after_partial_commit(self):
        import torch
        from sign_language.training import repair_epoch_artifacts
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = {'epoch': 3, 'metrics': {'epoch': 3, 'val': {'loss': 1.2}},
                          'best_val_loss': 1.2, 'config': {}, 'labels': {'A': 0},
                          'model': {'weight': torch.tensor([2.])}}
            repair_epoch_artifacts(output, checkpoint)
            self.assertEqual(json.loads((output/'epoch-003.json').read_text()), checkpoint['metrics'])
            best = torch.load(output/'best.pt', weights_only=True)
            self.assertEqual(best['epoch'], 3)
            torch.testing.assert_close(best['model']['weight'], torch.tensor([2.]))

    def test_augmentation_preserves_masks_padding_time_order_and_local_depth(self):
        import torch
        from sign_language.regularization import augment_batch
        torch.manual_seed(7)
        frames = torch.zeros(2, 8, 254)
        frames[:, :5] = torch.randn(2, 1, 254).repeat(1, 5, 1)*.05
        frames[:, :5, 26:39] = 1; frames[:, :5, 249:] = 1
        before = frames.clone(); after = augment_batch(frames)
        torch.testing.assert_close(frames, before)
        torch.testing.assert_close(after[..., 26:39], before[..., 26:39])
        torch.testing.assert_close(after[..., 249:], before[..., 249:])
        torch.testing.assert_close(after[..., 125:249:3], before[..., 125:249:3])
        self.assertTrue((after[:, 5:] == 0).all())
        torch.testing.assert_close(after[:, :1].expand(-1, 5, -1), after[:, :5])
        self.assertFalse(torch.equal(after[:, :5, :26], before[:, :5, :26]))

    def test_augmentation_is_training_only(self):
        import torch
        from sign_language.gru_model import SignGRU
        from sign_language.training import epoch_pass
        model = SignGRU(input_size=4, hidden_size=8, layers=1, classes=3, dropout=0)
        batches = [(torch.randn(2, 5, 4), torch.tensor([5, 3]), torch.tensor([0, 1]))]
        with patch('sign_language.training.augment_batch', side_effect=lambda x: x.clone()) as augment, \
             patch('sign_language.training.torch.cuda.synchronize'):
            epoch_pass(model, batches, torch.nn.CrossEntropyLoss(), 'cpu', augment=True)
            augment.assert_not_called()
            optimizer = torch.optim.AdamW(model.parameters())
            epoch_pass(model, batches, torch.nn.CrossEntropyLoss(), 'cpu', optimizer=optimizer, augment=True)
            augment.assert_called_once()
