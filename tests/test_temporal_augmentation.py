import importlib.util
import unittest
from unittest.mock import patch

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from sign_language.temporal_augmentation import TemporalCollator, temporal_clip
    from sign_language.gru_geometry_model import SignGeometryMeanPoolGRU, geometry_batch
    from sign_language.regularization import augment_batch


@unittest.skipUnless(HAS_TORCH, 'Optional PyTorch unavailable')
class TemporalAugmentationTests(unittest.TestCase):
    def fixture(self, frames=60):
        x = torch.zeros(frames, 254)
        x[:, 0] = torch.arange(frames)
        x[10:40, 249] = x[10:40, 251] = 1
        return x

    def test_sampling_retains_span_endpoints_order_masks_and_original_input(self):
        x = self.fixture()
        before = x.clone()
        y = temporal_clip(x, speed_factor=.85, edge_factors=(.5, 1.5))
        self.assertTrue(torch.equal(x, before))
        self.assertTrue(torch.all(torch.diff(y[:, 0]) >= 0))
        for index in (0, 9, 10, 39, 40, 59):
            self.assertTrue(torch.any(y[:, 0] == index))
        self.assertTrue(torch.equal(y, x[y[:, 0].long()]))
        self.assertTrue(set(y[:, 249].tolist()).issubset({0., 1.}))

    def test_cap_short_sequences_and_absent_hands_do_not_create_fake_observations(self):
        x = self.fixture(128)
        y = temporal_clip(x, speed_factor=1.15, edge_factors=(1.5, 1.5))
        self.assertLessEqual(len(y), 128)
        self.assertEqual(y[0, 0], 0)
        self.assertEqual(y[-1, 0], 127)
        absent = torch.zeros(12, 254)
        self.assertTrue(torch.equal(temporal_clip(absent, speed_factor=.9, edge_factors=(1., 1.)), absent))
        one = torch.zeros(1, 254)
        one[:, 249] = 1
        self.assertEqual(len(temporal_clip(one, speed_factor=1.15, edge_factors=(1., 1.))), 1)

    def test_collation_is_seeded_preserves_targets_lengths_and_clean_mode(self):
        x = torch.zeros(128, 254)
        x[:60] = self.fixture()
        before = x.clone()
        sample = [(x, 60, 14), (x, 60, 5)]
        torch.manual_seed(42)
        a = TemporalCollator(True)(sample)
        torch.manual_seed(42)
        b = TemporalCollator(True)(sample)
        for left, right in zip(a, b, strict=True):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(a[2].tolist(), [14, 5])
        for frames, length in zip(a[0], a[1], strict=True):
            self.assertTrue(torch.all(frames[length:] == 0))
        clean = TemporalCollator(False)(sample)
        self.assertTrue(torch.equal(clean[0][0], x))
        self.assertEqual(clean[1].tolist(), [60, 60])
        self.assertTrue(torch.equal(x, before))

    def test_geometry_is_computed_from_actual_augmented_frames_and_15_logits(self):
        torch.set_num_threads(2)
        x = torch.zeros(128, 254)
        x[:60] = self.fixture()
        x[10:40, 123:249] = torch.randn(30, 126)
        frames, lengths, targets = TemporalCollator(True)([(x, 60, 14)])
        frames = augment_batch(frames)
        model = SignGeometryMeanPoolGRU(hidden_size=8, classes=15, dropout=0)
        with patch('sign_language.gru_geometry_model.geometry_batch', wraps=geometry_batch) as measured:
            logits = model(frames, lengths)
        self.assertIs(measured.call_args.args[0], frames)
        self.assertIs(measured.call_args.args[1], lengths)
        self.assertEqual(logits.shape, (1, 15))
        torch.nn.functional.cross_entropy(logits, targets).backward()
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())
