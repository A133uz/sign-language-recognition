import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    import numpy as np
    from sign_language.gru_geometry_model import (
        FEATURES, SignGeometryMeanPoolGRU, fit_geometry_statistics, geometry_batch,
    )
    from sign_language.gru_mean_pool_model import SignMeanPoolGRU
    from sign_language.hand_articulation import summarize_articulation
    from sign_language.recognition import SignRecognizer
    from sign_language.recognition_webcam import model_display_name
    from sign_language.preprocessing import CONFIG
    from sign_language.five_sign_pilot import sha256


@unittest.skipUnless(HAS_TORCH, 'Optional PyTorch extra unavailable')
class GeometryFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_initialization_and_rng_match_control_with_only_85_more_weights(self):
        args = dict(hidden_size=8, layers=2, classes=5, dropout=.5)
        torch.manual_seed(42)
        control = SignMeanPoolGRU(**args)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        candidate = SignGeometryMeanPoolGRU(**args)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for name, tensor in control.gru.state_dict().items():
            torch.testing.assert_close(tensor, candidate.gru.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(control.classifier.weight, candidate.classifier.weight[:, :8])
        self.assertEqual(sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in control.parameters()), 85)
        control.eval()
        candidate.eval()
        frames = torch.randn(2, 12, 254)
        lengths = torch.tensor([8, 12])
        with torch.inference_mode():
            torch.testing.assert_close(control(frames, lengths), candidate(frames, lengths, zero_geometry=True))

    def test_summary_ignores_padding_and_matches_existing_measurements(self):
        frames = torch.zeros(2, 16, 254)
        frames[:, :8, 249] = 1
        frames[:, :8, 251] = 1
        frames[:, :8, 123:249] = torch.randn(2, 8, 126)
        lengths = torch.tensor([8, 8])
        values = geometry_batch(frames, lengths)
        changed = frames.clone()
        changed[:, 8:] = float('nan')
        np.testing.assert_allclose(values, geometry_batch(changed, lengths), equal_nan=True)
        summary = summarize_articulation(frames[0, :8].numpy())
        expected = [np.nan if summary['metrics'][k] is None else summary['metrics'][k] for k in FEATURES]
        np.testing.assert_allclose(values[0], expected, rtol=1.e-6, equal_nan=True)
        self.assertTrue(np.isnan(geometry_batch(torch.zeros(1, 4, 254), torch.tensor([4]))).all())
        with self.assertRaises(ValueError):
            geometry_batch(frames, torch.tensor([0, 8]))

    def test_train_statistics_impute_missing_values_and_remain_frozen(self):
        x = np.tile(np.array([0., 2., 4.])[:, None], (1, 17))
        x[0, 0] = np.nan
        x[:, 1] = np.nan
        statistics = fit_geometry_statistics(x)
        self.assertEqual(statistics['median'][0], 3.)
        self.assertEqual(statistics['mean'][0], 3.)
        self.assertEqual(statistics['scale'][1], 1.)
        model = SignGeometryMeanPoolGRU(hidden_size=8)
        model.set_geometry_statistics(statistics)
        before = {k: v.clone() for k, v in model.named_buffers()}
        normalized = model.standardize_geometry(torch.tensor(x, dtype=torch.float32))
        self.assertTrue(torch.isfinite(normalized).all())
        model.standardize_geometry(torch.full((1, 17), 1.e6))
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, before[name])
        with self.assertRaises(ValueError):
            model.set_geometry_statistics({**statistics, 'scale': [0.] * 17})

    def test_gradients_reach_gru_and_geometry_weights_but_not_padded_frames(self):
        model = SignGeometryMeanPoolGRU(hidden_size=8, layers=2, dropout=0)
        frames = torch.randn(2, 12, 254, requires_grad=True)
        geometry = torch.randn(2, 17)
        loss = torch.nn.functional.cross_entropy(model(frames, torch.tensor([7, 12]), geometry), torch.tensor([0, 4]))
        loss.backward()
        self.assertGreater(model.classifier.weight.grad[:, 8:].abs().sum(), 0)
        self.assertGreater(model.gru.weight_ih_l0.grad.abs().sum(), 0)
        self.assertGreater(model.gru.weight_ih_l1.grad.abs().sum(), 0)
        self.assertEqual(frames.grad[0, 7:].abs().sum(), 0)

    def test_checkpoint_roundtrip_preserves_statistics_logits_and_ablation(self):
        model = SignGeometryMeanPoolGRU(hidden_size=8, dropout=0).eval()
        with torch.no_grad():
            model.classifier.weight[:, 8:] = .2
        model.set_geometry_statistics({'median': [1.] * 17, 'mean': [2.] * 17, 'scale': [3.] * 17})
        restored = SignGeometryMeanPoolGRU(hidden_size=8, dropout=0).eval()
        restored.load_state_dict(model.state_dict())
        frames = torch.randn(2, 9, 254)
        lengths = torch.tensor([9, 7])
        geometry = torch.randn(2, 17)
        with torch.inference_mode():
            torch.testing.assert_close(model(frames, lengths, geometry), restored(frames, lengths, geometry))
            self.assertFalse(torch.equal(model(frames, lengths, geometry), model(frames, lengths, zero_geometry=True)))

    def test_application_loads_geometry_architecture_and_rejects_schema_or_statistics_mismatch(self):
        source = Path(__file__).resolve().parents[1] / 'src/sign_language'
        statistics = {'median': [1.] * 17, 'mean': [2.] * 17, 'scale': [3.] * 17}
        config = {'architecture': 'gru-mean-pool-geometry-v1',
                  'model': {'hidden_size': 8, 'layers': 2, 'classes': 5, 'dropout': .5},
                  'preprocessing': CONFIG, 'geometry_features': list(FEATURES),
                  'geometry_statistics': statistics,
                  'source_sha256': {n: sha256(source / n) for n in
                      ('preprocessing.py', 'gru_model.py', 'gru_mean_pool_model.py',
                       'gru_geometry_model.py', 'hand_articulation.py')}}
        model = SignGeometryMeanPoolGRU(**config['model']).eval()
        model.set_geometry_statistics(statistics)
        labels = {g: i for i, g in enumerate(('HELLO', 'NO', 'PLEASE', 'THANKYOU', 'YES'))}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'hand_landmarker.task').touch()
            (root / 'pose_landmarker_lite.task').touch()
            (root / 'labels.json').write_text(json.dumps(labels))

            def save():
                torch.save({'config': config, 'model': model.state_dict(), 'labels': labels, 'epoch': 8}, root / 'best.pt')
                (root / 'config.json').write_text(json.dumps(config))

            save()
            loaded = SignRecognizer(root / 'best.pt', root, device='cpu')
            self.assertEqual(model_display_name(loaded), 'GRU mean-pool + geometry')
            self.assertEqual(loaded.epoch, 8)
            self.assertFalse(loaded.model.training)
            frames = torch.zeros(1, 12, 254)
            lengths = torch.tensor([9])
            with torch.inference_mode():
                torch.testing.assert_close(model(frames, lengths), loaded.model(frames, lengths))
            config['geometry_features'] = list(reversed(FEATURES))
            save()
            with self.assertRaisesRegex(ValueError, 'schema'):
                SignRecognizer(root / 'best.pt', root, device='cpu')
            config['geometry_features'] = list(FEATURES)
            config['geometry_statistics']['scale'][0] = 8.
            save()
            with self.assertRaisesRegex(ValueError, 'normalization'):
                SignRecognizer(root / 'best.pt', root, device='cpu')
