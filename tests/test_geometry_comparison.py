import importlib.util
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    import numpy as np
    from sign_language.geometry_comparison import compare_geometry_capture
    from sign_language.gru_mean_pool_model import SignMeanPoolGRU
    from sign_language.gru_geometry_model import SignGeometryMeanPoolGRU


@unittest.skipUnless(HAS_TORCH, 'Optional PyTorch extra unavailable')
class GeometryComparisonTests(unittest.TestCase):
    def test_same_capture_branch_intervention_explains_margin_without_exporting_inputs(self):
        torch.set_num_threads(2)
        labels = {'HELLO': 0, 'NO': 1, 'PLEASE': 2, 'THANKYOU': 3, 'YES': 4}
        features = np.zeros((12, 254), np.float32)
        features[:, 249] = features[:, 251] = 1
        points = features[:, 123:249].reshape(12, 2, 21, 3)
        points[:, 0, 8, 0] = 1
        points[:, 0, 9, :2] = [1, 1]
        control_model = SignMeanPoolGRU(hidden_size=8, classes=5, dropout=0).eval()
        candidate_model = SignGeometryMeanPoolGRU(hidden_size=8, dropout=0).eval()
        with torch.no_grad():
            for model in (control_model, candidate_model):
                for p in model.parameters():
                    p.zero_()
                model.classifier.bias[0] = 1
            candidate_model.classifier.weight[1, 8] = 2
        arrays = {'hand_present': np.tile([True, False], (12, 1))}
        observation = Mock(return_value=(arrays, (480, 640, 3), {'captured_frames': 12}))
        control = SimpleNamespace(labels=labels, names=list(labels), model=control_model,
                                  device=torch.device('cpu'), observe_frames=observation,
                                  config={'architecture': 'gru-mean-pool-v1', 'cache_id': 'same'})
        candidate = SimpleNamespace(labels=labels, names=list(labels), model=candidate_model,
                                    device=torch.device('cpu'),
                                    config={'architecture': 'gru-mean-pool-geometry-v1', 'cache_id': 'same'})
        quality = {'frames_with_hands': 12, 'frames_with_body_anchor': 12}
        before = {k: v.clone() for k, v in candidate_model.state_dict().items()}
        with patch('sign_language.geometry_comparison.normalize_sequence', return_value=(features, quality)):
            result = compare_geometry_capture(control, candidate, [None] * 12, list(range(12)), 'NO')
            self.assertEqual(observation.call_count, 1)
            other = compare_geometry_capture(control, candidate, [None] * 12, list(range(12)), 'HELLO')
        variants = result['variants']
        self.assertEqual(variants['original']['top5'][0]['label'], 'HELLO')
        self.assertEqual(variants['mean_pool']['top5'][0]['label'], 'NO')
        self.assertEqual(variants['geometry_off']['top5'][0]['label'], 'HELLO')
        for key in variants:
            self.assertEqual(variants[key]['top5'], other['variants'][key]['top5'])
        margin = result['hello_minus_no_margin']
        self.assertAlmostEqual(margin['full_model'], -1)
        self.assertAlmostEqual(margin['geometry_disabled_model'], 1)
        self.assertAlmostEqual(margin['actual_geometry_effect'], -2)
        self.assertAlmostEqual(margin['geometry_component_fp32'], -2)
        self.assertAlmostEqual(margin['amp_decomposition_residual'], 0)
        json.dumps(result, allow_nan=False)  # No tensors, coordinate or hidden arrays.
        self.assertEqual(len(candidate_model.classifier._forward_pre_hooks), 0)
        for name, value in candidate_model.state_dict().items():
            torch.testing.assert_close(value, before[name])
        self.assertTrue(all(p.grad is None for p in candidate_model.parameters()))

    def test_comparison_refuses_different_training_caches(self):
        a = SimpleNamespace(labels={'NO': 0}, config={'cache_id': 'a'})
        b = SimpleNamespace(labels={'NO': 0}, config={'cache_id': 'b'})
        with self.assertRaises(ValueError):
            compare_geometry_capture(a, b, [], [], 'NO')
