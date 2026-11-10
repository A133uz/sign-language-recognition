import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sign_language.preprocessing import CONFIG, normalize_sequence

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from sign_language.gru_model import SignGRU
    from sign_language.recognition import SignRecognizer
    from sign_language.recognition_diagnostics import observation_statistics, score_observations, compare_capture
    from sign_language.recognition_webcam import run_recognition, model_display_name, state_headline
    from sign_language.stgcn_model import SignSTGCN


def observations(frames=12):
    pose = np.zeros((frames, 33, 3), np.float32)
    pose[:, 11, :2] = [.35, .4]
    pose[:, 12, :2] = [.65, .4]
    hands = np.zeros((frames, 2, 21, 3), np.float32)
    hands[:, 0, :, 0] = np.linspace(.25, .35, 21)
    hands[:, 0, :, 1] = np.linspace(.2, .4, 21)
    return {'pose_xyz': pose, 'pose_valid_mask': np.ones((frames, 33), bool),
            'hands_xyz': hands, 'hand_present': np.tile([True, False], (frames, 1)),
            'handedness': np.tile([0, -1], (frames, 1)),
            'handedness_score': np.tile([1., 0.], (frames, 1))}


@unittest.skipUnless(HAS_TORCH, 'Optional PyTorch extra not installed')
class RecognitionTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = SignRecognizer.__new__(SignRecognizer)
        self.recognizer.device = torch.device('cpu')
        self.recognizer.names = [str(i) for i in range(6)]
        self.recognizer.labels = {name: i for i, name in enumerate(self.recognizer.names)}
        self.recognizer.model = SignGRU(hidden_size=4, layers=1, classes=6).eval()

    def test_constructor_restores_saved_weights_and_label_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Path(__file__).resolve().parents[1]/'src/sign_language'
            config = {'model': {'input_size': 254, 'hidden_size': 4, 'layers': 1, 'classes': 6},
                      'preprocessing': CONFIG,
                      'source_sha256': {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                                        for name in ('gru_model.py', 'preprocessing.py')}}
            labels = {name: i for i, name in enumerate(self.recognizer.names)}
            torch.save({'config': config, 'labels': labels, 'epoch': 37,
                        'model': self.recognizer.model.state_dict()}, root/'best.pt')
            (root/'config.json').write_text(json.dumps(config))
            (root/'labels.json').write_text(json.dumps(labels))
            (root/'hand_landmarker.task').touch()
            (root/'pose_landmarker_lite.task').touch()
            restored = SignRecognizer(root/'best.pt', root, device='cpu')
            self.assertEqual(restored.names, self.recognizer.names)
            for name, value in restored.model.state_dict().items():
                self.assertTrue(torch.equal(value, self.recognizer.model.state_dict()[name]))

    def test_predictions_match_prepared_feature_precision_and_preserve_weights(self):
        raw = observations(150)
        before = {k: v.clone() for k, v in self.recognizer.model.state_dict().items()}
        result = self.recognizer.predict_observations(raw, (480, 640, 3))
        features, _ = normalize_sequence(raw, (480, 640, 3))
        batch = torch.from_numpy(features.astype(np.float16).astype(np.float32))[None]
        with torch.inference_mode():
            scores, indices = self.recognizer.model(batch, torch.tensor([128])).softmax(-1).topk(5)
        self.assertEqual(result['input_frames'], 128)
        self.assertEqual([item['label'] for item in result['top5']], [str(i) for i in indices[0].tolist()])
        np.testing.assert_allclose([item['score'] for item in result['top5']], scores[0])
        for key, value in self.recognizer.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.assertTrue(all(p.grad is None for p in self.recognizer.model.parameters()))
        self.assertFalse(self.recognizer.model.training)

    def test_absent_hands_or_shoulders_do_not_produce_arbitrary_label(self):
        for group in ('hand_present', 'pose_valid_mask'):
            raw = observations()
            raw[group][:] = False
            with self.assertRaisesRegex(ValueError, 'Hands and shoulders'):
                self.recognizer.predict_observations(raw, (480, 640, 3))

    def test_too_short_capture_is_rejected_before_creating_trackers(self):
        with self.assertRaisesRegex(ValueError, 'eight frames'):
            self.recognizer.predict_frames([np.zeros((10, 10, 3), np.uint8)], [0])

    def test_stgcn_live_input_matches_fixed_grid_used_in_cache_evaluation(self):
        self.recognizer.model = SignSTGCN(classes=6, dropout=0).eval()
        raw = observations(12)
        features, _ = normalize_sequence(raw, (480,640,3))
        expected_input = np.zeros((1,128,254), np.float32)
        expected_input[0,:12] = features.astype(np.float16).astype(np.float32)
        with torch.inference_mode():
            expected = self.recognizer.model(torch.from_numpy(expected_input),torch.tensor([12])).softmax(-1).topk(5)
        with patch.object(self.recognizer.model,'forward',wraps=self.recognizer.model.forward) as forward:
            result = self.recognizer.predict_observations(raw,(480,640,3))
        frames,lengths = forward.call_args.args
        self.assertEqual(tuple(frames.shape),(1,128,254))
        self.assertEqual(int(lengths[0]),12)
        self.assertEqual(int(torch.count_nonzero(frames[:,12:])),0)
        self.assertEqual([item['label'] for item in result['top5']], [str(i) for i in expected.indices[0].tolist()])
        np.testing.assert_allclose([item['score'] for item in result['top5']],expected.values[0],rtol=1e-6,atol=1e-8)

    def test_visible_status_uses_actual_stgcn_class_for_every_state(self):
        self.recognizer.model = SignSTGCN(classes=6, dropout=0).eval()
        name = model_display_name(self.recognizer)
        self.assertEqual(name, 'ST-GCN')
        for state in ('ready','countdown','recording','processing'):
            text = state_headline(state, 2., 3., 1., name)
            self.assertNotIn('GRU', text)
        self.assertIn('ST-GCN',state_headline('processing',2.,3.,1.,name))

    def test_diagnostic_scoring_removes_hook_and_preserves_observations(self):
        raw = observations()
        before = {key: value.copy() for key, value in raw.items()}
        stats = observation_statistics(raw, (480, 640, 3))
        result = score_observations(self.recognizer, raw, (480, 640, 3), '0')
        self.assertEqual(stats['hand_frame_fraction'], 1.)
        self.assertEqual(len(result['top5']), 5)
        self.assertEqual(len(self.recognizer.model.gru._forward_hooks), 0)
        self.assertTrue(all(p.grad is None for p in self.recognizer.model.parameters()))
        for key, value in before.items():
            np.testing.assert_array_equal(raw[key], value)
        json.dumps(stats, allow_nan=False)

    def test_comparison_scores_all_variants_on_same_capture_without_storing_arrays(self):
        raw = observations()
        metadata = {'captured_frames': 12, 'capture_seconds': 1., 'landmark_seconds': .1}
        self.recognizer.observe_frames = lambda frames, times: (raw, (480, 640, 3), metadata)
        result = compare_capture(self.recognizer, [np.zeros((10, 10, 3), np.uint8)]*12,
                                 list(range(12)), '0')
        self.assertEqual(set(result['variants']),
                         {'original', 'first_80pct', 'last_80pct', 'middle_60pct', 'mirrored_pixels'})
        self.assertEqual(result['expected'], '0')
        json.dumps(result, allow_nan=False)

    def test_camera_failure_releases_device_and_records_failure_without_media(self):
        camera = SimpleNamespace(isOpened=lambda: True, set=lambda *a: True,
                                 read=lambda: (False, None), release=lambda: None)
        recognizer = SimpleNamespace(checkpoint=Path('best.pt'), checkpoint_sha256='hash',
                                     epoch=37, names=['hello'], device='cpu', hand_model=Path('hands.task'))
        with tempfile.TemporaryDirectory() as directory, \
             patch('sign_language.recognition_webcam.cv2.VideoCapture', return_value=camera), \
             patch.object(camera, 'release') as release, \
             patch('sign_language.recognition_webcam.cv2.destroyAllWindows'):
            report = Path(directory)/'report.json'
            with self.assertRaisesRegex(RuntimeError, 'Camera stopped'):
                run_recognition(recognizer, report)
            saved = json.loads(report.read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertFalse(saved['media_saved'])
            self.assertFalse(saved['coordinates_saved'])
            self.assertFalse(saved['training_performed'])
            release.assert_called_once()
