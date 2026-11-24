"""Safety and preprocessing checks for experimental streaming inference."""

import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch
from pathlib import Path
import json
import sys
import time

import numpy as np
import torch

from sign_language.face_gru_recognition import normalize_joined
from sign_language.streaming_face import RollingObservations, score_window

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import webcam_sliding_experiment as app


def sample(index=0, hand=True, face=True):
    body = {
        'hands_xyz': np.zeros((2, 21, 3), np.float32),
        'hand_present': np.array([hand, False]),
        'handedness': np.array([0, -1]),
        'handedness_score': np.array([1., 0.]),
        'pose_xyz': np.zeros((33, 3), np.float32),
        'pose_valid_mask': np.ones(33, bool),
    }
    body['pose_xyz'][11, :2] = [.3, .4]
    body['pose_xyz'][12, :2] = [.7, .4]
    body['pose_xyz'][15, :2] = [.4, .6]
    for joint in range(21):
        body['hands_xyz'][0, joint] = [.4 + joint * .003, .6 + joint * .002, index * .001]
    points = np.zeros((478, 3), np.float32)
    points[33, :2], points[263, :2] = [.4, .3], [.6, .3]
    return body, points, face, np.zeros(52, np.float32)


class FakeModel:
    def score(self, data, lengths):
        self.data = data.detach().clone()
        self.lengths = lengths.clone()
        return torch.tensor([[0., 2.]])


class StreamTests(unittest.TestCase):
    def test_buffer_bounds_and_copies_observations(self):
        buffer = RollingObservations(1000, max_items=8)
        for index in range(12):
            body, face, present, blend = sample(index)
            buffer.append(index * 200, body, face, present, blend, (480, 640, 3))
            body['hands_xyz'][:] = 999
        timestamps, body, *_ = buffer.snapshot()
        self.assertEqual(timestamps.tolist(), [1200, 1400, 1600, 1800, 2000, 2200])
        self.assertLess(float(body['hands_xyz'].max()), 1)
        with self.assertRaises(ValueError):
            buffer.append(2200, *sample(), (480, 640, 3))
        with self.assertRaises(ValueError):
            buffer.append(2400, *sample(), (720, 1280, 3))

    def test_window_model_input_matches_existing_normalizer(self):
        for count in (15, 150):
            with self.subTest(count=count):
                buffer = RollingObservations(20000, max_items=180)
                for index in range(count):
                    buffer.append(index * 100, *sample(index), (480, 640, 3))
                snap = buffer.snapshot()
                expected, quality = normalize_joined(snap[1], snap[2], snap[3], snap[4], snap[5])
                model = FakeModel()
                recognizer = SimpleNamespace(device=torch.device('cpu'), model=model, names=['NO', 'HELLO'])
                result = score_window(recognizer, snap)
                np.testing.assert_array_equal(model.data.numpy()[0], expected)
                self.assertEqual(result['quality'], quality)
                self.assertEqual(result['label'], 'HELLO')
                self.assertEqual(model.lengths.tolist(), [min(count, 128)])

    def test_missing_detections_do_not_yield_a_prediction(self):
        for missing in ('hand', 'face'):
            buffer = RollingObservations(2000)
            for index in range(10):
                buffer.append(index * 100, *sample(index, hand=missing != 'hand', face=missing != 'face'), (480, 640, 3))
            recognizer = SimpleNamespace(device=torch.device('cpu'), model=FakeModel(), names=['NO', 'HELLO'])
            with self.assertRaisesRegex(ValueError, 'visible'):
                score_window(recognizer, buffer.snapshot())

    def test_camera_loop_saves_only_aggregate_information(self):
        class Camera:
            def isOpened(self): return True
            def set(self, *args): pass
            def read(self): return True, np.zeros((120, 160, 3), np.uint8)
            def release(self): pass
        class Tracker:
            def __init__(self, recognizer): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def observe(self, frame, timestamp): return (*sample(), None)
        clock = iter(i * .25 for i in range(500))
        keys = iter([0] * 35 + [ord('q')])
        recognizer = SimpleNamespace(checkpoint_sha256='test', epoch=36,
                                     device=torch.device('cpu'), model=FakeModel(), names=['NO', 'HELLO'])
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'session.json'
            fake_time = SimpleNamespace(monotonic=lambda: next(clock), perf_counter=time.perf_counter)
            with patch.object(app, 'time', fake_time), patch.object(app.cv2, 'VideoCapture', return_value=Camera()), \
                 patch.object(app, 'StreamingFaceTracker', Tracker), patch.object(app, 'draw_hands', side_effect=lambda f, _: f), \
                 patch.object(app.cv2, 'namedWindow'), patch.object(app.cv2, 'imshow'), \
                 patch.object(app.cv2, 'destroyAllWindows'), patch.object(app.cv2, 'waitKey', side_effect=lambda _: next(keys)), \
                 patch.object(app.cv2, 'getWindowProperty', return_value=1):
                report = app.run(recognizer, path)
            saved = json.loads(path.read_text())
            self.assertEqual(saved, report)
            self.assertEqual(saved['status'], 'complete')
            self.assertGreater(saved['processed_frames'], 0)
            self.assertFalse(saved['video_saved'])
            self.assertFalse(saved['landmarks_saved'])
            self.assertNotIn('prediction', saved)


if __name__ == '__main__':
    unittest.main()
