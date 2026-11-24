"""Storage invariants for consented local landmark collection."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import time
from types import SimpleNamespace
from unittest.mock import patch
import sys

import numpy as np

from sign_language.activity_collection import SHAPES, save_trial, validate_trial
from test_streaming_face import sample
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import collect_activity_landmarks as collector
from verify_activity_collection import inspect
from analyze_activity_collection import analyze


def observation(i):
    item = sample(i)[0]
    item.update(pose_visibility=np.ones(33, np.float32),
                pose_presence=np.ones(33, np.float32), pose_detected=np.asarray(True))
    return item


class ActivityCollectionTests(unittest.TestCase):
    def test_numeric_only_roundtrip_and_no_overwrite(self):
        with TemporaryDirectory() as directory:
            values = [observation(i) for i in range(10)]
            metadata = save_trial(directory, 1, 'HELLO', 'known_sign', values,
                                  list(range(0, 1000, 100)), (480, 640, 3))
            self.assertTrue(validate_trial(directory, metadata))
            self.assertEqual(metadata['prompt_confirmation'], 'pending')
            self.assertFalse(metadata['video_saved'])
            self.assertFalse(metadata['face_mesh_saved'])
            with np.load(Path(directory)/metadata['file'], allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), set(SHAPES) | {'timestamp_ms'})
                self.assertNotIn('face_xyz', archive.files)
                self.assertNotIn('image', archive.files)
            self.assertEqual(json.loads((Path(directory)/'trial-001.json').read_text())['sha256'], metadata['sha256'])
            with self.assertRaises(FileExistsError):
                save_trial(directory, 1, 'HELLO', 'known_sign', values, list(range(0, 1000, 100)), (480, 640, 3))

    def test_invalid_or_unexpected_data_never_saved(self):
        with TemporaryDirectory() as directory:
            values = [observation(i) for i in range(10)]
            values[0]['image'] = np.zeros((10, 10, 3))
            with self.assertRaises(ValueError):
                save_trial(directory, 1, 'HELLO', 'known_sign', values, list(range(0,1000,100)), (480,640,3))
            self.assertEqual(list(Path(directory).iterdir()), [])
            del values[0]['image']
            with self.assertRaises(ValueError):
                save_trial(directory, 1, 'HELLO', 'known_sign', values, [0]*10, (480,640,3))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_tampering_is_detected(self):
        with TemporaryDirectory() as directory:
            metadata = save_trial(directory, 1, 'REST', 'rest', [observation(i) for i in range(10)],
                                  list(range(0, 1000, 100)), (480, 640, 3))
            (Path(directory)/metadata['file']).write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'hash'): validate_trial(directory, metadata)

    def test_simulated_capture_writes_only_consented_observations(self):
        class Camera:
            def isOpened(self): return True
            def set(self, *args): pass
            def read(self): return True, np.zeros((120, 160, 3), np.uint8)
            def release(self): pass
        class Tracker:
            def __init__(self, recognizer): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def observe(self, frame, timestamp): return observation(0), None, False, None, None
        ticks = iter(i*.1 for i in range(1000))
        keys = iter([ord(' ')] + [0]*300)
        fake_time = SimpleNamespace(monotonic=lambda: next(ticks))
        recognizer = SimpleNamespace(checkpoint_sha256='fake')
        with TemporaryDirectory() as directory:
            with patch.object(collector, 'time', fake_time), \
                 patch.object(collector, 'PROMPTS', [('HELLO','known_sign','Sign once')]), \
                 patch.object(collector.cv2, 'VideoCapture', return_value=Camera()), \
                 patch.object(collector, 'StreamingFaceTracker', Tracker), \
                 patch.object(collector, 'draw_hands', side_effect=lambda frame, _: frame), \
                 patch.object(collector.cv2, 'namedWindow'), patch.object(collector.cv2, 'imshow'), \
                 patch.object(collector.cv2, 'waitKey', side_effect=lambda _: next(keys)), \
                 patch.object(collector.cv2, 'getWindowProperty', return_value=1), \
                 patch.object(collector.cv2, 'destroyAllWindows'):
                manifest = collector.run(recognizer, directory, duration=3)
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['prompt_confirmation'], 'pending')
            self.assertTrue(validate_trial(directory, manifest['trials'][0]))
            self.assertEqual({p.suffix for p in Path(directory).iterdir()}, {'.json', '.npz'})
            summary = inspect(directory)
            self.assertEqual(summary['trials'][0]['prompt'], 'HELLO')
            self.assertGreater(summary['trials'][0]['frames_with_hands'], 0)
            (Path(directory)/'user-confirmation.json').write_text(json.dumps({
                'manifest_sha256': summary['manifest_sha256'], 'confirmed_trials': [1]}))
            result = analyze(directory)
            self.assertTrue(result['one_signer_only'])
            self.assertFalse(result['sign_boundaries_annotated'])
            self.assertGreater(result['trials'][0]['windows'], 0)
            (Path(directory)/'user-confirmation.json').write_text(json.dumps({
                'manifest_sha256': 'wrong', 'confirmed_trials': [1]}))
            with self.assertRaisesRegex(ValueError, 'confirmation|Confirmation'):
                analyze(directory)


if __name__ == '__main__': unittest.main()
