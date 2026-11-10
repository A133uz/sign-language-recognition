import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sign_language import extraction
from sign_language.raw_landmarks import frame_observation
from sign_language.raw_landmarks import extract_clip


class ObservationTests(unittest.TestCase):
    def test_partial_decode_is_rejected_and_camera_is_released(self):
        import cv2
        camera = SimpleNamespace(isOpened=lambda: True,
                                 get=lambda key: 30 if key == cv2.CAP_PROP_FPS else 2,
                                 read=lambda: None, release=lambda: None)
        with patch('sign_language.raw_landmarks.cv2.VideoCapture', return_value=camera), \
             patch.object(camera, 'read', side_effect=[(True, np.zeros((32,32,3), np.uint8)), (False,None)]), \
             patch.object(camera, 'release') as release, \
             patch('sign_language.raw_landmarks.HandTracker') as tracker:
            tracker.return_value.__enter__.return_value.detect.return_value = SimpleNamespace(
                hand_landmarks=[], handedness=[])
            with self.assertRaisesRegex(ValueError, 'Incomplete decode'):
                extract_clip(Path('fixture.mp4'), Path('fixture.task'), None)
        release.assert_called_once()

    def test_absence_and_detection_order_are_explicit(self):
        empty = frame_observation(SimpleNamespace(hand_landmarks=[], handedness=[]))
        self.assertFalse(empty['hand_present'].any())
        self.assertTrue((empty['handedness'] == -1).all())
        point = SimpleNamespace(x=.25, y=.4, z=-.1)
        result = SimpleNamespace(
            hand_landmarks=[[point] * 21, [point] * 21],
            handedness=[[SimpleNamespace(category_name='Right', score=.9)],
                        [SimpleNamespace(category_name='Left', score=.8)]],
        )
        observed = frame_observation(result)
        self.assertEqual(observed['handedness'].tolist(), [1, 0])
        self.assertEqual(observed['hands_xyz'].shape, (2, 21, 3))
        self.assertEqual(observed['hands_xyz'].dtype, np.float32)

    def test_low_confidence_pose_is_retained_but_masked(self):
        point = SimpleNamespace(x=.2, y=.3, z=.1, visibility=.2, presence=.9)
        observed = frame_observation(
            SimpleNamespace(hand_landmarks=[], handedness=[]),
            SimpleNamespace(pose_landmarks=[[point] * 33]),
        )
        self.assertTrue(observed['pose_detected'])
        self.assertFalse(observed['pose_valid_mask'].any())
        self.assertAlmostEqual(float(observed['pose_xyz'][0, 0]), .2)


class ResumeTests(unittest.TestCase):
    def test_duplicate_writer_is_rejected_and_lock_is_released(self):
        with tempfile.TemporaryDirectory() as directory:
            work, output = Path(directory)/'work', Path(directory)/'out'
            with extraction.output_lock(work, output):
                with self.assertRaisesRegex(RuntimeError, 'Another extractor'):
                    with extraction.output_lock(work, output):
                        self.fail('Second writer acquired the lock')
            with extraction.output_lock(work, output):
                pass

    def test_resume_damage_repair_failure_reporting_and_config_guard(self):
        rows = [
            {'video_file': f'{i}.mp4', 'participant_id': str(i), 'gloss': 'HELLO',
             'split': 'train', 'class_id': 0} for i in range(3)
        ]
        calls = []
        failed_name = extraction.shard_plan(rows, ['train'], 2, None)[-1][1][-1]['video_file']

        def worker(job):
            row = job[0]
            calls.append(row['video_file'])
            if row['video_file'] == failed_name:
                return {**row, 'status': 'failed', 'error': 'fixture decode failure'}, None
            buffer = io.BytesIO()
            np.savez_compressed(buffer, hands_xyz=np.zeros((2, 2, 21, 3), np.float32))
            return {**row, 'status': 'ok', 'frames': 2,
                    'frames_with_hands': 1, 'frames_with_pose': 1}, buffer.getvalue()

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / 'raw'
            (root / 'videos').mkdir(parents=True)
            output = base / 'features'
            args = dict(root=root, output=output, model_dir=base/'models', work_dir=base/'work',
                        splits=['train'], shard_size=2)
            with patch.object(extraction, 'load_rows', return_value=(rows, {'HELLO': 0}, {'train': 'fixture'})), \
                 patch.object(extraction, 'setup_models', return_value=({'hands': 'fixture'}, {'hands': {'sha256': 'fixture'}})), \
                 patch.object(extraction, 'extract_job', side_effect=worker):
                first = extraction.extract_dataset_landmarks(**args, max_shards=1)
                self.assertEqual(first['status'], 'partial')
                self.assertEqual(first['successful_clips'], 2)
                self.assertEqual(len(calls), 2)
                second = extraction.extract_dataset_landmarks(**args)
                self.assertEqual(second['status'], 'complete_with_failures')
                self.assertEqual(second['failed_clips'], 1)
                self.assertEqual(len(calls), 3)
                self.assertFalse(second['training_performed'])
                # Same configuration must do no extraction on resume.
                extraction.extract_dataset_landmarks(**args)
                self.assertEqual(len(calls), 3)
                # A damaged committed archive must not be silently skipped.
                damaged = output / 'train-00000.zip'
                damaged.write_bytes(b'corrupt')
                extraction.extract_dataset_landmarks(**args)
                self.assertGreater(len(calls), 3)
                with self.assertRaisesRegex(ValueError, 'configuration differs'):
                    extraction.extract_dataset_landmarks(**{**args, 'include_pose': False})
                with self.assertRaisesRegex(ValueError, 'Raw|raw|outside'):
                    extraction.extract_dataset_landmarks(**{**args, 'output': root/'features'})
            for path in output.glob('*.zip'):
                with zipfile.ZipFile(path) as archive:
                    self.assertIsNone(archive.testzip())
                    info = json.loads(archive.read('records.json'))
                    for record in info['records']:
                        if record['status'] == 'ok':
                            with np.load(io.BytesIO(archive.read(record['member'])), allow_pickle=False) as arrays:
                                self.assertEqual(arrays['hands_xyz'].shape, (2, 2, 21, 3))
