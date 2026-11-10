import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sign_language.preprocessing import INPUT_SIZE, normalize_sequence


def fixture(frames=8):
    pose = np.zeros((frames, 33, 3), dtype=np.float32)
    pose[:, :, :2] = [.5, .4]
    pose[:, 11, :2] = [.35, .4]; pose[:, 12, :2] = [.65, .4]
    pose[:, 15, :2] = [.3, .6]; pose[:, 16, :2] = [.7, .6]
    hands = np.zeros((frames, 2, 21, 3), dtype=np.float32)
    for slot, x in enumerate((.3, .7)):
        hands[:, slot, :, 0] = x+np.arange(21)*.002
        hands[:, slot, :, 1] = .6+np.arange(21)*.003
        hands[:, slot, :, 2] = np.arange(21)*.001
    return {'hands_xyz': hands, 'hand_present': np.ones((frames, 2), bool),
            'handedness': np.tile([0, 1], (frames, 1)),
            'handedness_score': np.ones((frames, 2), dtype=np.float32),
            'pose_xyz': pose, 'pose_valid_mask': np.ones((frames, 33), bool)}


class PreparationTests(unittest.TestCase):
    def test_atomic_json_retries_transient_windows_lock(self):
        from sign_language.dataset import write_json
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'status.json'
            target.write_text('{"old": true}')
            replace = Path.replace
            attempts = []

            def temporarily_locked(source, destination):
                attempts.append(1)
                if len(attempts) <= 2:
                    self.assertEqual(json.loads(target.read_text()), {'old': True})
                    raise PermissionError('fixture reader holds destination')
                return replace(source, destination)

            with patch.object(Path, 'replace', temporarily_locked), patch('sign_language.dataset.time.sleep'):
                write_json(target, {'new': True})
            self.assertEqual(json.loads(target.read_text()), {'new': True})
            self.assertEqual(len(attempts), 3)

    def test_prepared_cache_resumes_checksums_and_repairs_damaged_slice(self):
        from sign_language import training_cache as cache
        from sign_language.extraction import commit_shard, fingerprint

        class SerialPool:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def map(self, function, jobs):
                return map(function, jobs)

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); raw = base/'raw'; raw.mkdir()
            rows = [{'video_file': f'{split}.mp4', 'participant_id': split, 'gloss': 'HELLO',
                     'split': split, 'class_id': 0} for split in ('train', 'val', 'test')]
            (raw/'run-manifest.json').write_text(json.dumps({'run_id': 'fixture', 'identity': {'csv_sha256': {}}}))
            (raw/'summary.json').write_text(json.dumps({'status': 'complete', 'failed_clips': 0}))
            (raw/'labels.json').write_text(json.dumps({'HELLO': 0}))
            buffer = io.BytesIO(); np.savez_compressed(buffer, **fixture(3))
            for row in rows:
                archive = base/'local.zip'
                with zipfile.ZipFile(archive, 'w') as z:
                    z.writestr('clip.npz', buffer.getvalue())
                    z.writestr('records.json', json.dumps({'run_id': 'fixture', 'rows_sha256': fingerprint([row]),
                        'records': [{**row, 'status': 'ok', 'member': 'clip.npz', 'frame_shape': [480, 640, 3]}]}))
                commit_shard(archive, raw, f'{row["split"]}-00000', 'fixture', [row])
            output = base/'prepared'
            with patch.object(cache, 'load_rows', return_value=(rows, {'HELLO': 0}, {})), \
                 patch.object(cache, 'ProcessPoolExecutor', return_value=SerialPool()), \
                 patch.object(cache, 'prepare_shard', wraps=cache.prepare_shard) as prepare:
                first = cache.prepare_cache(raw, base/'dataset', output)
                self.assertEqual(first['processed'], 3)
                self.assertEqual(prepare.call_count, 3)
                self.assertEqual(int(np.load(output/'train-lengths.npy')[0]), 3)
                self.assertTrue((np.load(output/'train-features.npy')[0, 3:] == 0).all())
                cache.prepare_cache(raw, base/'dataset', output)
                self.assertEqual(prepare.call_count, 3)
                damaged = np.load(output/'train-features.npy', mmap_mode='r+')
                damaged[0, 0, 0] = 7.; damaged.flush(); del damaged
                cache.prepare_cache(raw, base/'dataset', output)
                self.assertEqual(prepare.call_count, 4)
                (raw/'train-00000.zip').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'integrity'):
                    cache.prepare_cache(raw, base/'dataset', output)

    def test_translation_and_scale_invariance_without_mutating_raw(self):
        raw = fixture(); original = raw['hands_xyz'].copy()
        expected, _ = normalize_sequence(raw, [480, 640, 3])
        transformed = {k: v.copy() for k, v in raw.items()}
        for key in ('hands_xyz', 'pose_xyz'):
            transformed[key] *= .8
            transformed[key][..., :2] += [.07, .03]
        actual, _ = normalize_sequence(transformed, [480, 640, 3])
        np.testing.assert_allclose(actual, expected, atol=2e-5)
        np.testing.assert_array_equal(raw['hands_xyz'], original)

    def test_detection_order_swaps_do_not_change_features(self):
        raw = fixture(); expected, _ = normalize_sequence(raw, [480, 640, 3])
        for key in ('hands_xyz', 'hand_present', 'handedness', 'handedness_score'):
            raw[key][1::2] = raw[key][1::2, ::-1]
        actual, _ = normalize_sequence(raw, [480, 640, 3])
        np.testing.assert_allclose(actual, expected)

    def test_missing_hands_and_shoulders_are_explicit_not_fake_geometry(self):
        raw = fixture(); raw['hand_present'][:] = False; raw['pose_valid_mask'][:] = False
        feature, quality = normalize_sequence(raw, [480, 640, 3])
        self.assertEqual(feature.shape, (8, INPUT_SIZE))
        self.assertTrue((feature == 0).all())
        self.assertEqual(quality['frames_with_body_anchor'], 0)

    def test_resampling_covers_whole_sign_and_short_signs_are_not_upsampled(self):
        raw = fixture(300)
        motion = np.linspace(0, .1, 300)
        raw['hands_xyz'][:, 0, :, 0] += motion[:, None]
        feature, _ = normalize_sequence(raw, [480, 640, 3])
        self.assertEqual(len(feature), 128)
        # First body's hand-x coordinate starts after 26 pose XY and 13 masks.
        self.assertAlmostEqual(float(feature[-1, 39]-feature[0, 39]), .1/.3, places=5)
        short, _ = normalize_sequence(fixture(3), [480, 640, 3])
        self.assertEqual(len(short), 3)

    def test_nonfinite_coordinates_are_rejected(self):
        raw = fixture(); raw['hands_xyz'][0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            normalize_sequence(raw, [480, 640, 3])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional PyTorch extra is not installed')
class GRUTests(unittest.TestCase):
    def test_cuda_scaler_skips_overflow_without_changing_weights_then_recovers(self):
        import torch
        from sign_language.training import backward_step
        if not torch.cuda.is_available():
            self.skipTest('CUDA not available')
        model = torch.nn.Linear(1, 1, bias=False).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        scaler = torch.amp.GradScaler('cuda')
        before = model.weight.detach().clone()
        finite_loss = model.weight.sum()*1e35
        self.assertTrue(torch.isfinite(finite_loss))
        self.assertFalse(backward_step(model, optimizer, finite_loss, scaler))
        torch.testing.assert_close(model.weight, before)
        self.assertLess(scaler.get_scale(), 65536)
        optimizer.zero_grad(set_to_none=True)
        self.assertTrue(backward_step(model, optimizer, model.weight.square().sum(), scaler))
        self.assertFalse(torch.equal(model.weight, before))

    def test_padding_is_ignored_and_unsorted_batch_order_is_preserved(self):
        import torch
        from sign_language.gru_model import SignGRU
        torch.manual_seed(1)
        model = SignGRU(input_size=4, hidden_size=8, layers=2, classes=3, dropout=0).eval()
        frames = torch.randn(2, 9, 4); lengths = torch.tensor([3, 7])
        with torch.no_grad():
            expected = model(frames, lengths)
            frames[0, 3:] = 1000; frames[1, 7:] = -1000
            actual = model(frames, lengths)
            single = model(frames[0:1, :3], torch.tensor([3]))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(single[0], expected[0])

    def test_backward_and_atomic_checkpoint_roundtrip(self):
        import torch
        from sign_language.gru_model import SignGRU
        from sign_language.training import atomic_checkpoint
        model = SignGRU(input_size=4, hidden_size=8, layers=1, classes=3, dropout=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        frames = torch.randn(4, 8, 4); lengths = torch.tensor([8, 7, 6, 5]); targets = torch.tensor([0, 1, 2, 0])
        before = model.classifier.weight.detach().clone()
        loss = torch.nn.functional.cross_entropy(model(frames, lengths), targets)
        loss.backward(); optimizer.step()
        self.assertFalse(torch.equal(before, model.classifier.weight))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'last.pt'
            atomic_checkpoint(path, {'model': model.state_dict(), 'epoch': 1})
            state = torch.load(path, weights_only=True)
            restored = SignGRU(input_size=4, hidden_size=8, layers=1, classes=3, dropout=0)
            restored.load_state_dict(state['model'])
            torch.testing.assert_close(model(frames, lengths), restored(frames, lengths))
            self.assertFalse(path.with_suffix('.tmp').exists())


if __name__ == '__main__':
    unittest.main()
