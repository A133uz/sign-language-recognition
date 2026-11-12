import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sign_language.expanded_pilot import SIGNS, prepare_subset, digest_chunk, verify_parent_chunks
from sign_language.five_sign_pilot import sha256
from sign_language.preprocessing import CONFIG


class ExpandedSubsetTests(unittest.TestCase):
    def parent(self, root):
        root.mkdir()
        source = Path(__file__).resolve().parents[1] / 'src/sign_language/preprocessing.py'
        manifest = {'cache_id': 'fixture', 'identity': {'preprocessing': CONFIG, 'preprocessing_sha256': sha256(source)},
                    'labels': {'AA': 3, 'BB': 5}}
        (root / 'manifest.json').write_text(json.dumps(manifest))
        (root / 'status.json').write_text(json.dumps({'status': 'complete', 'cache_id': 'fixture'}))
        for split, person, count in [('train', 'P1', 4), ('val', 'P2', 2)]:
            rows = [{'gloss': 'AA' if i % 2 == 0 else 'BB', 'participant_id': person,
                     'class_id': 3 if i % 2 == 0 else 5, 'video_file': f'{split}{i}.mp4'} for i in range(count)]
            (root / f'{split}-index.json').write_text(json.dumps(rows))
            arrays = [np.zeros((count, 128, 254), np.float16), np.full(count, 12, np.int16),
                      np.asarray([r['class_id'] for r in rows], dtype=np.int64)]
            for name, values in zip(('features', 'lengths', 'targets'), arrays, strict=True):
                np.save(root / f'{split}-{name}.npy', values)
            marker = {'cache_id': 'fixture', 'start': 0, 'count': count, 'sha256': digest_chunk(arrays, 0, count)}
            (root / f'{split}-00000.prepared.json').write_text(json.dumps(marker))
        (root / 'test-index.json').write_text('THIS MUST NOT BE READ')

    def test_remaps_selected_rows_preserves_sources_and_ignores_test_split(self):
        self.assertEqual(len(SIGNS), 15)
        self.assertEqual(SIGNS[:5], ('HELLO', 'NO', 'PLEASE', 'THANKYOU', 'YES'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.parent(root / 'parent')
            before = sha256(root / 'parent/train-features.npy')
            result = prepare_subset(root / 'parent', root / 'new', ('AA', 'BB'))
            self.assertEqual(result['labels'], {'AA': 0, 'BB': 1})
            self.assertEqual(np.load(root / 'new/train-targets.npy').tolist(), [0, 1, 0, 1])
            self.assertEqual(sha256(root / 'parent/train-features.npy'), before)
            self.assertFalse((root / 'new/test-features.npy').exists())
            self.assertTrue(verify_parent_chunks(root / 'parent', result))
            with self.assertRaises(FileExistsError):
                prepare_subset(root / 'parent', root / 'new', ('AA', 'BB'))

    def test_corrupted_parent_chunk_is_rejected_before_creating_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.parent(root / 'parent')
            values = np.load(root / 'parent/train-features.npy', mmap_mode='r+')
            values[0, 0, 0] = 7
            values.flush()
            del values  # Release the fixture's Windows memory-map handle before cleanup.
            with self.assertRaisesRegex(ValueError, 'checksum'):
                prepare_subset(root / 'parent', root / 'new', ('AA', 'BB'))
            self.assertFalse((root / 'new').exists())

    def test_signer_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.parent(root / 'parent')
            path = root / 'parent/val-index.json'
            rows = json.loads(path.read_text())
            for row in rows:
                row['participant_id'] = 'P1'
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, 'Signer overlap'):
                prepare_subset(root / 'parent', root / 'new', ('AA', 'BB'))
