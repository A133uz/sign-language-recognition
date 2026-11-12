"""Check the actual 15-label application loader against frozen validation scores."""
import json
from pathlib import Path
import sys

import torch
import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sign_language.expanded_pilot import SIGNS
from sign_language.five_sign_pilot import write_json
from sign_language.recognition import SignRecognizer
from sign_language.training import LandmarkDataset


def main():
    torch.set_num_threads(2)
    results = {}
    for name in ('control', 'temporal'):
        root = ROOT / 'models/asl-gru-fifteen-signs-temporal-v1' / name
        reference = json.loads((root / 'report.json').read_text())
        recognizer = SignRecognizer(root / 'best.pt', ROOT / 'models', device='cuda')
        if recognizer.names != list(SIGNS) or recognizer.checkpoint_sha256 != reference['checkpoint_sha256']:
            raise ValueError('Runtime checkpoint or label order differs')
        probabilities = []
        with torch.inference_mode():
            for frames, lengths, _ in DataLoader(LandmarkDataset(ROOT / 'data/training/asl-fifteen-signs-128-v1', 'val'), batch_size=16):
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = recognizer.model(frames.cuda(), lengths)
                probabilities.append(logits.float().softmax(1).cpu().numpy())
        actual = np.concatenate(probabilities)
        expected = np.asarray([p['scores'] for p in reference['validation']['predictions']])
        np.testing.assert_allclose(actual, expected, rtol=1.e-5, atol=1.e-6)
        if not np.array_equal(actual.argmax(1), expected.argmax(1)):
            raise AssertionError('Runtime predictions differ')
        results[name] = {'epoch': recognizer.epoch, 'checkpoint_sha256': recognizer.checkpoint_sha256,
                         'classes': len(recognizer.names), 'compared_rows': len(actual),
                         'max_score_difference': float(np.abs(actual-expected).max()), 'all_predictions_match': True}
        del recognizer
        torch.cuda.empty_cache()
    write_json(ROOT / '.cache/fifteen-sign-runtime-20261112.json',
               {'status': 'verified', 'runs': results, 'webcam_opened': False, 'training_performed': False, 'test_read': False})
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
