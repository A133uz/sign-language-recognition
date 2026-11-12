"""Verify the application loader against frozen validation predictions; no camera."""
import json
from pathlib import Path
import sys

import torch
import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sign_language.recognition import SignRecognizer
from sign_language.recognition_webcam import model_display_name
from sign_language.training import LandmarkDataset
from sign_language.five_sign_pilot import write_json


def main():
    torch.set_num_threads(2)
    checkpoint = ROOT / 'models/asl-gru-five-signs-geometry-v1/geometry/best.pt'
    reference = json.loads((checkpoint.parent / 'report.json').read_text())
    recognizer = SignRecognizer(checkpoint, ROOT / 'models', device='cuda')
    if recognizer.checkpoint_sha256 != reference['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed')
    if model_display_name(recognizer) != 'GRU mean-pool + geometry':
        raise ValueError('Model display identity differs')
    batches = []
    with torch.inference_mode():
        for frames, lengths, _ in DataLoader(LandmarkDataset(ROOT / 'data/training/asl-five-signs-128-v1', 'val'), batch_size=16):
            with torch.autocast('cuda', dtype=torch.float16):
                logits = recognizer.model(frames.cuda(), lengths)
            batches.append(logits.float().softmax(1).cpu().numpy())
    actual = np.concatenate(batches)
    expected = np.asarray([p['scores'] for p in reference['validation']['predictions']])
    np.testing.assert_allclose(actual, expected, rtol=1.e-5, atol=1.e-6)
    if not np.array_equal(actual.argmax(1), expected.argmax(1)):
        raise AssertionError('Runtime predictions differ')
    report = {'status': 'verified', 'checkpoint': str(checkpoint),
              'checkpoint_sha256': recognizer.checkpoint_sha256, 'epoch': recognizer.epoch,
              'model_class': type(recognizer.model).__name__, 'display_model': model_display_name(recognizer),
              'classes': recognizer.names, 'device': str(recognizer.device),
              'compared_validation_rows': len(actual), 'max_score_difference': float(np.abs(actual - expected).max()),
              'all_predictions_match': True, 'camera_opened': False,
              'training_performed': False, 'test_split_read': False}
    write_json(ROOT / '.cache/gru-geometry-runtime-20261112.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
