"""Aggregate dataset geometry/tracking for the three prompted signs; no training."""

import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import torch

from sign_language.dataset import write_json
from sign_language.recognition import SignRecognizer
from sign_language.recognition_diagnostics import observation_statistics, score_observations, distribution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    torch.set_num_threads(2)
    recognizer = SignRecognizer(ROOT/'models/asl-gru-128-cv-regularized-v1/official-refit/best.pt', ROOT/'models')
    raw = ROOT/'data/landmarks/asl-citizen-raw-v1-cpu-hands-pose'
    cache = ROOT/'data/training/asl-gru-128-v1'
    records = []
    for split in ('train', 'val'):
        index = json.loads((cache/f'{split}-index.json').read_text())
        groups = defaultdict(list)
        for i, row in enumerate(index):
            if row['gloss'] in ('HELLO', 'THANKYOU', 'NO'):
                groups[i//100].append((i%100, row))
        for shard, selected in groups.items():
            with ZipFile(raw/f'{split}-{shard:05d}.zip') as archive:
                metadata = json.loads(archive.read('records.json'))['records']
                for offset, row in selected:
                    record = metadata[offset]
                    assert record['video_file'] == row['video_file']
                    with np.load(io.BytesIO(archive.read(record['member'])), allow_pickle=False) as npz:
                        arrays = {key: npz[key] for key in npz.files}
                    records.append({**row, 'fps': record['fps'], 'source_frames': record['frames'],
                                    'statistics': observation_statistics(arrays, record['frame_shape']),
                                    'prediction': score_observations(recognizer, arrays, record['frame_shape'], row['gloss'])})
        print('Statistics processed:', split, len(records), flush=True)
    summaries = {}
    for name in ('HELLO', 'THANKYOU', 'NO'):
        cases = [r for r in records if r['gloss'] == name and r['split'] == 'train']
        summaries[name] = {'training_clips': len(cases),
            'duration_seconds': distribution([r['source_frames']/r['fps'] for r in cases]),
            'hand_frame_fraction': distribution([r['statistics']['hand_frame_fraction'] for r in cases]),
            'two_hand_frame_fraction': distribution([r['statistics']['two_hand_frame_fraction'] for r in cases]),
            'shoulder_width_image_fraction_median': distribution([r['statistics']['shoulder_width_image_fraction']['median'] for r in cases if r['statistics']['shoulder_width_image_fraction']]),
            'palm_size_pixels_median': distribution([r['statistics']['palm_size_pixels']['median'] for r in cases if r['statistics']['palm_size_pixels']]),
            'feature_clipped_fraction': distribution([r['statistics']['feature_clipped_fraction'] for r in cases]),
            'hidden_saturated_fraction': distribution([r['prediction']['hidden_saturated_fraction'] for r in cases])}
    write_json(args.output, {'status': 'complete', 'training_performed': False, 'media_saved': False,
                            'coordinates_saved': False, 'splits': ['train', 'val'],
                            'checkpoint_sha256': recognizer.checkpoint_sha256, 'records': records, 'training_summaries': summaries})
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
