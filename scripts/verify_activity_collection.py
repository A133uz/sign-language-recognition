"""Validate a local recording and optionally attach user prompt confirmation."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sign_language.activity_collection import validate_trial
from sign_language.dataset import write_json


def inspect(directory):
    directory = Path(directory)
    manifest_path = directory / 'session.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    trials = []
    for item in manifest['trials']:
        if Path(item['file']).name != item['file']:
            raise ValueError('Unsafe trial path')
        metadata = json.loads((directory / f"trial-{item['trial']:03d}.json").read_text(encoding='utf-8'))
        if metadata != item:
            raise ValueError('Manifest and trial metadata differ')
        validate_trial(directory, item)
        with np.load(directory / item['file'], allow_pickle=False) as data:
            hand = data['hand_present'].any(axis=1)
            shoulders = data['pose_valid_mask'][:, [11, 12]].all(axis=1)
            trials.append({'trial': item['trial'], 'prompt': item['prompt'],
                           'category': item['category'], 'frames': len(hand),
                           'frames_with_hands': int(hand.sum()),
                           'frames_with_raw_shoulders': int(shoulders.sum()),
                           'duration_ms': item['duration_ms']})
    numbers = [item['trial'] for item in trials]
    if numbers != list(range(1, len(trials)+1)):
        raise ValueError('Duplicate or missing trial numbers')
    return {'status': manifest['status'], 'prompt_confirmation': manifest['prompt_confirmation'],
            'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'video_saved': manifest['video_saved'], 'face_mesh_saved': manifest['face_mesh_saved'],
            'trials': trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--confirm-all', action='store_true',
                        help='Only after the user confirms all prompts were correctly performed')
    args = parser.parse_args()
    summary = inspect(args.directory)
    if args.confirm_all:
        if summary['status'] != 'complete' or len(summary['trials']) != 14:
            parser.error('Cannot confirm a partial session')
        destination = args.directory / 'user-confirmation.json'
        if destination.exists(): parser.error('Confirmation already exists')
        write_json(destination, {'manifest_sha256': summary['manifest_sha256'],
                   'confirmation': 'All 14 prompts completed normally',
                   'confirmed_trials': [item['trial'] for item in summary['trials']],
                   'sign_boundaries_annotated': False})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
