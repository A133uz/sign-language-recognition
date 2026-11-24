"""Summarize coarse motion proxies from confirmed local recordings; no training."""

import argparse
import json
from pathlib import Path
import sys
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sign_language.activity_collection import validate_trial
from sign_language.activity_metrics import motion_summary
from verify_activity_collection import inspect

WRIST = 'wrist_speed_p75_shoulder_widths_per_second'
FINGERS = 'handshape_speed_p75_palm_widths_per_second'


def analyze(directory):
    directory = Path(directory)
    verified = inspect(directory)
    confirmation_path = directory / 'user-confirmation.json'
    confirmation = json.loads(confirmation_path.read_text(encoding='utf-8'))
    if confirmation['manifest_sha256'] != verified['manifest_sha256']:
        raise ValueError('Confirmation does not match the manifest')
    manifest = json.loads((directory / 'session.json').read_text(encoding='utf-8'))
    trials = []
    for item in manifest['trials']:
        validate_trial(directory, item)
        with np.load(directory / item['file'], allow_pickle=False) as source:
            times = source['timestamp_ms']
            # Non-overlapping source ownership: each trial is analyzed separately.
            body = {key: source[key] for key in source.files if key != 'timestamp_ms'}
            windows = []
            end_ms = 2000
            while end_ms <= int(times[-1]):
                indexes = np.flatnonzero((times >= end_ms-2000) & (times <= end_ms))
                if len(indexes) >= 8 and times[indexes[-1]] - times[indexes[0]] >= 1600:
                    raw = {key: value[indexes] for key, value in body.items()}
                    # Face data is unused by motion_summary; do not invent or persist it.
                    snapshot = (times[indexes], raw, None, None, None, tuple(item['frame_shape']))
                    values = motion_summary(snapshot)
                    windows.append({'end_ms': end_ms, 'wrist': values[WRIST], 'handshape': values[FINGERS]})
                end_ms += 500
            valid = [w for w in windows if w['wrist'] is not None and w['handshape'] is not None]
            trials.append({'trial': item['trial'], 'prompt': item['prompt'], 'category': item['category'],
                           'windows': len(windows), 'windows_with_valid_motion': len(valid),
                           'median_wrist_speed': median(w['wrist'] for w in valid) if valid else None,
                           'median_handshape_speed': median(w['handshape'] for w in valid) if valid else None})
    return {'status': manifest['status'], 'confirmed_trials': confirmation['confirmed_trials'],
            'sign_boundaries_annotated': False, 'one_signer_only': True, 'trials': trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory), indent=2, allow_nan=False))


if __name__ == '__main__': main()
