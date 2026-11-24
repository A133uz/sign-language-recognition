"""Summarize coverage and scoring failures without exposing landmark data."""

import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(report):
    if not report.get('health_measurements'):
        raise ValueError('This is not a window-health diagnostic')
    trials = []
    for trial in report['trials']:
        failures = trial.get('unscored_details', [])
        counts = Counter(item['reason'] for item in failures)
        if sum(counts.values()) != trial.get('unscored_windows', 0):
            raise ValueError('Failure details disagree with total')
        scored = trial['windows']
        coverage = [w['health'] for w in scored] + [w['health'] for w in failures]
        trials.append({'number': trial['number'], 'prompt': trial['prompt'],
                       'completed': trial['completed'], 'scored': len(scored),
                       'unscored': len(failures), 'failure_reasons': dict(counts),
                       'aggregate': {'observed_frames_in_windows': sum(c['observed_frames'] for c in coverage),
                                     'raw_hand_frames_in_windows': sum(c['raw_hand_frames'] for c in coverage),
                                     'raw_shoulder_frames_in_windows': sum(c['raw_shoulder_frames'] for c in coverage),
                                     'raw_face_frames_in_windows': sum(c['raw_face_frames'] for c in coverage)}})
    return {'status': report['status'], 'confirmation': report['confirmation'], 'trials': trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(json.loads(args.report.read_text(encoding='utf-8'))), indent=2))


if __name__ == '__main__': main()
