"""Summarize motion proxies by trial without persisting landmark coordinates."""

import argparse
import json
from pathlib import Path
from statistics import median

KEYS = ('wrist_speed_p75_shoulder_widths_per_second',
        'handshape_speed_p75_palm_widths_per_second')


def summarize(report):
    if not report.get('activity_measurements'):
        raise ValueError('This is not an activity-measurement report')
    trials = []
    for trial in report['trials']:
        windows = trial['windows']
        item = {'number': trial['number'], 'prompt': trial['prompt'],
                'completed': trial['completed'], 'scored_windows': len(windows),
                'duration_seconds': trial['duration_seconds'], 'phases': {}}
        for name, selected in [('all', windows),
                               ('early_proxy', [w for w in windows if w['end_seconds'] <= 3]),
                               ('late_proxy', [w for w in windows if w['end_seconds'] >= 5.5])]:
            measures = {}
            for key in KEYS:
                values = [w['motion'][key] for w in selected if w['motion'][key] is not None]
                measures[key] = median(values) if values else None
            item['phases'][name] = {'windows': len(selected), 'median': measures}
        trials.append(item)
    return {'status': report['status'], 'confirmation': report['confirmation'], 'trials': trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(json.loads(args.report.read_text(encoding='utf-8'))), indent=2))


if __name__ == '__main__': main()
