"""Summarize matched windows without treating unconfirmed prompts as truth."""

import argparse
import json
from pathlib import Path

MODES = ('raw', 'threshold', 'threshold_and_stability')
SIGNS = {'HELLO', 'THANKYOU', 'PLEASE', 'YES', 'NO'}


def summarize(report):
    trials = []
    for trial in report['trials']:
        windows = trial['windows']
        results = {}
        for mode in MODES:
            labels = [window['decisions'][mode] for window in windows]
            episodes = sum(label is not None and (i == 0 or labels[i-1] != label)
                           for i, label in enumerate(labels))
            item = {'accepted_windows': sum(label is not None for label in labels),
                    'display_runs': episodes}
            if trial['prompt'] in SIGNS:
                item['windows_matching_prompt'] = sum(label == trial['prompt'] for label in labels)
                item['matched_prompt_at_least_once'] = any(label == trial['prompt'] for label in labels)
            results[mode] = item
        trials.append({'number': trial['number'], 'prompt': trial['prompt'],
                       'completed': trial['completed'], 'scored_windows': len(windows),
                       'unscored_windows': trial.get('unscored_windows', 0),
                       'duration_seconds': trial['duration_seconds'], 'modes': results})
    return {'status': report['status'], 'confirmation': report['confirmation'],
            'threshold_experimental': report['threshold_experimental'], 'trials': trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(json.loads(args.report.read_text(encoding='utf-8'))), indent=2))


if __name__ == '__main__': main()
