"""Summarize a completed, user-confirmed geometry intervention; no model inference."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sign_language.five_sign_pilot import sha256, write_json

NAMES = {'original': 'Pooled control', 'mean_pool': 'Geometry ON', 'geometry_off': 'Geometry OFF'}


def summarize(session):
    if session['status'] != 'complete' or session.get('camera_open') is not False:
        raise ValueError('Session must be complete with camera released')
    if session.get('mode') != 'geometry_branch_comparison':
        raise ValueError('Wrong diagnostic mode')
    if not session.get('checkpoint_unchanged') or not session.get('candidate_checkpoint_unchanged'):
        raise ValueError('Checkpoint integrity was not confirmed')
    trials = session['trials']
    if [t.get('expected') for t in trials] != ['NO', 'HELLO'] * 3:
        raise ValueError('Six prescribed NO/HELLO trials are required')
    if [t.get('capture_id') for t in trials] != list(range(1, 7)):
        raise ValueError('Capture identities differ')
    for trial in trials:
        if 'error' in trial or not trial.get('identical_observations_for_all_variants'):
            raise ValueError('Failed or unmatched capture')
        for key in NAMES:
            entry = trial['variants'][key]
            if entry['correct'] != (entry['top5'][0]['label'] == trial['expected']):
                raise ValueError('Stored outcome disagrees with expected sign')
    return {key: {
        'correct': sum(t['variants'][key]['correct'] for t in trials), 'trials': len(trials),
        'per_class': {label: {'correct': sum(t['variants'][key]['correct'] for t in trials if t['expected'] == label),
                              'trials': sum(t['expected'] == label for t in trials)} for label in ('NO', 'HELLO')},
    } for key in NAMES}


def plot(report, output):
    trials = report['live']['trials']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle('Confirmed NO / HELLO: same-capture geometry intervention', fontsize=15)
    x = np.arange(2)
    for offset, key, color in [(-.25, 'original', '#64748b'), (0, 'mean_pool', '#2563eb'), (.25, 'geometry_off', '#d97706')]:
        result = report['summary'][key]['per_class']
        bars = axes[0].bar(x + offset, [result[g]['correct'] for g in ('NO', 'HELLO')], .24, color=color, label=NAMES[key])
        for bar, label in zip(bars, ('NO', 'HELLO'), strict=True):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + .04,
                         f'{result[label]["correct"]}/3', ha='center', fontsize=10)
    axes[0].set(xticks=x, xticklabels=['NO', 'HELLO'], yticks=[0, 1, 2, 3], ylim=(0, 3.8),
                ylabel='Correct first guesses', title='Three confirmed attempts of each sign')
    axes[0].legend(loc='upper left', fontsize=8)
    y = np.arange(6)
    margins = [t['hello_minus_no_margin'] for t in trials]
    axes[1].barh(y - .17, [m['geometry_disabled_model'] for m in margins], .32, color='#d97706', label='Geometry OFF')
    axes[1].barh(y + .17, [m['full_model'] for m in margins], .32, color='#2563eb', label='Geometry ON')
    axes[1].axvline(0, color='black', linewidth=1)
    axes[1].set(yticks=y, yticklabels=[f'#{t["capture_id"]} {t["expected"]}' for t in trials],
                xlabel='HELLO minus NO logit margin', title='Right favors HELLO; left favors NO')
    axes[1].invert_yaxis()
    axes[1].legend(loc='lower right', fontsize=8)
    fig.supxlabel('Six diagnostic trials, not general accuracy. Pairwise margin does not describe the other three classes.', fontsize=9)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--confirm-prompts', action='store_true', help='Use only after the user confirms all six prompted captures were normal')
    args = parser.parse_args()
    if not args.confirm_prompts:
        parser.error('User confirmation is required before reporting accuracy')
    session = json.loads(args.report.read_text())
    summary = summarize(session)
    source_sha = sha256(args.report)
    session['user_completion_confirmed'] = True
    no = [t for t in session['trials'] if t['expected'] == 'NO']
    report = {'live': session, 'summary': summary, 'source_sha256': source_sha,
              'confirmation_source': 'User: Completed all six; capture was normal',
              'source_capture_log_modified': False,
              'no_attempts_geometry_reduced_hello_margin': sum(t['hello_minus_no_margin']['actual_geometry_effect'] < 0 for t in no),
              'max_absolute_amp_decomposition_residual': max(abs(t['hello_minus_no_margin']['amp_decomposition_residual']) for t in session['trials']),
              'limits': ['Three attempts per class from one user; diagnostic, not a general performance estimate.',
                         'Geometry OFF retains the candidate recurrent weights and is not a separately trained control.',
                         'Aggregate tracking measures do not establish anatomical joint accuracy.',
                         'Capture-ending visibility differences are observational, not a controlled timing intervention.',
                         'No live media, coordinate arrays or embeddings retained. No training or official test evaluation.']}
    output = ROOT / 'docs/assets/gru-geometry-guided-2026-11-12.json'
    if output.exists():
        raise FileExistsError('Preserve the prior report; use a new version')
    write_json(output, report)
    plot(report, output.with_suffix('.png'))
    if sha256(args.report) != source_sha:
        raise AssertionError('Source capture log changed')
    print(json.dumps({'summary': summary, 'geometry_helped_no_margin': report['no_attempts_geometry_reduced_hello_margin'],
                      'max_amp_residual': report['max_absolute_amp_decomposition_residual']}, indent=2))


if __name__ == '__main__':
    main()
