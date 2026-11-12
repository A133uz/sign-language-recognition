"""Report a confirmed ten-trial all-five-sign session without model inference."""
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

LABELS = ('HELLO', 'NO', 'PLEASE', 'THANKYOU', 'YES')
NAMES = {'original': 'Pooled GRU control', 'mean_pool': 'GRU + geometry',
         'geometry_off': 'Same candidate, geometry OFF', 'geometry_only': 'Geometry only'}


def summarize(session):
    if session['status'] != 'complete' or session.get('camera_open') is not False:
        raise ValueError('Session incomplete or camera still open')
    if session.get('mode') != 'five_class_geometry_only_comparison':
        raise ValueError('Unexpected session mode')
    if not all(session.get(k) for k in ('checkpoint_unchanged', 'candidate_checkpoint_unchanged', 'geometry_probe_unchanged')):
        raise ValueError('Artifact integrity not confirmed')
    trials = session['trials']
    if [t.get('expected') for t in trials] != ['HELLO', 'THANKYOU', 'PLEASE', 'YES', 'NO'] * 2:
        raise ValueError('Expected all ten prescribed prompts')
    if [t.get('capture_id') for t in trials] != list(range(1, 11)):
        raise ValueError('Capture IDs differ')
    results = {}
    for key in NAMES:
        matrix = np.zeros((5, 5), dtype=np.int64)
        for trial in trials:
            if 'error' in trial or not trial.get('identical_observations_for_all_variants'):
                raise ValueError('Failed or unmatched trial')
            entry = trial['variants'][key]
            predicted = entry['top5'][0]['label']
            if entry['correct'] != (predicted == trial['expected']):
                raise ValueError('Stored correctness disagrees with label')
            matrix[LABELS.index(trial['expected']), LABELS.index(predicted)] += 1
        tp = matrix.diagonal().astype(float)
        precision = np.divide(tp, matrix.sum(0), out=np.zeros(5), where=matrix.sum(0) > 0)
        recall = tp / matrix.sum(1)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(5), where=(precision + recall) > 0)
        results[key] = {'correct': int(tp.sum()), 'trials': 10, 'top1': float(tp.sum()/10),
                        'macro_f1': float(f1.mean()), 'labels': list(LABELS), 'confusion': matrix.tolist(),
                        'per_class': {name: {'correct': int(tp[i]), 'trials': int(matrix[i].sum()),
                                             'recall': float(recall[i]), 'f1': float(f1[i])} for i, name in enumerate(LABELS)}}
    return results


def plot(report, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    fig.suptitle('Confirmed fresh field test: two attempts per sign', fontsize=15)
    x = np.arange(5)
    for offset, key, color in [(-.22, 'mean_pool', '#2563eb'), (.22, 'geometry_only', '#0f766e')]:
        result = report['summary'][key]
        bars = axes[0].bar(x + offset, [result['per_class'][g]['correct'] for g in LABELS], .4,
                           label=f'{NAMES[key]} ({result["correct"]}/10)', color=color)
        for bar, label in zip(bars, LABELS, strict=True):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + .03,
                         f'{result["per_class"][label]["correct"]}/2', ha='center', fontsize=9)
    axes[0].set(xticks=x, xticklabels=LABELS, ylim=(0, 2.65), yticks=[0, 1, 2],
                ylabel='Correct first guesses', title='Identical captures, frozen classifiers')
    axes[0].tick_params(axis='x', labelsize=8)
    axes[0].legend(loc='upper left', fontsize=8)
    matrix = np.asarray(report['summary']['geometry_only']['confusion'])
    axes[1].imshow(matrix / 2, vmin=0, vmax=1, cmap='Blues')
    for (i, j), value in np.ndenumerate(matrix):
        axes[1].text(j, i, str(value), ha='center', va='center', color='white' if value == 2 else 'black')
    axes[1].set(xticks=x, yticks=x, xticklabels=LABELS, yticklabels=LABELS,
                xlabel='Predicted', ylabel='True label', title='Geometry only: one THANKYOU → HELLO error')
    axes[1].tick_params(labelsize=8)
    plt.setp(axes[1].get_xticklabels(), rotation=25, ha='right')
    fig.supxlabel('Ten prompted trials from one user. Earlier NO failures remain unresolved evidence; no general accuracy claim.', fontsize=9)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--confirm-prompts', action='store_true')
    args = parser.parse_args()
    if not args.confirm_prompts:
        parser.error('Use only after the user confirms normal completion of all ten prompts')
    session = json.loads(args.report.read_text())
    summary = summarize(session)
    files = {
        'models/asl-gru-five-signs-geometry-v1/control/best.pt': session['checkpoint_sha256'],
        'models/asl-gru-five-signs-geometry-v1/geometry/best.pt': session['candidate']['checkpoint_sha256'],
        'models/asl-five-class-geometry-only-v1/model.pkl': session['geometry_probe']['sha256'],
    }
    if any(sha256(ROOT / name) != digest for name, digest in files.items()):
        raise ValueError('Selected model changed after the session')
    source_sha = sha256(args.report)
    session['user_completion_confirmed'] = True
    report = {'live': session, 'summary': summary, 'source_sha256': source_sha,
              'confirmation_source': 'User: Completed all ten; capture was normal',
              'source_capture_log_modified': False, 'current_model_files_match': True,
              'limits': ['Two attempts per class, one signer, one session; not general accuracy.',
                         'Fresh captures after freezing models, from the same user involved in earlier development.',
                         'Do not discard previous failed NO captures or infer a fix from this session.',
                         'Tracking counts do not measure anatomical landmark accuracy. Softmax scores are uncalibrated.',
                         'No model fitting, threshold adjustment, live media/coordinates/embeddings or official test evaluation.']}
    output = ROOT / 'docs/assets/five-class-geometry-field-test-2026-11-12.json'
    if output.exists():
        raise FileExistsError('Preserve the prior report; use a new version')
    write_json(output, report)
    plot(report, output.with_suffix('.png'))
    if sha256(args.report) != source_sha:
        raise AssertionError('Source log changed')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
