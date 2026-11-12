"""Plot the frozen five-class geometry report; no fitting or rescoring."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'docs/assets/five-class-geometry-only-2026-11-12.json'
    report = json.loads(source.read_text())
    if report['status'] != 'complete':
        raise ValueError('Completed report required')
    labels = report['validation']['labels']
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle('Five-class geometry-only classifier: fixed diagnostic', fontsize=15)
    for ax, key, title in [(axes[0, 0], 'train_grouped_oof', 'Training-only signer-grouped OOF'),
                            (axes[0, 1], 'validation', 'Official validation')]:
        metrics = report[key]
        matrix = np.asarray(metrics['confusion'])
        rates = matrix / matrix.sum(1, keepdims=True)
        ax.imshow(rates, vmin=0, vmax=1, cmap='Blues')
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha='center', va='center', color='white' if rates[i, j] > .6 else 'black')
        ax.set(xticks=range(5), yticks=range(5), xticklabels=labels, yticklabels=labels,
               title=f'{title}: {metrics["correct"]}/{metrics["samples"]}', xlabel='Predicted', ylabel='True label')
        ax.tick_params(labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
    x = np.arange(5)
    references = report['same_validation_sequence_model_reference']
    for offset, result, label, color in [(-.25, references['control'], 'Pooled GRU', '#64748b'),
                                        (0, references['geometry'], 'GRU + geometry', '#d97706'),
                                        (.25, report['validation'], 'Geometry only', '#2563eb')]:
        items = [result['per_class'][g] for g in labels]
        bars = axes[1, 0].bar(x + offset, [v['recall'] * 100 for v in items], .24, color=color, label=label)
        for bar, item in zip(bars, items, strict=True):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{item["correct"]}/{item["support"]}', ha='center', fontsize=7)
    axes[1, 0].set(xticks=x, xticklabels=labels, ylim=(0, 113), ylabel='Recall (%)', title='Same validation clips; selected/fixed models')
    axes[1, 0].tick_params(axis='x', labelsize=8)
    axes[1, 0].legend(loc='lower left', fontsize=8)
    predictions = report['retrospective_live']['predictions']
    expected_scores = [p['scores'][labels.index(p['expected'])] * 100 for p in predictions]
    bars = axes[1, 1].bar(range(6), expected_scores, color=['#2563eb' if p['expected'] == 'NO' else '#0f766e' for p in predictions])
    for bar, p in zip(bars, predictions, strict=True):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        p['predicted'], ha='center', fontsize=8)
    axes[1, 1].set(xticks=range(6), xticklabels=[f'#{p["capture_id"]}\n{p["expected"]}' for p in predictions],
                   ylim=(0, 116), ylabel='Expected-sign score (%, uncalibrated)',
                   title='Six saved captures: 6/6, retrospective only')
    fig.supxlabel('One fixed fit, no tuning. Small development sets; no official test or new webcam capture.', fontsize=10)
    output = source.with_suffix('.png')
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
