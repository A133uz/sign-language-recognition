"""Plot saved paired training/validation metrics; never train or score a model."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'docs/assets/gru-geometry-trial-2026-11-12.json'
    report = json.loads(source.read_text())
    if report['status'] != 'complete':
        raise ValueError('Only completed trials may be plotted')
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle('Five-class mean-pool GRU: explicit hand geometry comparison', fontsize=15)
    colors = {'control': '#2563eb', 'geometry': '#b45309'}
    names = {'control': 'Control', 'geometry': '+17 geometry features'}
    for key in ('control', 'geometry'):
        rows = report[key]['epoch_metrics']
        epochs = [r['epoch'] for r in rows]
        color = colors[key]
        axes[0, 0].plot(epochs, [r['train_clean']['top1'] * 100 for r in rows], '--',
                        color=color, label=names[key] + ' train')
        axes[0, 0].plot(epochs, [r['val']['top1'] * 100 for r in rows], '-',
                        color=color, label=names[key] + ' val')
        axes[0, 0].axvline(report[key]['selection']['selected_epoch'], color=color, alpha=.3)
        axes[0, 1].plot(epochs, [r['val']['loss'] for r in rows], color=color, label=names[key])
    axes[0, 0].set(title='Clean train / validation; vertical lines: selected epochs',
                   xlabel='Epoch', ylabel='Top-1 correct (%)', ylim=(0, 105))
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title='Validation cross-entropy (19 clips)', xlabel='Epoch', ylabel='Cross-entropy')
    axes[0, 1].legend(fontsize=8)
    labels = report['control']['validation']['labels']
    positions = np.arange(len(labels))
    for offset, key in [(-.19, 'control'), (.19, 'geometry')]:
        result = report[key]['validation']
        bars = axes[1, 0].bar(positions + offset,
                             [result['per_class'][g]['recall'] * 100 for g in labels],
                             .36, color=colors[key], label=names[key])
        for bar, label in zip(bars, labels, strict=True):
            item = result['per_class'][label]
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f'{item["correct"]}/{item["support"]}', ha='center', fontsize=8)
    axes[1, 0].set(xticks=positions, xticklabels=labels, ylim=(0, 118), ylabel='Recall (%)',
                   title='Selected checkpoints: validation by class')
    axes[1, 0].tick_params(axis='x', labelsize=8)
    axes[1, 0].legend(fontsize=8, loc='lower left')
    matrix = np.asarray(report['geometry']['validation']['confusion'])
    rates = matrix / matrix.sum(1, keepdims=True)
    axes[1, 1].imshow(rates, vmin=0, vmax=1, cmap='Blues')
    for (i, j), value in np.ndenumerate(matrix):
        axes[1, 1].text(j, i, str(value), ha='center', va='center',
                        color='white' if rates[i, j] > .6 else 'black')
    axes[1, 1].set(xticks=positions, yticks=positions, xticklabels=labels, yticklabels=labels,
                   xlabel='Predicted', ylabel='True label', title='Geometry candidate: validation counts')
    axes[1, 1].tick_params(axis='both', labelsize=8)
    for label in axes[1, 1].get_xticklabels():
        label.set_rotation(30)
        label.set_ha('right')
    fig.supxlabel('One seed, development validation only. Official test and new webcam inputs unused.', fontsize=10)
    target = source.with_suffix('.png')
    fig.savefig(target, dpi=170)
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    main()
