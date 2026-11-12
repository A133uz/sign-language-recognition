"""Plot frozen paired 15-class results without fitting or scoring models."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'docs/assets/fifteen-sign-temporal-2026-11-12.json'
    report = json.loads(source.read_text())
    if report['status'] != 'complete':
        raise ValueError('Completed trial required')
    fig = plt.figure(figsize=(14, 9), layout='constrained')
    grid = fig.add_gridspec(2, 2)
    accuracy, loss = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    recalls = fig.add_subplot(grid[1, :])
    fig.suptitle('15-class geometry GRU: temporal augmentation comparison', fontsize=16)
    colors = {'control': '#2563eb', 'temporal': '#b45309'}
    names = {'control': 'Spatial control', 'temporal': '+ Temporal augmentation'}
    for key in ('control', 'temporal'):
        run = report[key]
        rows = run['epoch_metrics']
        x = [r['epoch'] for r in rows]
        for split, style in [('train_clean', '--'), ('val', '-')]:
            accuracy.plot(x, [r[split]['top1']*100 for r in rows], style, color=colors[key],
                          label=names[key] + (' train' if split == 'train_clean' else ' val'))
        accuracy.axvline(run['selection']['selected_epoch'], color=colors[key], alpha=.3)
        loss.plot(x, [r['val']['loss'] for r in rows], color=colors[key], label=names[key])
    accuracy.set(title='Clean train / validation; vertical lines: selected epochs', xlabel='Epoch', ylabel='Top-1 (%)', ylim=(0, 105))
    accuracy.legend(fontsize=8)
    loss.set(title='Validation cross-entropy', xlabel='Epoch', ylabel='Cross-entropy')
    loss.legend(fontsize=8)
    labels = report['specification']['classes']
    x = np.arange(len(labels))
    for offset, key in [(-.19, 'control'), (.19, 'temporal')]:
        metrics = report[key]['validation']
        bars = recalls.bar(x + offset, [metrics['per_class'][g]['recall']*100 for g in labels], .36,
                           label=f'{names[key]} ({metrics["correct"]}/{metrics["samples"]})', color=colors[key])
        for bar, label in zip(bars, labels, strict=True):
            item = metrics['per_class'][label]
            recalls.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                         f'{item["correct"]}/{item["support"]}', ha='center', fontsize=7)
    recalls.set(xticks=x, xticklabels=labels, ylim=(0, 120), ylabel='Validation recall (%)',
                title='Same held-out signers and rows; each class has only 3-5 validation clips')
    recalls.tick_params(axis='x', labelsize=9, rotation=35)
    recalls.legend(loc='upper left', fontsize=8, ncols=2)
    fig.supxlabel('One seed, fixed settings, official test untouched. Five-class historical accuracy is a different task.', fontsize=10)
    fig.savefig(source.with_suffix('.png'), dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    for ax, key in zip(axes, ('control', 'temporal'), strict=True):
        matrix = np.asarray(report[key]['validation']['confusion'])
        ax.imshow(matrix / matrix.sum(1, keepdims=True), vmin=0, vmax=1, cmap='Blues')
        for (i, j), value in np.ndenumerate(matrix):
            if value:
                ax.text(j, i, str(value), ha='center', va='center', fontsize=8,
                        color='white' if value / matrix[i].sum() > .6 else 'black')
        ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
               title=names[key], xlabel='Predicted', ylabel='True label')
        ax.tick_params(labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=65, ha='right')
    fig.suptitle('15-class validation confusion: counts; empty cells are zero', fontsize=15)
    target = source.with_name(source.stem + '-confusion.png')
    fig.savefig(target, dpi=170)
    plt.close(fig)
    print(source.with_suffix('.png'))
    print(target)


if __name__ == '__main__':
    main()
