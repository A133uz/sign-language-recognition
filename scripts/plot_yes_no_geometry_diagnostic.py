"""Render the frozen step-one report without fitting or scoring any model."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / 'docs/assets/yes-no-geometry-diagnostic-2026-11-12.json'
    report = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.2))
    fig.subplots_adjust(left=.06, right=.99, bottom=.20, top=.78, wspace=.30)
    keys = ('train_fit', 'train_grouped_oof', 'validation')
    titles = ('Training fit (19 signers)', 'Grouped out-of-fold (3 folds)',
              'Validation (3 other signers)')
    for ax, key, title in zip(axes, keys, titles, strict=True):
        metrics = report[key]
        matrix = np.asarray(metrics['confusion'])
        rates = matrix / matrix.sum(1, keepdims=True)
        ax.imshow(rates, vmin=0, vmax=1, cmap='Blues')
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha='center', va='center', fontsize=17,
                    color='white' if rates[i, j] > .6 else 'black')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['NO', 'YES'],
               yticklabels=['NO', 'YES'], xlabel='Predicted label', ylabel='True label',
               title=f"{title}\n{metrics['correct']}/{metrics['samples']} correct")
    fig.suptitle('Step 1: fixed geometry-only binary diagnostic', fontsize=16, y=.97)
    fig.text(.5, .035, 'Counts shown; color normalized within each true class. Small sample; test split untouched.',
             ha='center', fontsize=10)
    target = path.with_suffix('.png')
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    main()
