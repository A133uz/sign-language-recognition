"""Write a factual comparison only after ST-GCN training/test evaluation completes."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'models/asl-stgcn27-v1'
    report = json.loads((ROOT/'data/evaluations/asl-stgcn27-test-v1/report.json').read_text())
    baseline = json.loads((ROOT/'data/evaluations/asl-gru-regularized-test-v1/report.json').read_text())
    status = json.loads((output/'status.json').read_text())
    if report['status'] != 'complete' or status['status'] != 'completed':
        raise ValueError('Training and frozen evaluation must complete first')
    if hashlib.sha256((output/'best.pt').read_bytes()).hexdigest() != report['checkpoint_sha256']:
        raise ValueError('Evaluated checkpoint differs')
    lines = ['# ST-GCN27 experiment results', '',
             f'Selected epoch: **{report["checkpoint_epoch"]}**, minimum official validation loss.',
             f'Training finished after {status["epoch"]} epochs; stop reason: {status["stop_reason"]}.',
             f'Parameters: **{status["parameters"]:,}**. Official test: **{report["metrics"]["samples"]:,} clips**, 2,731 labels.', '',
             '| Metric | Regularized GRU validation | ST-GCN validation | Regularized GRU test | ST-GCN test |',
             '| --- | ---: | ---: | ---: | ---: |']
    for key, name in (('top1','Top-1'),('top5','Top-5'),('macro_precision','Macro precision'),
                      ('macro_recall','Macro recall'),('macro_f1','Macro F1'),('weighted_f1','Weighted F1'),('loss','Cross-entropy')):
        values = [baseline['validation_at_selection'][key], report['validation_at_selection'][key],
                  baseline['metrics'][key], report['metrics'][key]]
        cells = [f'{value:.5f}' if key == 'loss' else f'{value*100:.2f}%' for value in values]
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    epochs = [json.loads(path.read_text()) for path in output.glob('epoch-*.json')]
    lines += ['', f'Summed recorded epoch time (training + validation; excludes checkpoint writes): **{sum(x["seconds"] for x in epochs)/60:.1f} minutes**.',
              f'Frozen test evaluation elapsed: **{report["elapsed_seconds"]:.2f} seconds**.', '',
              'The graph and input representation differ from the GRU: 27 selected joints, XY only,',
              'spatial/temporal graph convolutions and masked pooling. This is a model-plus-input',
              'comparison, not an isolated change to the sequence layer. It is trained from scratch',
              'with project preprocessing, not the authors\' pretrained checkpoint or an exact reproduction.', '',
              'These are offline dataset results. ST-GCN webcam recognition has not been evaluated.',
              'No claim that it resolves the documented webcam context/tracking gap is made.', '',
              f'Checkpoint SHA-256: `{report["checkpoint_sha256"]}`.', '',
              'See [configuration and controls](stgcn-training.md). Per-class/per-signer outputs,',
              'predictions and the confusion matrix are under `data/evaluations/asl-stgcn27-test-v1/`.']
    text = '\n'.join(lines)+'\n'
    destination = ROOT/'docs/stgcn27-results.md'
    if destination.exists() and destination.read_text(encoding='utf-8') != text:
        raise ValueError('Results document already exists with different contents')
    destination.write_text(text, encoding='utf-8')
    print(destination, flush=True)
    # This script is a future fresh subprocess of the already-running pipeline.
    # Adding the hook here requires no trainer restart or frozen-engine edits.
    subprocess.run([sys.executable, '-u', str(ROOT/'scripts/build_project_notebooks.py'),
                    '--require-complete'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
