# ST-GCN27 experiment results

Selected epoch: **39**, minimum official validation loss.
Training finished after 49 epochs; stop reason: early_stopping.
Parameters: **3,788,165**. Official test: **32,941 clips**, 2,731 labels.

| Metric | Regularized GRU validation | ST-GCN validation | Regularized GRU test | ST-GCN test |
| --- | ---: | ---: | ---: | ---: |
| Top-1 | 61.06% | 67.23% | 49.92% | 57.25% |
| Top-5 | 85.90% | 88.92% | 76.24% | 82.32% |
| Macro precision | 64.76% | 70.48% | 52.76% | 59.98% |
| Macro recall | 61.64% | 67.78% | 49.93% | 57.32% |
| Macro F1 | 60.14% | 65.98% | 49.47% | 56.74% |
| Weighted F1 | 60.03% | 65.93% | 49.55% | 56.76% |
| Cross-entropy | 1.65689 | 1.39338 | 2.44304 | 2.08828 |

Summed recorded epoch time (training + validation; excludes checkpoint writes): **103.1 minutes**.
Frozen test evaluation elapsed: **82.31 seconds**.

The graph and input representation differ from the GRU: 27 selected joints, XY only,
spatial/temporal graph convolutions and masked pooling. This is a model-plus-input
comparison, not an isolated change to the sequence layer. It is trained from scratch
with project preprocessing, not the authors' pretrained checkpoint or an exact reproduction.

These are offline dataset results. ST-GCN webcam recognition has not been evaluated.
No claim that it resolves the documented webcam context/tracking gap is made.

Checkpoint SHA-256: `6617c495405197dd4c232e4a27e2a438fd698c8f9ba6b0d1edc1b795b5e86fad`.

See [configuration and controls](stgcn-training.md). Per-class/per-signer outputs,
predictions and the confusion matrix are under `data/evaluations/asl-stgcn27-test-v1/`.
