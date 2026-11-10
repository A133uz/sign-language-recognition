# Frozen regularized GRU: official test evaluation

Completed successfully on **2026-11-10 at approximately 07:55 Asia/Tashkent**,
with exit code 0. This is the first held-out test evaluation of the selected
regularized model, not another training run or checkpoint-selection experiment.

## Frozen model and data

- Checkpoint: `models/asl-gru-128-cv-regularized-v1/official-refit/best.pt`.
- Selected epoch: **37**, chosen by minimum official validation loss before
  accessing test results. Two-layer, 256-unit GRU, 2,731 output labels.
- Checkpoint SHA-256 before and after evaluation:
  `6e09a1b3c62ea364e2a86b5551eaefe71651332f75fb64a060ed184de7032b39`.
- All **32,941 official test clips**, **11 test signers**, **2,731 labels**.
  No examples were excluded. Official metadata validation checks signer
  separation and cached row/label order.
- Same prepared cache/preprocessing as training; every test slice passed its
  feature/length/label checksum before evaluation.
- `model.eval()` and `torch.inference_mode()`; no optimizer, gradients,
  dropout, augmentation, fine-tuning or parameter updates.
- RTX 4060, PyTorch 2.11.0+cu128, CUDA AMP and the recorded non-cuDNN CUDA
  backend, batch size 64, two CPU loading workers, no shuffle.

## Results

| Metric | Official validation at selection | Official test |
| --- | ---: | ---: |
| Top-1 accuracy | 61.06% | **49.92%** |
| Top-5 accuracy | 85.90% | **76.24%** |
| Macro precision | 64.76% | **52.76%** |
| Macro recall | 61.64% | **49.93%** |
| Macro F1 | 60.14% | **49.47%** |
| Weighted F1 | 60.03% | **49.55%** |
| Cross-entropy loss | 1.65689 | **2.44304** |

Top-1: **16,445 / 32,941** correct. Top-5: **25,115 / 32,941** correct.
Macro metrics average over all 2,731 classes; undefined class precision/recall
uses zero. The confusion matrix has true labels on rows and predictions on
columns. Softmax scores are saved as model scores, not calibrated confidence.

The test top-1 score is **11.14 percentage points below validation**, indicating
weaker generalization to these additional unseen signers. Per-signer top-1
ranges from approximately 30.40% to 69.39%. This does not establish live-webcam,
unknown-sign rejection, continuous-signing or RSL accuracy. Use these as the
frozen model's held-out results; further model selection should use training
and validation data rather than optimizing against this test score.

## Verification and artifacts

Total elapsed time: **64.92 seconds**, including **39.88 seconds** of checkpoint,
metadata and cache verification. The batched inference/prediction-export loop
took **24.47 seconds**. This is not end-to-end webcam latency: video decoding
and MediaPipe extraction were already completed offline.

Independent export validation confirmed all 32,941 ordered records match the
official test index; top-1/top-5 counts match the report; confusion totals and
an independently calculated macro F1 agree. The checkpoint hash is unchanged.
All **43 project checks** passed, including inference-only behavior, checkpoint
preservation, prediction ordering and metric calculations.

Outputs under `data/evaluations/asl-gru-regularized-test-v1/`:

- `report.json`, `status.json`: aggregate metrics and evaluation provenance.
- `predictions.jsonl`: per-clip target, prediction, top-five labels/scores and
  participant/video IDs in official cached order.
- `confusion.npz`: counts and label names, loadable without pickle.
- `per-class.csv`: support, precision, recall and F1 for every label.
- `per-signer.json`: counts and top-1/top-5 per test participant.

Entry point: `scripts/evaluate_gru.py`. It refuses a nonempty output directory.
Any explicitly requested repeat must choose a new `--output`; it does not
overwrite the original report. Runtime/model/data files remain ignored by Git.
Original training/CV status flags describe those training jobs and remain
unchanged; this separate report records the subsequent test evaluation.
