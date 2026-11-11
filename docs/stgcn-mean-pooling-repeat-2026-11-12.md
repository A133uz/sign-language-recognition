# Five-sign ST-GCN mean-pooling training — 2026-11-12

The user requested adding mean pooling to ST-GCN and training the same five
classes. Source inspection found that **ST-GCN already uses masked mean pooling**:

```python
pooled = x.sum((2, 3)) / (lengths[:, None].clamp_min(1) * v)
```

This averages the final learned features over valid reduced time steps and all
27 graph nodes before dropout and the linear classifier. Padding is masked after
each block; real no-hand/rest frames remain included. Node absence is not a
separate pooling mask. The old five-class ST-GCN was trained with this readout.
No extra pooling layer or new architecture is introduced.

After explaining this finding, prepared a fresh separate run as requested. It
is a **repeat of the existing mean-pooled architecture**, not evidence for a
new pooling improvement. `stgcn_model.py`, `stgcn_graph.py` and normalization
remain unchanged so existing checkpoints retain source compatibility.

## Fixed training protocol

- Same HELLO/NO/PLEASE/THANKYOU/YES cache: 75 train / 19 validation / 60 test.
- Same 27 XY nodes, ten graph/temporal blocks, widths 64/128/256, dropout .5.
- Seed 42, AdamW .001, weight decay .001, batch 4, two loader workers, existing
  roll/scale augmentation, RTX 4060 CUDA AMP, cuDNN convolutions enabled.
- Same 33-epoch cap, warm-up, early-stopping and validation-only selection policy.
- One test evaluation after selection; previously reported test set, not a new
  unseen benchmark. No checkpoint is selected using test/live outcomes.
- Preserve every existing `best.pt` and source-cache payload by hash; refuse
  an existing output directory. Keep the default webcam checkpoint unchanged.
- Separate output: `models/asl-stgcn27-five-signs-mean-pool-repeat-v1/`.

Command:

```powershell
.\.venv\Scripts\python.exe -u scripts/run_stgcn_mean_pool_trial.py
```

A hooked model test checks that the actual classifier input equals the
arithmetic mean of the last block's features over each sample's true reduced
length and every graph node. All **106 project tests passed** before training,
along with compilation, lockfile and dependency checks.

## Completed results

Training ran on the local RTX 4060 for **7 epochs / 11.73 seconds**, stopping
after three validation-loss epochs without improvement. The unchanged policy
selected **epoch 4**. All epoch files and source snapshots are retained.

| Metric | Previous five-sign ST-GCN | Fresh mean-pool repeat |
|---|---:|---:|
| Selected epoch | 4 | 4 |
| Parameters | 3,087,583 | 3,087,583 |
| Clean training top-1 | 53/75 (70.67%) | 53/75 (70.67%) |
| Validation top-1 | 13/19 (68.42%) | 13/19 (68.42%) |
| Test top-1 | 39/60 (65.0%) | 39/60 (65.0%) |
| Test macro F1 | 59.04% | 59.04% |
| Test cross-entropy | 0.8475 | 0.8475 |

All selected model tensors, including learned parameters and BatchNorm buffers,
are **bit-for-bit identical** to the previous run: maximum tensor difference
0.0. All 60 top-1 test predictions also agree, with zero improved/regressed clips.
Checkpoint file bytes can differ because their configuration/source metadata and
serialization paths differ; this does not imply changed model tensor values.

| Sign | Correct / test clips |
|---|---:|
| HELLO | 14/14 |
| NO | 1/11 |
| PLEASE | 9/13 |
| THANKYOU | 9/12 |
| YES | 6/10 |

NO remains particularly weak. This is a reproducibility result on the same tiny
five-class task, not evidence for an architecture improvement, new independent
test sample, or reliable live performance. There was no missing mean-pooling
layer to add to this ST-GCN. No pooling variant was silently substituted.

## Verification and preserved artifacts

- All 19 validation examples match raw-observation runtime preprocessing and
  cached inference exactly: identical rankings, max score difference **0.0**.
- Every earlier `best.pt` and all subset payload hashes are unchanged. The
  corpus audit remains stopped and the existing default webcam model is retained.
- No webcam was opened, no landmarks were re-extracted, and no source model,
  normalization, graph, augmentation or class split was changed.
- The first epoch had three recoverable AMP overflow skips; the existing
  GradScaler lowered its scale and skipped those updates. Final tensor equality
  and runtime checks passed; no optimizer or training-policy change was made.
- New weights, all seven epoch records, config and source snapshots:
  `models/asl-stgcn27-five-signs-mean-pool-repeat-v1/`.
- [Complete repeat comparison evidence](assets/stgcn-mean-pool-repeat-2026-11-12.json)
- [Learning curves and test confusion matrix](assets/stgcn-mean-pool-repeat-2026-11-12.png)

Explicit webcam command for these separately saved weights, if requested:

```powershell
.\.venv\Scripts\python.exe scripts/recognize_webcam.py --checkpoint models/asl-stgcn27-five-signs-mean-pool-repeat-v1/best.pt
```

Its model header is ST-GCN / epoch 4 / 5 labels. Rebuild this report without
training or inference with `scripts/report_stgcn_mean_pool_trial.py`.

## Interactive webcam launch (2026-11-12)

At the user's request, opened the separately saved repeat checkpoint with
`scripts/recognize_webcam.py --checkpoint models/asl-stgcn27-five-signs-mean-pool-repeat-v1/best.pt --report .cache/webcam-stgcn-five-signs-20261112-01.json`.
Startup confirms **SignSTGCN / epoch 4 / CUDA / five labels**, and the ready
window identifies **ST-GCN**. This session uses manual Space start/finish
controls and saves only predictions and aggregate timing/tracking metadata.
No new training or default-model change occurred.

### Completed interactive feedback

The user reported matching first guesses for HELLO, THANKYOU and PLEASE, and
HELLO as the first guess for both YES and NO. The closed session contains ten
captures: eight HELLO first guesses, one PLEASE and one THANKYOU. Actual loaded
class is SignSTGCN, epoch 4, with the repeat checkpoint SHA matching its saved
test report. The camera closed normally.

Capture numbers/intended labels were not supplied for this interactive session,
so the qualitative feedback cannot be assigned to individual entries or turned
into a formal live accuracy. The predominance of HELLO in the log is consistent
with the feedback, but not every HELLO prediction can be called an error.

This is not a successful general live fix. The fixed test already shows NO
correct in only 1/11 examples and YES in 6/10. Thus these classes have measured
weakness beyond a single webcam session. Similar symptoms in the pooled GRU
and ST-GCN do not prove identical causes: the GRU uses the 254-feature body/local
xyz representation, whereas ST-GCN selects 27 body-relative xy graph nodes.
Their shared data, context sensitivity and imperfect tracking remain relevant.
The earlier scalar diagnostic found measurable NO/YES articulation differences;
overlapping preview dots alone are not a confirmed cause of these errors.

No changes to weights, input processing, class labels, defaults or source data
were made in response to this feedback.
[Session snapshot and feedback](assets/stgcn-five-sign-live-feedback-2026-11-12.json).
