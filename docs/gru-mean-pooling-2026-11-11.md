# Five-sign GRU: masked mean pooling comparison

The user authorized the proposed masked mean pooling experiment on 2026-11-11.
The first comparison changes only the temporal readout: the same two-layer,
256-unit GRU exposes its top-layer outputs at every real time step, averages
them over the true length, then applies the existing dropout and linear head.
Padding is excluded; real no-hand/rest frames remain included. No attention,
boundary detector, temporal augmentation, depth suppression or new landmarks
are introduced in this comparison.

The old `gru_model.py` is unchanged. The new model has the same parameter names,
shapes, parameter count and seeded initialization; its definition is in a
separate `gru_mean_pool_model.py`. Runtime selects it using the checkpoint's
`gru-mean-pool-v1` architecture and checks both model-source hashes. The webcam
header identifies the actual model as **GRU mean-pool**.

## Fixed protocol

- Source cache: `data/training/asl-five-signs-128-v1`, unchanged 75/19/60
  official signer-disjoint examples, five classes, up to 128 frames, 254 features.
- Train from scratch, seed 42, hidden size 256, two layers, dropout .5,
  AdamW learning rate .001 / weight decay .001, batch 4, two loader workers,
  existing roll/scale augmentation, CUDA AMP, cuDNN disabled as for the baseline.
- Same 28-epoch upper limit, early-stopping rules and validation-only checkpoint
  selection as the regularized last-state GRU. Preserve every epoch separately.
- One official test evaluation after selection; this is the same previously
  reported test set, not a fresh unseen benchmark, and it cannot select the epoch.
- Separate output: `models/asl-gru-five-signs-mean-pool-v1/`. Refuse existing
  output directories. Preserve every previous `best.pt` and cache payload by hash.
- Default webcam checkpoint stays the original epoch-18 GRU until a deliberate
  change is justified. A pooling improvement is not assumed in advance.

Command:

```powershell
.\.venv\Scripts\python.exe -u scripts/run_mean_pool_trial.py
```

## Verification and results

Completed on 2026-11-12, including the user-confirmed paired webcam check.
**Masked mean pooling did not establish an overall webcam improvement.** It
improves HELLO recognition in this session, but misclassifies both NO attempts
as HELLO. Preserve it as an experimental alternative; retain the original
last-state GRU as the default. No further model variants were trained here.

### Training and fixed test comparison

Training completed ten epochs in 33.49 seconds on the local RTX 4060; the
unchanged sustained-generalization-gap rule stopped the run. The unchanged
validation-only selection rule chose epoch **4**. Epoch 9 has minimum validation
loss, but was not substituted after seeing test/live outcomes. All ten epoch
files, selected weights and source snapshots are preserved.

| Metric | Original last-state GRU | Masked mean GRU |
|---|---:|---:|
| Selected epoch | 18 | 4 |
| Parameters | 789,253 | 789,253 |
| Clean training top-1 | 66/75 (88.0%) | 55/75 (73.3%) |
| Validation top-1 | 15/19 (78.95%) | 15/19 (78.95%) |
| Test top-1 | 42/60 (70.0%) | 47/60 (78.33%) |
| Test macro F1 | 67.24% | 74.69% |
| Test cross-entropy | 0.7457 | 0.6377 |

Paired test predictions improve ten previously incorrect clips and regress five
previously correct clips, for a net gain of five. These are one-seed results on
the same small, previously reported test set, not proof of general superiority.

| Test sign | Original correct | Mean-pool correct |
|---|---:|---:|
| HELLO | 12/14 | 13/14 |
| NO | 3/11 | 3/11 |
| PLEASE | 7/13 | 13/13 |
| THANKYOU | 12/12 | 12/12 |
| YES | 8/10 | 6/10 |

Most of the aggregate test gain comes from PLEASE. NO remains weak and YES
regresses. A smaller training/validation gap alone is not the success criterion.

### Confirmed paired webcam comparison

Session: `.cache/mean-pool-comparison-20261111-01.json`. The user confirmed
"Completed all six; capture was normal." Both frozen models receive identical
observations from one MediaPipe extraction for each capture, with the same
normalization, quantization, frame lengths and padding. The actual loaded model
identities are SignGRU epoch 18 and SignMeanPoolGRU epoch 4. The camera is closed.

| Capture | Performed | Original first guess / score | Mean-pool first guess / score |
|---|---|---|---|
| 1 | HELLO | HELLO / 57.26% | HELLO / 60.31% |
| 2 | THANKYOU | THANKYOU / 90.69% | THANKYOU / 99.40% |
| 3 | NO | NO / 94.94% | HELLO / 96.99% |
| 4 | HELLO | NO / 94.39% | HELLO / 91.64% |
| 5 | THANKYOU | THANKYOU / 92.25% | THANKYOU / 99.39% |
| 6 | NO | NO / 95.13% | HELLO / 98.13% |

Original: **5/6**. Mean pool: **4/6**. On the same captures it fixes one baseline
HELLO error but creates two NO errors. This session's baseline count differs
from the previous session's 4/6; use this paired result, not cross-session counts,
to compare the models. Six attempts from one signer are a diagnostic pilot,
not an estimate of real-world webcam accuracy. Scores are uncalibrated.

For the sampled 25/50/75/100% prefixes, the pooled model keeps the same top label
throughout all six captures. The original has three top-label transitions across
those sampled prefixes. This is consistent with increased stability, but both
pooled NO predictions are **stably wrong**. Incomplete prefixes need not contain
a complete sign, so their accuracy is not a separate benchmark.

Artificially appending 16 or 32 training-derived rest frames reduces original
correct counts from 5 to 4 and pooled correct counts from 4 to 3. The pooled
model's first HELLO changes to THANKYOU after appended rest. Pooling is therefore
not generally invariant to capture endings. These controls never changed the
application's real inputs or training data and are not valid fixes.

### Verification and preservation

- **96 project tests passed**, plus compilation, lockfile and dependency checks.
  Coverage includes padding exclusion/gradient isolation, real rest retention,
  unsorted batch order, equal seeded initialization, gradient flow through both
  recurrent layers, checkpoint restoration/source checks, model identity, and
  six automatic paired captures with aggregate-only exports and camera release.
- All **19 validation clips** match cached and runtime-path predictions exactly:
  same rankings, maximum probability difference **0.0**.
- All previous `best.pt` hashes and source-cache payload hashes remain unchanged.
  Candidate SHA256:
  `909e90daffb8d244c7492d1303ab8dfeeddd8b325c0335ad74207711385466c9`.
- No new webcam video, stills, joint coordinates or embeddings were saved. Live
  JSON contains predictions and aggregate measurements only; no training uses it.
- Source extraction/audit remains stopped; no source cleaning or data changes.
- Default recognition checkpoint remains the original epoch-18 GRU. Candidate
  weights remain independently available at
  `models/asl-gru-five-signs-mean-pool-v1/best.pt`.

The result supports keeping temporal aggregation as an experimental direction,
but **plain masked averaging alone does not solve the reported behavior**. Any
follow-up should explicitly measure NO discrimination and robustness to pose/
visibility changes. It must not be selected solely from aggregate test gains or
high webcam scores, and the existing test/live pilot must not become training data.

## Artifacts and reproduction

- [Complete comparison evidence](assets/gru-mean-pooling-2026-11-11.json)
- [Learning curves, per-sign test results and paired live graph](assets/gru-mean-pooling-2026-11-11.png)
- Local experiment, epoch records, test predictions, runtime checks and source
  snapshots: `models/asl-gru-five-signs-mean-pool-v1/` (ignored by Git).

Explicit experimental webcam command:

```powershell
.\.venv\Scripts\python.exe scripts/recognize_webcam.py --checkpoint models/asl-gru-five-signs-mean-pool-v1/best.pt
```

Rebuild the report from saved results without rescoring or retraining:

```powershell
.\.venv\Scripts\python.exe scripts/report_mean_pool_trial.py --live-report .cache/mean-pool-comparison-20261111-01.json
```

## Interactive pooled-model session (2026-11-12)

At the user's request, opened the mean-pooling checkpoint alone with
`scripts/recognize_webcam.py --checkpoint models/asl-gru-five-signs-mean-pool-v1/best.pt --report .cache/webcam-mean-pool-20261112-01.json`.
Startup confirms **SignMeanPoolGRU / epoch 4 / CUDA / five labels** and window
title **GRU mean-pool**. This is the trained mean-pooling model; no max-pooling
model has been trained. Manual Space start/finish controls apply. Only scores
and aggregate timing/tracking metadata are saved. No model/default changes or
training occurred. The user subsequently reported correct first guesses for
HELLO/THANKYOU/PLEASE and incorrect guesses for NO/YES, observing overlapping
finger points and fist-like endings for the latter two. Capture numbers were
not supplied, so no per-trial labels or formal accuracy are inferred. See the
[compact-hand observation and bounded feature check](compact-hand-diagnostic-2026-11-12.md).
