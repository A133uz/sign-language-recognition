# Signer-grouped regularized GRU study

Authorized and started on 2026-11-09, approximately **21:55 Asia/Tashkent**.
Cross-validation measures consistency across held-out signers; it does not
itself regularize a model or guarantee a higher accuracy.

**Completed successfully on 2026-11-10.** The final official-split regularized
model achieved validation top-1 **61.06%**, versus the original baseline's
55.73%. Grouped CV top-1 was **38.29% +/- 2.42 percentage points** on its
different held-out signer partitions. See [completed comparison results](model-comparison-results-2026-11-10.md).

## Protocol

Use three `StratifiedGroupKFold` folds, seed 42, grouped by participant ID,
**within the original 40,154 official training clips only**. The original
10,304 validation clips are excluded from CV. The 32,941 test clips remain
unused throughout this study. Every clip is held out once, each fold's train
and validation signer sets are disjoint, and every partition covers all
2,731 classes. The saved fold plan includes exact indices, metadata/cache
identity and scikit-learn version (1.9.1).

| Fold | Training clips | Held-out clips | Training signers | Held-out signers |
| --- | ---: | ---: | ---: | ---: |
| 1 | 27,329 | 12,825 | 28 | 7 |
| 2 | 25,915 | 14,239 | 13 | 22 |
| 3 | 27,064 | 13,090 | 29 | 6 |

Signer counts differ because contributors supplied very different numbers of
clips; stratification balances class distributions and clip counts rather
than forcing equal numbers of people. Group separation takes priority.

## Regularization candidate

Keep the two-layer 256-unit GRU, 128-frame maximum, batch size 64, CUDA AMP
and verified ATen CUDA path. Change the baseline as follows:

| Setting | Original baseline | Regularized candidate |
| --- | --- | --- |
| Between-layer and output dropout | 0.3 | 0.5 |
| AdamW weight decay | 0.0001 | 0.001 |
| Training augmentation | None | Small per-sign body-coordinate scale and camera-roll perturbations |

Augmentation uses one scale in [0.9, 1.1] and one roll in [-5, 5] degrees per
sign. Apply scale/roll consistently to body-relative pose/hand XY, and the
same roll to wrist-relative hand XY. Leave local hand depth, masks, padding,
sequence length, frame order and labels unchanged. Missing coordinates stay
zero. This is bounded training noise, not rotation canonicalization or
mirroring. Validation is deterministic and receives no augmentation.

This is one combined candidate, not an ablation isolating which individual
change helps. No label smoothing, reversed motion, random sign cropping or
aggressive finger distortion is used in this experiment. No landmarks are
re-extracted and the prepared cache remains unchanged.

## Four sequential runs

1. Train fresh regularized models on each of the three folds. Each uses up to
   50 epochs and validation-loss early stopping with patience 10. Do not carry
   learned weights or optimizer state between folds.
2. Aggregate the selected fold-checkpoint metrics as an unweighted mean and
   sample standard deviation. Report each fold's results as well. Fold holdouts
   are also used for checkpoint selection, so these are development CV scores,
   not an independent final test estimate or a confidence interval.
3. Train a fresh regularized model on the complete official training split,
   using the official validation split for checkpoint selection. This enables
   comparison against the original model's **55.73% validation top-1** under
   the same split and checkpoint-selection rule. The original baseline was not
   rerun on the CV folds, so its official validation score must not be compared
   directly with a CV mean as proof of improvement.

Each run retains its own `best.pt`, `last.pt`, configuration, labels, epoch
metrics and status. The final official-split model is not automatically
promoted over the baseline; compare results first. The original baseline and
its selected-checkpoint SHA-256 remain unchanged. No test or webcam accuracy
is claimed by starting this study.

## Commands and recovery

Install optional grouping support while keeping CUDA:

```powershell
uv --cache-dir .cache/uv sync --locked --extra cuda --extra training
.venv\Scripts\python.exe scripts/cross_validate.py plan
.venv\Scripts\python.exe scripts/cross_validate.py smoke
```

The actual study was launched hidden in the background:

```powershell
Start-Process -WindowStyle Hidden `
  -FilePath "$PWD\.venv\Scripts\python.exe" `
  -ArgumentList @('-u', "$PWD\scripts\local_cv_job.py", 'run') `
  -WorkingDirectory "$PWD"
```

Check or stop:

```powershell
.venv\Scripts\python.exe scripts/local_cv_job.py status
.venv\Scripts\python.exe scripts/local_cv_job.py stop
.venv\Scripts\python.exe scripts/cross_validate.py summary
```

Status includes the live supervisor, study phase and current fold's epoch and
latest complete epoch metrics. Stopping terminates only this study's process
tree; an interrupted epoch repeats. Running the same supervisor command
resumes the current stage from `last.pt` and skips matching completed stages.
Source/configuration changes are rejected on resume. The trainer also repairs
best-checkpoint/metric artifacts if interrupted immediately after committing
`last.pt`. Automatic idle-sleep prevention is scoped to the supervisor, with
no permanent power-plan change. There is no desktop notification.

Study output: `models/asl-gru-128-cv-regularized-v1/`. Fold stage logs are under
its `logs/`; supervisor control/logs are in `.cache/cross-validation/`.
`summary.json` reports CV metrics separately from the final `official_refit`
results; its CV scope flags refer to the folds. The separate
`official_validation_used_for_refit` flag records use of official validation
by the final run. All model/runtime files remain ignored by Git.

Budget roughly **3–5 hours** for all four runs on this machine, based on the
completed baseline's timing. This is a planning estimate; early stopping,
fold sizes, augmentation and other machine load can change it.

A separate 64-unit baseline comparison was queued on 2026-11-10 to run only
after this study finishes successfully. CV continues using its original
256-unit training engine, now frozen under `engine_snapshot/` inside the
study's output directory. Source routing preserves existing hashes for future
stages and resume. See [the small-model experiment](small-gru-comparison.md).

Validation before launch: all **37 checks** passed. A bounded regularized
fold-1 CUDA smoke run completed and exited with code 0. Tests cover signer
separation, full out-of-fold coverage, plan tampering, training-only
augmentation, unchanged masks/padding/local depth, and checkpoint recovery.

References: [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html),
[PyTorch dropout](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Dropout.html).
