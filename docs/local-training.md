# Local ASL Citizen GRU training

The user authorized local GPU training on 2026-11-09. This is an ASL-only
baseline; no RSL model or webcam sign classifier is included in this step.
Raw extraction is complete: 83,399 successful clips, zero failed clip records,
836 archives (5,372,618,913 bytes). Native worker crashes during extraction
were recovered and must not be confused with failed clip records.

## Environment and selected baseline

Install the locked CUDA extra into the existing environment:

```powershell
uv --cache-dir .cache/uv sync --locked --extra cuda
.venv\Scripts\python.exe scripts/train_local.py cuda-check
```

Verified locally: PyTorch 2.11.0+cu128, CUDA runtime 12.8, RTX 4060 with 8 GB
VRAM. The CUDA check executes a forward/backward operation and synchronizes;
an NVIDIA inventory alone is not treated as proof. The CPU MediaPipe GPU
delegate experiments are unrelated to PyTorch CUDA training.

| Setting | Initial value |
| --- | --- |
| Model | Unidirectional GRU, 2 layers, 256 hidden units |
| Classifier | Dropout 0.3, then Linear(256, 2731) |
| Sequence | Maximum 128 frames, uniformly sampled across longer signs |
| Short sequences | Right-padded; packed sequences exclude padding from GRU state |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 |
| Loss | Cross-entropy on raw logits |
| Batch size | 64 |
| DataLoader | 2 CPU workers, pinned memory, bounded prefetch |
| Arithmetic | CUDA automatic mixed precision with gradient scaling |
| RNN backend | ATen CUDA; cuDNN disabled after a reproducible native Windows GRU shutdown crash |
| Gradient clipping | Maximum norm 1.0; nonfinite losses stop; AMP overflows skip updates and lower the scale; 8 consecutive overflows stop |
| Maximum epochs | 50 |
| Early stopping | 10 epochs without a lower validation loss |
| LR schedule | Halve after 3 plateau epochs via ReduceLROnPlateau |
| Augmentation | None in this first baseline |
| Random seed | 42; not a guarantee of bitwise GPU reproducibility |

These are baseline settings, not an accuracy claim. Subsequent augmentation
or architecture experiments must use validation results rather than test results.
Keep the official signer-separated train/val/test splits:
40,154 / 10,304 / 32,941. Test features may be deterministically prepared,
but the trainer never loads the test split or reports test accuracy.

## Feature preparation

The versioned implementation is `src/sign_language/preprocessing.py`,
`asl-body-local-hands-v1`. It does not modify the raw archives.

1. Correct image aspect ratio using the saved frame dimensions.
2. Assign anatomical left/right slots using pose wrists, handedness and
   short-term wrist continuity; this is a heuristic that requires validation.
3. Subtract each frame's shoulder midpoint and divide by the sign's median
   valid shoulder width. Interpolate only internal shoulder gaps of at most
   five frames. If anchors are unavailable, body features are zero/masked;
   hand-local features may still be present.
4. Add wrist-relative hand XYZ divided by median usable palm size for that
   hand. Hand z stays local; pose z is omitted. Preserve orientation and
   motion direction; no mirroring or rotation canonicalization is applied.
5. Retain masks, clip feature values to [-8, 8] to bound outliers, and sample
   at most 128 observations in order, including endpoints. Do not upsample
   short clips. Padding is separate from landmark absence.

Each time step has 254 features in this order:

| Block | Values |
| --- | ---: |
| 13 upper-body/head pose points, body-relative XY | 26 |
| Valid masks for those pose points | 13 |
| Two hands, 21 points each, body-relative XY | 84 |
| Two hands, wrist-relative XYZ | 126 |
| Hand-present masks | 2 |
| Usable local-hand geometry masks | 2 |
| Valid body-anchor mask | 1 |

Prepared arrays use float16 storage and become float32 batches before CUDA
autocast. The raw float32 landmarks are retained. The exact preprocessing,
source hash, raw run ID and label map identify the cache. No dataset-wide
normalization statistics are fitted, so validation/test information does not
enter a learned feature scaler.

```powershell
.venv\Scripts\python.exe scripts/train_local.py prepare --workers 2
```

Output: `data/training/asl-gru-128-v1/`, about 5.38 GB of feature arrays plus
indexes, lengths, targets and markers. Every raw shard is checksummed and
matched to official records. Prepared-shard markers contain hashes covering
features, lengths and labels; interrupted preparation resumes verified slices.
Verification can take time before the first preparation progress line.

Windows can briefly lock an existing progress file while it is being read.
Atomic JSON/checkpoint replacements now retry permission errors for a bounded
period, leaving the previous complete file intact until replacement succeeds.
The first preparation pass stopped on such a status-file lock; resume verified
the completed slices and finished all 83,399 clips without re-extraction.

## Smoke check and full background run

First run the separate bounded CUDA smoke check (8 training batches and
4 validation batches). Its metrics are diagnostic, not dataset accuracy:

```powershell
.venv\Scripts\python.exe scripts/train_local.py train --smoke --workers 2
```

Smoke output is kept in `models/asl-gru-128-smoke/`. The full run starts
from a fresh seeded initialization, not the smoke checkpoint.

```powershell
Start-Process -WindowStyle Hidden `
  -FilePath "$PWD\.venv\Scripts\python.exe" `
  -ArgumentList @('-u', "$PWD\scripts\local_training_job.py", 'run') `
  -WorkingDirectory "$PWD"
```

Check or stop the job:

```powershell
.venv\Scripts\python.exe scripts/local_training_job.py status
.venv\Scripts\python.exe scripts/local_training_job.py stop
```

The supervisor prevents duplicate instances and requests automatic idle-sleep
prevention only while active. The display can sleep; no power-plan changes
are made. No desktop notification is currently implemented. Attempt logs and
control state are in `.cache/training/`; the status includes live supervisor
state and the trainer's current phase/epoch and most recent complete epoch.

Stopping terminates that job's process tree. Completed epoch checkpoints stay
usable; an interrupted epoch repeats on resume. Run the same supervisor command
again to resume `last.pt` with matching settings. Crashes are reported rather
than automatically retried indefinitely. Do not start another trainer against
the same output directory or change source/configuration mid-run.

## Outputs and interpretation

`models/asl-gru-128-baseline/` contains:

- `config.json`, `labels.json`: architecture, normalization/sequence policy,
  source hashes, cache identity and all output label IDs.
- `best.pt`: inference weights for the lowest validation loss, label map and
  complete model/preprocessing configuration.
- `last.pt`: atomic epoch checkpoint with model, optimizer, scheduler, scaler,
  RNG/shuffle state and early-stopping state.
- `epoch-NNN.json`: loss, top-1/top-5 accuracy, validation macro precision,
  recall/F1 and weighted F1, train/validation time and peak CUDA allocation.
- `status.json`: running/completed/failed, current phase, selected checkpoint
  information and whether test evaluation occurred (false in this trainer).

These logged diagnostic metrics do not finalize the project's presentation
metrics. The final reporting protocol still needs separate discussion.
Measure several full epochs to forecast duration; initial CUDA startup,
DataLoader startup and filesystem caching can make the first epoch slower.
No accuracy or fixed completion time is promised before measurement.

## Initial runtime correction

The first full attempt completed six epochs, reaching validation top-1 34.4%,
then stopped at epoch 7 because the initial gradient guard treated an AMP
overflow as fatal. Those checkpoints/logs are retained under
`models/asl-gru-128-baseline-first-attempt/`. The corrected trainer lets
GradScaler skip an overflowing update and reduce its scale; eight consecutive
overflows or a nonfinite loss still stop the run. A direct CUDA regression test
verifies that an overflowing update leaves weights unchanged and a subsequent
finite update succeeds. The corrected baseline restarts from the original
seed in `models/asl-gru-128-baseline/` with its own configuration/source hashes.
Early full epochs measured roughly 12 seconds including validation, much
faster than the old unmeasured 1–6-hour budget; use the corrected run's epoch
records for its final duration and quality rather than the short smoke check.

Further isolation reproduced process exit `0xC0000409` with a tiny standalone
CUDA GRU after its forward/backward computations, even without MediaPipe or
DataLoader workers and even in full precision. Disabling cuDNN made the same
probe exit normally. The corrected trainer therefore uses the ATen CUDA GRU
path (`torch.backends.cudnn.enabled = False`), recorded in its configuration.
This is a local backend workaround, not a change to the GRU architecture or
a switch to CPU. The earlier cuDNN timing is not a forecast for this backend.

PyTorch references: [GRU](https://docs.pytorch.org/docs/2.11/generated/torch.nn.GRU.html),
[packed sequences](https://docs.pytorch.org/docs/2.11/generated/torch.nn.utils.rnn.pack_padded_sequence.html),
[automatic mixed precision](https://docs.pytorch.org/docs/2.11/amp.html),
[backend controls](https://docs.pytorch.org/docs/2.11/backends.html#torch.backends.cudnn.enabled).

## Current background run: 2026-11-09

The corrected run started at approximately **20:36 Asia/Tashkent**
(`20261109T153620Z` logs in `.cache/training/`). Its first complete epoch used
all 40,154 training clips and all 10,304 validation clips, took **108.234 seconds**
(99.516 train, 8.687 validation), and skipped zero AMP updates. Peak allocated
CUDA memory was 287,053,824 bytes; total device usage also includes CUDA context,
libraries and desktop applications. Its exported best checkpoint was reloaded
and produced finite `[2, 2731]` GPU predictions, with a clean process exit.

This is an initial timing, not a guarantee: 50 epochs at the same speed would
take about 90 minutes, excluding feature preparation and any interruptions.
Early stopping can shorten the run. All **32 checks** passed.

The run subsequently completed successfully after **33 epochs** at approximately
**21:34 Asia/Tashkent**, using early stopping. Its selected checkpoint is epoch
23: validation top-1 **55.73%**, top-5 **82.50%**, macro F1 **54.67%**.
See [the complete results](training-results-2026-11-09.md) and learning curves
for the selection rule, generalization gap and runtime. Final test-set
evaluation and webcam recognition are not part of this training job.
