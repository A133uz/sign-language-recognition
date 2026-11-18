# Slovo RSL workflow

Slovo support is additive. The existing ASL Citizen commands, checkpoints,
reports, and defaults remain in place. The separate entry point is
`scripts/slovo.py`.

## Inputs and metadata audit

The default inputs are:

```text
SlovoDataset/annotations.csv
SlovoDataset/slovo_mediapipe.json
```

The Kaggle `annotations.csv` in this checkout is tab-delimited. The importer
detects its delimiter and reads columns by name. It streams the 1.26 GB landmark
JSON one clip at a time rather than loading the full object into memory.

Run the read-only audit:

```powershell
.venv\Scripts\python.exe scripts\slovo.py audit
```

The CSV has 20,400 rows: 20,000 gesture samples and 400 `no_event` rows. The
landmark JSON has 20,000 clip IDs; all match gesture rows, while the 400
`no_event` IDs have no feature records. `no_event` is excluded from the isolated
gesture vocabulary. The gesture set contains 1,000 labels, with 15 train and 5
test examples per label.

The metadata's official `train` flag has 92 `user_id` values in both sides
(112 train signers and 174 test signers). This does not match the README's
description of a subject-based split. The importer preserves the original flag
and reports two evaluation protocols rather than rewriting the source metadata.

## Features and cache

Slovo landmarks contain hands but no body pose, so they use a separate 132-value
per-frame schema: each hand contributes wrist-relative XYZ for 21 points,
normalized by that clip's median palm scale, wrist image XY, and a presence flag.
Hands are placed left-to-right by wrist x. Frames with a third detected hand
drop a near-coincident wrist duplicate when one is within 0.05 frame units;
otherwise the importer keeps the most spatially separated pair. The audit and
cache record how often each rule is used. This is a heuristic because the JSON
does not provide a handedness label or a body pose for assigning hands.

Ten gesture IDs have empty frame lists. The cache retains their metadata with a
zero sequence length; training and evaluation skip those rows instead of making
up landmark values.

Prepare a separate cache; this does not modify the source CSV or JSON:

```powershell
.venv\Scripts\python.exe scripts\slovo.py prepare
```

The default cache is `data/training/slovo-hands-128-v1/`. It stores separate
train, validation, and official-test arrays, exact Russian labels, source hashes,
and split metadata. The validation set is selected by `user_id` groups from the
official training rows. On the supplied CSV, the deterministic selection covers
all labels with 4,097 metadata rows from 7 signers (27.3% of the official
training examples). Six rows have empty landmarks, leaving 4,091 usable
validation sequences. The train cache has 10,901 usable sequences; the official
test cache has 4,998.

## Training and evaluation

The GRU architecture, configuration, data loader, and training loop are in
[`03_slovo_training_and_evaluation.ipynb`](../notebooks/03_slovo_training_and_evaluation.ipynb).
The notebook defaults to report-only mode. After preparing the cache, set
`RUN_TRAINING = True` in its training cell to train; set `SMOKE_TRAINING = True`
for a bounded train/validation pass. Training uses only the train and signer-
grouped validation partitions and never reads the test arrays. CUDA is required.
Training has not been started as part of this migration.

After training, evaluate both the full official flag test set and a stricter
subset whose signers never occur in the official training rows:

```powershell
.venv\Scripts\python.exe scripts\slovo.py evaluate `
  --checkpoint models/slovo-gru-v1/best.pt
```

For the supplied metadata, the unseen-signer subset contains 1,610 rows from
82 signers; one has an empty landmark sequence, leaving 1,609 scored clips. It
covers 639 of 1,000 labels. Its report includes supported-class counts so its
macro score is not mistaken for full-vocabulary coverage. The full official
test score includes samples from signers also present in training.

The live RSL webcam entry point uses the same hands-only schema after a Slovo
checkpoint has been trained:

```powershell
.venv\Scripts\python.exe scripts/recognize_slovo_webcam.py
```

The webcam overlay uses Pillow and a Cyrillic-capable system font to render
Russian predictions.

The original `scripts/recognize_webcam.py` and its ASL checkpoint remain the
ASL workflow. Slovo checkpoints are stored under `models/slovo-gru-v1/`.
