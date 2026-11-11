# Model archive and Git delivery — 2026-11-12

The user requested pushing all project changes and archiving all new models'
weights. The previous archive was local and Git-ignored; this backup follows
that arrangement. Model ZIP bytes remain local, while this guide, the complete
member/hash catalog, source, notebooks, graphs and reports are published to Git.

## Contents and preservation

Archive: `models/archives/all-model-checkpoints-2026-11-12.zip`.
Sidecar: `models/archives/all-model-checkpoints-2026-11-12.zip.sha256`.

This is a complete snapshot of retained models, not just the latest best files:

- Five-sign GRUs: both early-training versions, baseline and regularized runs,
  and the new mean-pooling model, including retained per-epoch weights.
- Five-sign ST-GCN: earlier run and separately preserved mean-pooling repeat.
- Full-vocabulary GRU/ST-GCN, 64-unit GRU, CV folds/refit, last/resume files,
  and earlier attempts/smoke runs. Diagnostic/incomplete runs are preserved for
  recovery, not recommended for inference.
- Configurations, labels, metrics, source/engine snapshots and MediaPipe `.task`
  assets. Model aliases and repeated checkpoints remain separate source files.
- A separate `project_source/` snapshot of current Python source/tests and
  dependency files, with per-file hashes. Its base Git commit predates this
  delivery; the actual worktree source bytes are identified by the manifest.

Exclude old archives (including the prior epoch-37 GRU ZIP), lock files,
compiled caches and temporary files from the new ZIP. Old archives and all
original models stay in place. Dataset videos/features and `.cache` webcam
stills/logs are not part of this model archive. No model is trained or rescored
by archiving. Existing default checkpoint and class splits remain unchanged.

The archive contains an internal `MANIFEST.json` and `README_RESTORE.md`.
Every payload member has its size and SHA256 recorded. Verification reads every
member completely for CRC and SHA256, then rechecks original files for mutation.
The archive itself has a SHA256 sidecar and a Git-tracked catalog at
[`assets/model-archive-2026-11-12-manifest.json`](assets/model-archive-2026-11-12-manifest.json).

## Restore without overwriting original models

Keep the ZIP separately from the repository clone. A Git clone alone does not
include these binaries. First compare its SHA256 with the sidecar/catalog:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath models/archives/all-model-checkpoints-2026-11-12.zip
.\.venv\Scripts\python.exe scripts/archive_model_weights.py --verify models/archives/all-model-checkpoints-2026-11-12.zip
```

Extract into a **new empty staging folder**:

```powershell
if (Test-Path -LiteralPath .cache/model-restore-20261112) { throw 'Choose a fresh staging folder.' }
Expand-Archive -LiteralPath models/archives/all-model-checkpoints-2026-11-12.zip -DestinationPath .cache/model-restore-20261112
```

Copy only the selected `models/<run>` folder to a new destination model folder.
Keep `best.pt`, `config.json`, `labels.json` and its source snapshots together.
Restore the root `models/*.task` files if needed. Use the explicit checkpoint
argument, for example:

```powershell
.\.venv\Scripts\python.exe scripts/recognize_webcam.py --checkpoint models/asl-gru-five-signs-mean-pool-v1/best.pt
```

The recognizer's source-hash checks must still agree with the saved config.
Historical code snapshots are preserved for this reason. Current `.gitattributes`
keeps Python sources at LF and preserves JSON/notebook artifact bytes verbatim
so checkout conversion does not invalidate source/metadata hashes. The archive's
`project_source/` is the immutable creation-time snapshot; the repository's final
weekly-report exclusion and artifact-byte rules were adjusted afterward. Weekly
report files were never archive payloads. Training resumption also requires the matching
dataset/feature cache, which is not included here.

## Delivery scope

Include the full reviewed worktree: five-sign training/diagnostics, pooled GRU,
ST-GCN repeat, all-class dataset notebook and its snapshot/figures, scalar live
reports and documentation. Per the user's final delivery instruction, **weekly
reports are excluded**: the entire `output/weekly_progress_reports/` directory
remains local and is ignored, including its DOCX files and ZIP bundle.
Model/data archive binaries also remain excluded from Git.
No secrets-pattern hits or files above 50 MiB were found in the reviewed source/
deliverable set. The full **108-test suite** passed, including archive roundtrip,
preservation, recursive-archive exclusion, unsafe names and tamper detection.

## Completed archive verification

- ZIP size: **788,180,748 bytes** (about **751.7 MiB**).
- **120 checkpoint files** and **2 MediaPipe task files**, retaining selected,
  resume, per-epoch, smoke and attempted-run weights.
- **808 payload members** plus `MANIFEST.json`: every payload member passed ZIP
  CRC and SHA256 verification; original source files also passed post-copy hashes.
- ZIP SHA256:
  `3e4374c88770ab1eea0b945e97712644b0d334424a59e09f9e02d35f7d5c8be1`.
- The old epoch-37 archive and all original models remain untouched. The new
  archive/catalog agree, and no excluded archive/lock/data path is a payload.
- Archive binaries stay local and ignored. The Git-tracked catalog identifies
  every retained file and selected checkpoint and supports independent checking.
- A fresh delivery-time comparison verified all **695 current model files**
  against the archive manifest, including every retained checkpoint. The ZIP
  SHA256 matches its sidecar and catalog. Weekly reports remain local and are
  excluded from the staged delivery.
