# Curated models and notebook-first training — 2026-11-14

The user's working rule is recorded in [AGENTS.md](../AGENTS.md): implement new
model architecture and training code in Jupyter notebooks, with configurations,
explanations and results. A notebook that merely launches a training script does
not satisfy this rule. Webcam/inference integration may remain in Python files.
Historical scripts and reporting notebooks remain as records. No training was
started by this cleanup.

## Retained references

The local checkpoint inventory was reduced from **185 files to seven**. Every
retained file has its original bytes, configuration and label map. The catalog
records checkpoint paths, hashes, dependencies and evaluation scopes:
[retained models](assets/retained-models-2026-11-14.json).

| Model | Vocabulary | Epoch | Recorded test top-1 | Why retained |
|---|---:|---:|---:|---|
| Regularized last-state GRU | 2,731 | 37 | 49.92%, 32,941 clips | Original full-vocabulary sequence pipeline |
| Regularized last-state GRU | 5 | 18 | 70.00%, 60 clips | Plain subset baseline and existing webcam default |
| Mean-pooled GRU | 5 | 4 | 78.33%, 60 clips | Pooling comparison without geometry |
| Mean-pooled GRU + geometry | 5 | 5 | 98.33%, 60 clips | Strong geometry representative |
| Frozen GRU + trajectory readout | 15 | 13 | 80.77%, 182 clips | Latest expanded-vocabulary implementation |
| ST-GCN27 | 2,731 | 39 | 57.25%, 32,941 clips | Distinct graph-convolution architecture |
| Official pretrained I3D | 2,731 outputs | Authors' weights | 64.29%, matched 182-clip subset | Independent pixel-input baseline |

These scores have different vocabularies and test scopes; they are not a single
ranking. Geometry epoch 5 is the retrospective leader of the completed test
benchmark. It is retained as such, **not** presented as a checkpoint selected
independently on validation. Its original `epoch-005.pt` filename is unchanged.
None of these scores establishes reliable webcam performance.

## Removed weights and preserved history

Removed **178 checkpoint files**, including failed/smoke runs, resume/last/epoch
duplicates, weak reduced-size baselines, repeated subset ST-GCN experiments and
near-equivalent readout/hyperparameter variants. Older broad and overlapping ZIP
archives were removed too, so unwanted weights do not survive in local backups.

Before deletion, **748 text metadata/source files** were copied and hash-verified
under `data/model-history/2026-11-14/models/`, preserving their original relative
paths. Metrics, configurations, source snapshots and evaluation reports remain
available. This history contains no removed checkpoint or archive payloads.
The reporting notebook builder resolves moved metadata from this history. Old
curves and benchmark rows remain historical results even when weights are gone.

Some historical diagnostic/training commands reference removed weights and can
no longer be replayed unchanged. Do not restore those weights merely to make an
old command pass. Use the retained catalog for current inference choices.
The existing default five-sign regularized GRU remains available; no recognizer
default, preprocessing or detector model was changed.

Dataset videos, feature caches, evaluation outputs and weekly reports were not
cleanup targets. No retraining, test rerun, Git commit or push was performed.

## Verification and backup

- [Frozen deletion inventory](assets/model-retention-plan-2026-11-14.json):
  each original file's size, SHA256 and keep/preserve/remove action.
- `scripts/prune_model_snapshots.ps1` checked containment, reparse points and
  every original hash before deleting individual model files. Metadata was
  verified first; all retained hashes were checked again afterward.
- [Native load checks](assets/retained-model-load-checks-2026-11-14.json):
  all seven checkpoints instantiate successfully with their actual model classes.
- Local curated backup: `models/archives/curated-models-2026-11-14.zip`.
  It contains only the seven retained checkpoints, detector assets, metadata
  history and notebook/Python/source/dependency records. It excludes dataset
  videos, feature caches and live media. Weights and ZIP bytes remain Git-ignored.
- [Backup member/hash manifest](assets/curated-model-archive-2026-11-14-manifest.json)
  and [final cleanup accounting](assets/model-retention-result-2026-11-14.json)
  record the verified result and space usage.

The older 2026-11-12 archive guides/manifests remain historical provenance. Their
ZIP files were superseded by this curated backup and are no longer installed.
