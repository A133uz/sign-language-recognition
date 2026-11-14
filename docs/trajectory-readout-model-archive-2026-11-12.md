# Latest tested trajectory readout: local model archive

> Historical record. This standalone ZIP was superseded during the user's
> 2026-11-14 cleanup. The exact C=0.1 checkpoint remains unchanged in the
> [curated model inventory and backup](model-retention-2026-11-14.md).

Created 2026-11-12; independently reverified 2026-11-14 (Asia/Tashkent).
This preserves the exact model used in the latest recorded webcam session,
`webcam-trajectory-c01-20261112-234320`.

- Archive: `models/archives/asl-gru-fifteen-signs-trajectory-readout-c01-2026-11-12.zip`
- Size: **14,176,820 bytes (13.52 MiB)**.
- Archive SHA256: `8b5fe5e4d052651aeeaf2bcf2a0926a697c89f021b2f7e5e64f35d7f6c6538bb`.
- Sidecar: the same filename followed by `.sha256`.
- [Complete member/hash catalog](assets/trajectory-readout-c01-archive-2026-11-12-manifest.json).

## Model and contents

The archive has one recognition checkpoint:
`models/asl-gru-fifteen-signs-trajectory-readout-c01/best.pt`.
Its SHA256 is `34b766c393cf20a77b001004376195e60ffb58943d7b1f9661bd23edbab81c91`.
It is the 15-label frozen epoch-13 GRU with 23 hand/path descriptors and the
C=0.1 refitted linear readout. Epoch 13 identifies the inherited GRU backbone.

Included: weights, configuration, label map, experiment/evaluation reports,
the run's source snapshot, two MediaPipe task assets, a separate project
Python/dependency snapshot, an internal restore guide and `MANIFEST.json`.
There are **171 payload members**, plus the manifest. The archive contains
no dataset videos/features, personal webcam media or weekly reports.

This backup preserves a model, not a claim of recognition quality. Development
validation was 54/60; the [subsequent live check](trajectory-readout-live-feedback-2026-11-12.md)
still found five ME attempts predicted as WE. See the
[model experiment](please-trajectory-fix-2026-11-12.md) for the training and
validation-tuning limitations.

## Verification and restoration

The independent verification read every ZIP payload for CRC/SHA256, checked the
archive against both sidecar and catalog, compared the internal manifest hash,
verified the nine trained source fingerprints and all 15 labels, and rehashed
every original payload source. All checks passed; originals are unchanged.
The older archives are preserved. Archive binaries remain local and Git-ignored.
The archive/catalog creation tests also passed, including selected-run isolation,
original-file preservation, unsafe names and tamper detection.

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath models/archives/asl-gru-fifteen-signs-trajectory-readout-c01-2026-11-12.zip
.\.venv\Scripts\python.exe scripts/archive_model_weights.py --verify models/archives/asl-gru-fifteen-signs-trajectory-readout-c01-2026-11-12.zip
```

To restore, extract into a **new empty staging directory**, following the
internal `README_RESTORE.md`. Keep `best.pt`, `config.json`, `labels.json` and
the source snapshots together. Copy the MediaPipe assets if needed. The
application's current source fingerprints must match the archived configuration;
`project_source/` preserves the creation-time code and dependencies. Do not
overwrite existing models or current project code to inspect this backup.

Creation command (existing output is preserved; a repeat needs new paths):

```powershell
.\.venv\Scripts\python.exe scripts/archive_model_weights.py --run asl-gru-fifteen-signs-trajectory-readout-c01 --output models/archives/asl-gru-fifteen-signs-trajectory-readout-c01-2026-11-12.zip --catalog docs/assets/trajectory-readout-c01-archive-2026-11-12-manifest.json
```

Only this selected run's weights are in the new ZIP. Other model runs remain
in place and in earlier archives. No training, webcam capture or Git push was
performed to create or verify this backup.
