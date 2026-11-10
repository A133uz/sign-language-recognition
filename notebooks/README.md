# Project records and model results

Two generated reporting notebooks accompany the existing Colab extraction notebook:

- `01_project_workflow_and_investigation.ipynb`: scope/history, data splits,
  extraction/recovery/benchmarks, normalization, model design, training steps,
  webcam failure investigation and remaining work.
- `02_model_training_and_evaluation.ipynb`: actual learning curves, validation
  and frozen-test metrics, grouped CV, per-signer/per-class analysis, focused
  confusion matrices, frequent confusions, runtime, capacity and LR schedules.

Graphs and tables are embedded. The notebooks can be read without the videos,
checkpoints or raw feature cache. Keep `artifacts/experiment_snapshot.json`
beside them to rerun analysis; it contains compact reports, histories and
provenance, not model weights or webcam images/coordinates. Exportable PNGs are
under `artifacts/figures/`. `artifacts/report_manifest.json` records file hashes.

Both notebooks now contain **completed ST-GCN results**, after successful
training and frozen test evaluation on 2026-11-11 (Asia/Tashkent). The metrics
notebook explains epoch-39 selection versus the final epoch 49 and compares
aggregate metrics and individual classes with the GRU. Aggregate gains do not
imply gains for every sign. There are 15 embedded figures across the notebooks.

The completion pipeline generated the first final version automatically.
`--require-complete` refuses incomplete ST-GCN results. Reload any previously
opened draft files to see the final embedded outputs. Updating these reports
does not restart training or change weights.

## Regenerate

```powershell
.\.venv\Scripts\python.exe scripts/build_project_notebooks.py
# Final reports only, after complete training and test evaluation:
.\.venv\Scripts\python.exe scripts/build_project_notebooks.py --require-complete
```

The generator executes the authored analysis cells in a Python process and
stores actual stdout and PNG outputs in notebook v4 format. It does not use a
Jupyter kernel during generation; this execution method is recorded in notebook
metadata. Every code cell is ordinary rerunnable Python. Run All reads the
snapshot and plots data; it does not download/extract data, train/evaluate a
model, open the camera or overwrite weights.

Open the `.ipynb` files in VS Code's notebook editor. Select the project `.venv`
when rerunning. Viewing the saved outputs needs no kernel installation. If VS
Code requests `ipykernel`, install it into that same environment after training:

```powershell
uv pip install --python .venv/Scripts/python.exe ipykernel
```

No additional environment, bootstrap venv or notebook server is required for
these saved reports. The active training environment is not modified by the
report generator. Generated files may be refreshed by that generator; copy a
notebook before making personal edits. The existing
`asl_citizen_landmark_extraction.ipynb` is left untouched.
