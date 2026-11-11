# Project records and model results

Three generated reporting notebooks accompany the existing Colab extraction notebook:

- `01_project_workflow_and_investigation.ipynb`: scope/history, data splits,
  extraction/recovery/benchmarks, normalization, model design, training steps,
  webcam failure investigation and remaining work.
- `02_model_training_and_evaluation.ipynb`: actual learning curves, validation
  and frozen-test metrics, grouped CV, per-signer/per-class analysis, focused
  confusion matrices, frequent confusions, runtime, capacity and LR schedules.
- `03_asl_citizen_class_balance_and_distribution.ipynb`: all 2,731 classes and
  83,399 source clips; official split counts, count histograms, imbalance
  statistics, ranked/Lorenz plots, smallest/largest classes, signer distributions,
  the current five-sign subset, and an inspectable full-class table.

Graphs and tables are embedded. The notebooks can be read without the videos,
checkpoints or raw feature cache. The first two use
`artifacts/experiment_snapshot.json`: compact reports, histories and provenance,
not model weights or webcam images/coordinates. Their exportable PNGs are under
`artifacts/figures/`, with hashes in `artifacts/report_manifest.json`.

The new distribution notebook uses the separate
`artifacts/asl_citizen_distribution_snapshot.json`. It includes aggregated
class/signer counts from local official metadata, verified against cached target
counts. Its seven PNGs are under `artifacts/dataset_distribution_figures/`, with
hashes in `artifacts/dataset_distribution_manifest.json`. Keep the relevant
snapshot beside its notebook when copying files. Run All reads the snapshot;
the final table's SEARCH/SORT_BY/LIMIT settings let you inspect every class.
This report counts metadata/labels; it does not decode videos or audit landmarks.

The first two notebooks contain **completed ST-GCN results**, after successful
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
# Dataset balance only; preserves both model-report notebooks:
.\.venv\Scripts\python.exe scripts/build_dataset_distribution_notebook.py
```

The generators execute authored analysis cells in a Python process and
store actual stdout, tables and PNG outputs in notebook v4 format. They do not use a
Jupyter kernel during generation; this execution method is recorded in notebook
metadata. Every code cell is ordinary rerunnable Python. Run All reads its
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
