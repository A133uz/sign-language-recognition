# Project records and model results

[Notebook 08](08_15_sign_face_gru_50epochs.ipynb) records the separately trained
50-epoch face-aware GRU without early stopping, with the original validation
selection rule, plots, metrics and comparison with the earlier face-aware run.

[Notebook 07](07_15_sign_face_gru.ipynb) implements the separately trained
15-sign hands/pose/face GRU, its frame alignment and augmentation, training,
frozen readout, metrics and graphs. Its saved model and report are under
`models/asl-gru-fifteen-signs-face-v1/`; it has not yet been connected to the
webcam. The same previously inspected test clips are used as the baseline.

The [700-sign classifier comparison](06_700_sign_classifier_comparison.ipynb)
freezes the selected 2×256 GRU and compares its original linear classifier, a
new linear control and a new two-layer classifier. It includes head training,
validation-loss selection, learning curves and previously inspected test-split
metrics. The separately exported combined model and reports are in
`models/asl-700-sign-gru-2x256-two-layer-head-v1/`; webcam integration is pending.

The [700-sign GRU comparison](05_700_signs_gru_comparison.ipynb) implements
separate two-layer/256-unit and five-layer/512-unit GRUs, training and
evaluation on the same selected ASL Citizen classes. It records its class
selection, official signer-disjoint splits, learning curves and metrics in the
notebook and under `models/asl-700-sign-gru-comparison-v1/` as the run proceeds.
The restarted run uses a maximum of 300 epochs, 150 batches of four clips per
epoch, and a shuffled sampler that continues across epoch boundaries. It applies
validation-loss early stopping after a minimum of 38 epochs.
Each sequence is limited to 83 cached landmark frames for training, validation
and test. Clips originally longer than 128 frames were already sampled once
when the source cache was created, before the new 83-frame sampling step.

The independent [five-layer, 512-unit GRU experiment](04_deep_gru_5layer_512.ipynb)
contains its model definition, training and frozen-readout code, per-epoch
accuracy/loss/F1 graphs and final per-class evaluation. Its checkpoint and
metrics are stored under `models/asl-gru-fifteen-signs-5layer-512-v1/`; the
notebook refuses to overwrite that directory on another run. The official test
split has already been inspected during earlier development.
The project `.venv` currently lacks `ipykernel`; select a compatible Jupyter
kernel with the project dependencies, or install a kernel into the environment
before running the notebook in VS Code. The saved notebook already includes the
completed outputs and plots.

**Working rule:** all new model architecture design and training implementation
belongs in Jupyter notebooks, including real definitions/training code rather
than wrappers around training scripts. Webcam integration and shared runtime
inference helpers remain Python files. Historical training scripts and the
read-only reporting notebooks remain records; updating a notebook does not
authorize starting training.

The [2026-11-14 retention record](../docs/model-retention-2026-11-14.md) lists
the seven checkpoints kept after pruning 178 duplicate, weak or marginal
snapshots. Both reporting notebooks show the retained-model table. Historical
curves still load preserved metrics; their presence does not mean those weights
remain installed.

Four generated reporting notebooks accompany the existing Colab extraction notebook:

Notebook 02 also records the independent **official pretrained ASL Citizen
I3D** baseline: original 2,731-label predictions, explicitly restricted five/
fifteen-label comparisons on identical test videos, and four confirmed webcam
trials. No MediaPipe input, training or existing-recognizer replacement is used.
See [the source and protocol](../docs/official-asl-citizen-baseline-2026-11-14.md).

Notebook 02 now starts with the completed **2026-11-14 official-test sweep**:
all 152 saved five/fifteen-sign checkpoint paths (133 distinct files), including
every stored epoch. It records selected-snapshot metrics, all-epoch test curves
and reference confusion matrices. Selected five-sign geometry epoch 8 scores
51/60; stored epoch 5 scores 59/60 retrospectively. Current fifteen-sign C=0.1
scores 147/182. These are historical sweep results; subsequent curation retains
the geometry epoch-5 checkpoint under its original filename and removes others.
See [the full record](../docs/small-vocab-snapshot-test-2026-11-14.md).

- `01_project_workflow_and_investigation.ipynb`: scope/history, data splits,
  extraction/recovery/benchmarks, normalization, model design, training steps,
  webcam failure investigation and remaining work.
- `02_model_training_and_evaluation.ipynb`: ASL learning curves, validation
  and frozen-test metrics, grouped CV, per-signer/per-class analysis, focused
  confusion matrices, frequent confusions, runtime, capacity and LR schedules.
- `03_asl_citizen_class_balance_and_distribution.ipynb`: all 2,731 classes and
  83,399 source clips; official split counts, count histograms, imbalance
  statistics, ranked/Lorenz plots, smallest/largest classes, signer distributions,
  the current five-sign subset, and an inspectable full-class table.
- `03_slovo_training_and_evaluation.ipynb`: Slovo GRU run comparison, official and
  unseen-signer test metrics, and split-coverage caveats. The notebook includes
  model-design and training cells; it does not start training by default.

Notebooks **01 and 02 now start with the trajectory-aware 15-class readout**:
frozen epoch-13 two-layer 256-unit GRU, masked mean pooling, 17 hand/body + six
wrist-path summaries, concatenation to 279 values and a refitted float32
15-output head. The selected C=0.1 readout scores 54/60 development validation,
PLEASE 4/4 and MY 4/4. Fresh application processing of their eight known source
videos preserves both results. These clips informed tuning; webcam and new
signer performance remain unverified. GOOD regresses by one clip. Both
notebooks retain the original 17-feature model as the earlier baseline.

That baseline has two 256-unit recurrent layers, masked mean pooling, 17 explicit
geometry summaries, concatenation to 273 values and a 15-output linear head.
They explain augmentation order and the separate training/inference paths.
Notebook 02 includes completed matched temporal/control curves, class recall
and confusion matrices. The selected temporal model scored 50/60 validation
versus 49/60 for the control; both missed all four validation PLEASE examples.
Those original training comparisons used train/validation only. Subsequent
live diagnostics and the frozen test sweep above provide the current status.
Historical full-vocabulary GRU/ST-GCN results are labeled separately.

Notebook 02 also records the **2026-11-14 ME/WE investigation**: all 37 relevant
training/validation clips, exact checkpoint/input reproduction, additive
classifier contributions, matched controls and nine confirmed live trials.
The user subsequently clarified that WE was misunderstood in both sessions;
WE captures 2, 5 and 8 are excluded, and earlier WE-failure conclusions are
withdrawn. The instrumented follow-up retains about 29.7 FPS but still gets
ME 1/3 and PLEASE 2/3. Matched prefix windows with refitted versus fixed
normalization both give ME 0/3 and PLEASE 3/3. These results keep the remaining
ME failure visible. No model/default change, personal media, landmark arrays
or embeddings are introduced.

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

The Slovo report reads its saved epoch records, cache manifest and evaluation
JSON. Its model definition and training loop are in the notebook, with training
explicitly disabled by default. Saved outputs remain readable without checkpoints
or feature arrays.

The first two notebooks contain **completed ST-GCN results**, after successful
training and frozen test evaluation on 2026-11-11 (Asia/Tashkent). The metrics
notebook explains epoch-39 selection versus the final epoch 49 and compares
aggregate metrics and individual classes with the GRU. Aggregate gains do not
imply gains for every sign. There are now **25 embedded figures** across the
first two notebooks: six in notebook 01 and nineteen in notebook 02. The
dataset-distribution notebook and its separate snapshot remain unchanged.

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

The ASL generators execute authored analysis cells in a Python process and
store actual stdout, tables and PNG outputs in notebook v4 format without a
Jupyter kernel. Run All reads their snapshots and plots data. The Slovo report
cells read its saved history, cache manifest and evaluation JSON; the separate
training cells contain the GRU definition and training loop and are gated off by
default. Viewing or rerunning the reports does not train/evaluate a model, open
the camera or overwrite weights.

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
