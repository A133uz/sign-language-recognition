# Project working rules

## Model design and training

- Implement model architectures and model-training work in Jupyter notebooks.
- Notebooks must contain the actual model-design and training code and its
  configuration, explanations and results; do not substitute a notebook that
  only launches a standalone training script.
- Scripts and shared inference modules needed to connect a model to the webcam
  remain Python files. Keep their definitions consistent with the notebook model.
- Existing historical training scripts and read-only reporting notebooks are
  records. Do not treat them as permission to put new training work in scripts.
- Do not run training merely when creating or updating a notebook.

## Model retention

- Keep a compact, documented set of models with meaningfully different
  architectures, implementations or evaluation vocabularies and useful results.
- Avoid accumulating per-epoch weights, smoke/failed runs and near-duplicate
  hyperparameter variants. Preserve metrics and provenance separately from weights.
- Consult `docs/model-retention-2026-11-14.md` and its catalog for retained models.
- Webcam defaults must reference an available retained model.
