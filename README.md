# Real-time ASL recognition

Python project for recognizing isolated American Sign Language signs from webcam
video using landmarks and a sequence model.

**Current state:** repository foundation, reproducible dependency profiles,
configuration validation, and environment diagnostics. Dataset preparation,
training, and live recognition are the next implementation steps. No trained
model or recognition accuracy is claimed yet.

## Scope

- Start with 10 everyday ASL signs, expanding to 25 after the pipeline works.
- Use WLASL as the initial source; review available clips before finalizing labels.
- Use MediaPipe landmarks and a PyTorch GRU/LSTM baseline.
- Target Windows x64, Python 3.12, and an RTX 4060 with 8 GB VRAM.
- Keep RSL as a later comparison. The supervisor accepts this split.
- Prioritize the working application and university report; publication is optional.

The draft vocabulary is in `configs/vocabulary.yaml`. `goodbye` is the initial
farewell class; `bye` is not a separate class pending overlap review. `welcome`
is not silently treated as “you're welcome.”

## Quick start on Windows

Run from a PowerShell terminal in this repository:

```powershell
.\scripts\setup.ps1 -Profile cuda
.\scripts\check.ps1
.\.venv\Scripts\python.exe -m sign_language doctor --vision --require-cuda
```

The setup script installs uv 0.13.5 in a local `.bootstrap` environment, then
creates `.venv` from the committed `uv.lock`. CUDA uses the explicit PyTorch
CUDA 12.8 index. The first full install downloads large binary packages.
Nothing needs to be installed into the global Python environment.

Use `-Profile dev` for configuration development without vision/PyTorch, or
`-Profile cpu` for the full stack with CPU PyTorch. Profiles share one `.venv`;
running setup switches its installed dependencies. CPU and CUDA are mutually
exclusive. Scripts use the virtual environment directly; activation is optional.

If local script execution is disabled, use the manual commands in
[the setup guide](docs/setup.md). They require no execution-policy changes.

## Repository layout

```text
src/sign_language/    Shared package and setup CLI
configs/             Versioned vocabulary and future experiment configuration
scripts/             Setup and verification commands
tests/               Configuration and diagnostic failure checks
data/manifests/      Versioned sample IDs, signer IDs, and split records
data/raw/            Local videos (ignored)
data/processed/      Generated landmarks (ignored)
assets/models/       MediaPipe model files (ignored)
checkpoints/         Trained model weights (ignored)
runs/                Generated experiment output (ignored)
reports/             Curated results and experiment records
```

## Development and Git

Dependencies are declared in `pyproject.toml`; resolved versions and hashes live
in `uv.lock`. Do not maintain a second handwritten requirements list. GitHub
Actions checks Windows and Linux development environments, lint, formatting,
tests, configuration, and packaging. Vision and GPU operation are checked locally.

See [CONTRIBUTING.md](CONTRIBUTING.md) for commands and Git workflow,
[docs/data.md](docs/data.md) for dataset rules, and
[docs/project-decisions.md](docs/project-decisions.md) for current decisions.
The [original summary](sign_language_project_summary.md) remains the longer
research proposal; its initial scope has been refined by the decision record.

## Licensing

No open-source code license has been selected yet. Dataset and pretrained-model
licenses are separate from the application code; see `docs/data.md`. Do not
redistribute WLASL videos as part of this repository.
