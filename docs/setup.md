# Environment setup

Supported development targets are Windows x64 and Linux x86-64 with Python 3.12.
Windows is the primary webcam/GPU target; Linux development checks run in CI.

## Windows prerequisites

- Python 3.12, 64-bit, including pip and venv.
- Git and a terminal/editor capable of running Python files.
- Standalone uv 0.13.5, installed once in the user account.
- NVIDIA driver for the CUDA profile; verify with `nvidia-smi`.
- Webcam permission for the later live application.
- Internet access for the initial install and enough disk space for binary wheels,
  dataset clips, cached packages, and extracted landmarks.

The local machine was inspected on 2026-10-25: Python 3.12.6, Git 2.45.1,
NVIDIA RTX 4060 / 8 GB, driver 591.74. This is an environment observation, not
a model-performance benchmark.

Foundation verification on that machine passed: 10 automated tests, Ruff lint
and formatting, vocabulary validation, dependency consistency, source/wheel
builds, and installation of the built wheel into an isolated environment.
OpenCV 5.0.0 and MediaPipe 1.1.1 imported successfully. PyTorch 2.11.0+cu128
completed CPU and RTX 4060 GPU calculations. Hosted CI has been configured but
has not run until the repository is pushed. Webcam capture and sign recognition
are not validated by these setup checks.

## Install uv once

Use the [official standalone installer](https://docs.astral.sh/uv/getting-started/installation/)
for the pinned version. In PowerShell:

```powershell
irm https://astral.sh/uv/0.13.5/install.ps1 | iex
```

The installer places executables under `%USERPROFILE%\.local\bin` and adds that
directory to the user PATH. Open a new terminal after installation and confirm
`uv --version` reports 0.13.5. If VS Code inherited the old PATH, restart VS Code.
The setup/check scripts also recognize the default install path from an existing
terminal. No separate Python environment is needed for uv.

## Manual Windows project setup

Run these commands in the repository, using the standalone uv executable:

```powershell
uv --cache-dir .cache/uv sync --locked --extra vision --extra training --extra cuda
.\.venv\Scripts\python.exe -m sign_language check-config configs/vocabulary.yaml
.\.venv\Scripts\python.exe -m sign_language doctor --vision --require-cuda
```

Replace `--extra cuda` with `--extra cpu` for the CPU profile. Omit all extras for
development-only work. Select an existing interpreter with `--python PATH` if
needed. The setup script's `-Python` parameter supports the same choice.

## Daily use and VS Code

Select `.venv\Scripts\python.exe` as the Python interpreter in VS Code. For an
activated PowerShell terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -m pip list
```

The interpreter should point inside this repository's `.venv`. Pip is included in
the development dependencies, including the full CPU/CUDA development profiles.
Uv is external tooling and will not appear in that list. Without activation, run
`uv pip list --python .venv/Scripts/python.exe`. Use setup to update dependencies;
manual pip installs are not recorded in the lockfile.

The `.cache/uv` directory stores downloaded packages, not another environment.
The project keeps exactly one virtual environment: `.venv`.

## Linux development environment

```bash
curl -LsSf https://astral.sh/uv/0.13.5/install.sh | sh
# Open a new shell so uv is on PATH, then run from the repository:
uv sync --locked
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m pytest
.venv/bin/python -m sign_language check-config configs/vocabulary.yaml
.venv/bin/python -m sign_language doctor
uv build
```

The vision profile may require system display/audio libraries on Linux. CI
intentionally validates the development profile without opening a camera or
assuming GPU access. Linux vision deployment needs its own runtime verification.

## Runtime checks

`doctor` checks base library imports. `--vision` additionally imports OpenCV and
the MediaPipe Tasks HandLandmarker API. `--torch` runs a PyTorch CPU calculation.
`--require-cuda` requires CUDA PyTorch
and executes a small calculation on the GPU. Requested capability failures
return a nonzero exit code.

These checks do not open a webcam, install landmark model assets, or establish
recognition performance. Review the selected MediaPipe Tasks documentation and
download the matching hand/pose `.task` assets into `assets/models/` when the
extraction pipeline is implemented. Record model source and checksum there.

Only one OpenCV distribution should be present. This project uses
`opencv-contrib-python`, which is also required by the selected MediaPipe package;
do not add `opencv-python` or headless variants alongside it.

PyTorch CPU/CUDA indexes are explicit, following
[uv's official PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/).
Both profiles use PyTorch 2.11.0 so switching backends does not change the
framework version. The CUDA profile uses its CUDA 12.8 build.
MediaPipe uses the
[Tasks Python API](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python).
