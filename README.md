# Real-time ASL recognition

Python project for recognizing isolated American Sign Language signs from webcam
video using landmarks and a sequence model.

**Current state:** an experimental isolated-sign pipeline can extract hand landmarks
from local WLASL clips, train a small GRU, and classify a short webcam capture.
An alternative can run a pretrained WLASL2000 video model. In an initial user
test, pilot-only video recognition correctly guessed `hello`, `thank you`, `please`
and `sorry`; `yes` and `no` were not first guesses. The unrestricted model and
earlier landmark model misclassified `hello` and `thank you`. Both
approaches are experimental and not reliable general-purpose sign recognizers.
A newly trained native ASL Citizen hand-shape model is available through the
guided landmark backend; it scored 104/146 on source held-out clips. In a guided
webcam test it matched yes/no on 3/3 attempts each, but me on 0/3, with 9/15
overall prompt agreement. Its native labels and scope are separate from
the full WLASL target.
Follow-up same-capture webcam experiments scored 9/15 for a timing-augmented
candidate versus original 7/15, and 3/10 for a geometry candidate versus original
6/10. Neither resolves the live failures; the geometry candidate was rejected
as a replacement. Original weights remain frozen. Records:
[timing](reports/2026-11-05-timing-augmentation.md) and
[geometry](reports/2026-11-05-body-geometry.md).
See [docs/data.md](docs/data.md)
for data preparation, split limitations, and reproducible commands.

## Scope

- Aim to support all WLASL glosses, subject to data availability and evaluation.
- Use the 10 everyday signs as an initial pipeline test, not the final vocabulary.
- Use WLASL as the initial source; review available clips before finalizing labels.
- Use MediaPipe landmarks and a PyTorch GRU/LSTM baseline.
- Target Windows x64, Python 3.12, and an RTX 4060 with 8 GB VRAM.
- Keep RSL as a later comparison. The supervisor accepts this split.
- Prioritize the working application and university report; publication is optional.

The draft pilot vocabulary is in `configs/vocabulary.yaml`; it is not the full
WLASL label set. `goodbye` is the initial
farewell class; `bye` is not a separate class pending overlap review. `welcome`
is not silently treated as “you're welcome.”

## Quick start on Windows

Install standalone uv 0.13.5 once using [the setup guide](docs/setup.md), then
run from a PowerShell terminal in this repository:

```powershell
.\scripts\setup.ps1 -Profile cuda
.\scripts\check.ps1
.\.venv\Scripts\python.exe -m sign_language doctor --vision --require-cuda
```

Uv is a standalone development tool installed in your user account. The setup
script creates or updates the single project environment, `.venv`, from the
committed `uv.lock`. CUDA uses the explicit PyTorch CUDA 12.8 index. The first
full install downloads large binary packages. Project libraries stay in `.venv`.

Use `-Profile dev` for configuration development without vision/PyTorch, or
`-Profile cpu` for the full stack with CPU PyTorch. Profiles share one `.venv`;
running setup switches its installed dependencies. CPU and CUDA are mutually
exclusive. Scripts use the virtual environment directly; activation is optional.

To inspect the installed project libraries, activate the environment and use pip:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip list
```

Uv itself is external to `.venv`, so it does not appear in that package list.
Without activation, use `uv pip list --python .venv/Scripts/python.exe`.
Manual setup commands and VS Code interpreter guidance are in
[the setup guide](docs/setup.md).

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
in `uv.lock`. Do not maintain a second handwritten requirements list. Run
`scripts/check.ps1` locally for lint, formatting, tests, and configuration checks.
The setup guide describes the separate vision and GPU checks.

To inspect a gloss in the locally downloaded WLASL metadata:

```powershell
.\.venv\Scripts\python.exe -m sign_language inspect-wlasl "hello"
```

This counts metadata entries, not confirmed downloadable videos. See
[docs/data.md](docs/data.md) for details and an alternative metadata path.

To inspect the full metadata and save per-gloss counts locally:

```powershell
.\.venv\Scripts\python.exe -m sign_language audit-wlasl --output runs/wlasl_metadata_audit.json
```

To test a few direct MP4 links and decode their first frame:

```powershell
.\.venv\Scripts\python.exe -m sign_language probe-wlasl "hello" --limit 3
```

This temporarily downloads small files and removes them after testing. It skips
YouTube pages and does not establish full-clip or dataset-wide availability.

To create a provisional signer-ID-disjoint pilot manifest:

```powershell
.\.venv\Scripts\python.exe -m sign_language split-wlasl --output runs/my_pilot_split.json
```

The output must be a new file. It lists metadata records, not verified available
clips; inspect its coverage and limitations before using it for training.

To inventory locally obtained files named by WLASL video ID:

```powershell
.\.venv\Scripts\python.exe -m sign_language inventory-wlasl --videos data/raw/videos --output runs/video_inventory.json
```

Create the video directory first. This checks local decoding; it does not download
clips or verify their sign intervals. See [docs/data.md](docs/data.md).

For the experimental pretrained WLASL2000 webcam test, install the CUDA profile,
obtain the separately distributed checkpoint and class list as described in
[docs/data.md](docs/data.md), then run:

```powershell
.\.venv\Scripts\python.exe -m sign_language webcam-i3d --device cuda --vocabulary configs/vocabulary.yaml
```

Press Space and perform one sign; press Q to close the window. The three displayed
labels are guesses from the ten-sign pilot, and the model cannot reliably identify
unknown gestures. Omit `--vocabulary` to choose among the full model's labels;
the unrestricted mode performed poorly in the initial webcam test.

For a guided check with randomized prompts and a predictions-only JSON report:

```powershell
.\.venv\Scripts\python.exe -m sign_language test-webcam-i3d --device cuda --output runs/webcam_check.json
```

Press Space to start a two-second countdown, then perform the prompted sign.
The check repeats each configured sign three times and compares preprocessing
on the same capture. Press Q to finish early and save the completed trials.
It keeps frames in memory and saves labels and timing only. Prompt agreement
still requires review of the performed signs; it is not independent accuracy.

An optional `--backend citizen` compares the separate ASL Citizen pretrained
baseline, using its own native vocabulary and preprocessing. It has passed
saved-clip CUDA checks but has not yet passed a human webcam test. Setup and
limitations are in [docs/data.md](docs/data.md).

Add `--quality` to the guided check to see the actual model crop and save
aggregate hand-size and crop-visibility measurements. The optional `--align-body`
comparison uses a separately generated training-only framing reference; see
the data guide. It is experimental and does not change the default recognizer.

See [CONTRIBUTING.md](CONTRIBUTING.md) for commands and Git workflow,
[docs/data.md](docs/data.md) for dataset rules, and
[docs/project-decisions.md](docs/project-decisions.md) for current decisions.
The [original summary](sign_language_project_summary.md) remains the longer
research proposal; its initial scope has been refined by the decision record.

## Licensing

No open-source code license has been selected yet. Dataset and pretrained-model
licenses are separate from the application code; see `docs/data.md`. Do not
redistribute WLASL videos as part of this repository.
