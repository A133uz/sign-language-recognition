# ASL Citizen and MediaPipe hand tracking

The current application shows live hand skeletons from a webcam. The dataset is
Microsoft ASL Citizen. The WLASL/I3D recognition implementation and its local
videos and weights have been removed.

There is no trained sign classifier yet. Training is on hold until the user
explicitly authorizes it. MediaPipe supplies a pretrained hand-landmark detector;
we will later train a GRU classifier on extracted landmarks. A small Transformer
is a possible comparison model. OpenCV captures frames; MediaPipe finds joints;
the future classifier will assign sign labels.

## 1. Install the environment

Use Python 3.12 and standalone uv. From the project root:

```powershell
uv --cache-dir .cache/uv sync --locked
.\scripts\activate.ps1
python scripts/project.py setup-hands
```

The single project environment is `.venv`. MediaPipe 1.2.2 and OpenCV-contrib
4.11 provide CPU hand tracking. Optional CPU/CUDA torch extras are reserved for
later classifier work; they are not required for the preview. No tooling venv is
used. Run `scripts/check.ps1` for lockfile, compilation and unit checks.

## 2. Download the official dataset

Sources:

- [Microsoft project and download instructions](https://www.microsoft.com/en-us/research/project/asl-citizen/)
- [Official code and model-specific READMEs](https://github.com/microsoft/ASL-citizen-code)
- [Pose-based preparation guide](https://github.com/microsoft/ASL-citizen-code/tree/main/ST-GCN)
- [Dataset license](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/)

The official guide begins by downloading and unzipping ASL Citizen. This project
uses the exact Microsoft ZIP URL from that guide:

```powershell
python scripts/project.py download-dataset
python scripts/project.py extract-dataset
python scripts/project.py audit-dataset
```

Do not run a second downloader while one is active. Interrupted downloads resume
from the retained prefix; HTTP ranges and archive version are validated. The
archive is 45,924,134,223 bytes (42.77 GiB); extracted ZIP entries total
49,604,368,459 bytes. Allow about 100 GB for archive plus extracted data.

The ZIP stays at `data/downloads/ASL_Citizen.zip`. Extraction verifies each member's
CRC and rejects unsafe paths. The archive SHA-256 and setup audit are recorded in
`data/asl_citizen/setup-report.json`. This is a local fingerprint; no publisher
SHA-256 has been independently established. CRC verification is not a claim that
every video has been decoded successfully.

The expected extracted layout is:

```text
data/asl_citizen/ASL_Citizen/
  videos/*.mp4
  splits/train.csv
  splits/val.csv
  splits/test.csv
```

Keep the official splits: 40,154 train / 10,304 validation / 32,941 test videos,
2,731 labels and 52 participants. The audit checks file presence, counts, class
coverage, duplicate filenames and participant separation. A label map is created
from sorted training glosses, preserving the source gloss strings.

To audit just the official CSV metadata after extraction, without checking
every video file:

```powershell
python scripts/project.py audit-metadata data/asl_citizen/ASL_Citizen/splits
```

The initial setup also fetched CRC-checked CSV snapshots separately during the
download under local `data/official-metadata/`. Those ignored snapshots are not
created by `download-dataset` and are not needed for a fresh installation.

Local setup completed on 2026-11-07: the entire archive was downloaded and
CRC-checked during extraction. All 83,399 video files passed the presence audit,
with the official counts, 2,731 labels and disjoint participants intact. See
`data/asl_citizen/setup-report.json` for the local archive fingerprint and audit.
The full video corpus has not been decoded or passed through MediaPipe yet.

## 3. See your hand skeletons

```powershell
python scripts/project.py webcam --report .cache/hands-preview-01.json
```

Move both hands into view, spread and curl your fingers, turn your palms and make
a fist. The window draws 21 points and connecting lines per detected hand, and
shows hand count, point count and measured FPS. Press **Q** or Escape to close.
Use `--camera 1` to choose another device; `--duration 60` closes after 60 seconds.
Use a new report filename for each run.

The image is converted from OpenCV BGR to MediaPipe RGB. Detection uses
unmirrored input with increasing timestamps; only the completed preview is
mirrored. No video, frames or landmark coordinates are saved. Optional reports
contain aggregate tracking counts and timing only. Hand count reflects detector
output, not independently verified anatomical correctness.

The live preview was tested for two minutes, processing 2,585 frames at about
21.4 FPS on the CPU detector. The user confirmed both hands and fingers tracked
well. One official dataset training clip also decoded all 92 frames and produced
finite 21-point hand detections. No classifier was trained by either check.

## 4. Future training: requires explicit authorization

The official pose baseline uses legacy MediaPipe Holistic and ST-GCN. Our
planned application instead uses MediaPipe landmarks with a GRU or small
Transformer, as proposed in the original project summary. This is a deliberate
adaptation of the dataset workflow, not a reproduction of Microsoft's ST-GCN
benchmark. The preview uses the current MediaPipe Tasks API.

After authorization, the intended steps are:

1. Finalize a shared hand/body landmark schema and inspect extraction quality.
2. Extract dataset landmarks, retaining official split and signer identities.
3. Define normalization, missing-landmark masks and temporal preparation.
4. Train a GRU baseline using only the training split, tune on validation, and
   evaluate on the held-out test split.
5. Save the classifier, exact label map, feature schema and preprocessing settings.
6. Add the same processing to webcam inference and evaluate new signers.

No bulk landmark extraction, classifier training or training command is included
in the current run. A 2,731-label target does not mean those signs are already
recognized. Body landmarks, sign segmentation and sentence translation remain
future work.

## 5. Planned Colab workflow

The user plans to download ASL Citizen directly into a chosen Colab notebook's
runtime. Preparation and GRU training may run there; webcam inference will stay
local. The notebook can launch ordinary Python modules rather than contain a
second implementation of the model.

See [the Colab preparation and training guide](docs/colab.md) for copy-paste
download/extraction cells, disk requirements, persistent storage, rough runtime
estimates, augmentation choices and the model export bundle.

The current MediaPipe extractor runs on the CPU. Selecting a T4 runtime does not
automatically enable GPU landmark extraction. No Colab session, GPU delegate,
bulk extraction or GRU training has been run or benchmarked by this project.
The Windows-only uv lock configuration is not a tested Colab/Linux environment.

Implementation status and actual runtime checks are in
[docs/project-decisions.md](docs/project-decisions.md). The original
[project summary](sign_language_project_summary.md) was refreshed on 2026-11-07
to reflect ASL Citizen, the planned GRU and the Colab workflow. Its earlier
reference version is retained in Git history.

Videos, dataset CSVs, derived features, model files, environments and reports are
ignored by Git. Dataset materials remain subject to Microsoft's research license.
