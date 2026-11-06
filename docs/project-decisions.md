# Project decisions and implementation status

## Decision on 2026-11-06

The user requested removal of the WLASL/I3D implementation, adoption of ASL Citizen,
and a MediaPipe webcam preview showing hand skeletons. Classifier training is
explicitly deferred until the user authorizes it. No training or bulk landmark
extraction is to be started as part of this setup.

MediaPipe is a pretrained landmark extractor. The later trainable recognizer is
a GRU or Transformer using landmark sequences, consistent with the original
project design. A hand skeleton is not evidence of sign recognition accuracy.

## Removed and retained

Removed the WLASL source/data directory, official I3D weights and downloaded
weight archives, I3D adapter, reference patch, downloader preparation script,
recognition command and I3D-specific tests. Existing Git history remains. The
reference summary was preserved during the resets, then updated on 2026-11-07
under the user's request to refresh all related documents. Old diagnostic
JSON/logs under ignored `.cache` are
historical records, not executable recognition code.

The summary's preserved pre-refresh version had SHA-256
`79d3867ec1f5f662e9200a50dddca28fd766b4606e1ceb674fda65a666984c95`.
That is a historical fingerprint, not the hash of the refreshed summary.

## Official workflow and deliberate adaptation

Sources consulted: Microsoft ASL Citizen project page, datasheet, dataset license,
repository README, ST-GCN README and pose.py. The official pose workflow is:
download ZIP, unzip, extract MediaPipe Holistic landmarks, train/test ST-GCN.
Only dataset setup is being followed now. The planned GRU/Transformer is our own
classifier choice; no official ST-GCN or I3D accuracy is claimed for this project.

The official legacy environment is not copied into the project: Python 3.12 and
the current MediaPipe Tasks hand API are used for the interactive preview.
Pose/body and face feature requirements will be settled before classifier work.

## Data preparation status

The complete official ZIP was downloaded from the Microsoft URL and dataset
preparation finished on 2026-11-07.
It contains 83,406 ZIP entries totaling 49,604,368,459 extracted bytes; archive size
is 45,924,134,223 bytes. The official split CSV entries were fetched with HTTP
ranges and CRC-checked while the video archive downloads. Metadata audit passed:

| Split | Videos | Labels | Participants |
| --- | ---: | ---: | ---: |
| Train | 40,154 | 2,731 | 35 |
| Validation | 10,304 | 2,731 | 6 |
| Test | 32,941 | 2,731 | 11 |

The participants are disjoint between splits. The first audit was metadata-only;
the final extracted-file audit also passed for all 83,399 videos. Every ZIP file
entry was read with CRC verification. This does not establish that all videos
decode correctly. The exact training-label mapping is saved locally in
`data/metadata/label-map.json`.

Full setup report: `data/asl_citizen/setup-report.json`. Extracted dataset root:
`data/asl_citizen/ASL_Citizen/`. Archive SHA-256:
`349c0e33f13b876f19ad4dee72205f4d2eae02937613e275200aa92283268eab`.
This is the locally calculated archive fingerprint, not an independently
published Microsoft SHA-256. The source URL, byte count and server ETag are saved
in `data/downloads/ASL_Citizen.source.json`.

All five earlier diagnostic labels occur in each split, using exact dataset
glosses: HELLO (14/4/14 train/val/test), THANKYOU (16/3/12), YES (15/5/10),
NO (16/3/11) and ME (13/5/13). Their presence does not imply recognition has been
trained or validated. No class subset has been selected for training.

## Preview and validation status

Implemented `HandTracker` with MediaPipe Tasks VIDEO mode, two hands, RGB input,
strictly increasing timestamps and a mirrored skeleton preview. Reports contain
aggregate hand counts and timing, without webcam images or landmark coordinates.
No sign prediction or training code is invoked by the preview.

The single `.venv` was synced to MediaPipe 1.2.2, OpenCV-contrib 4.11.0 and NumPy
1.26.4. Torch, torchvision and the old video-download dependencies were removed
from the installed environment; torch extras remain optional for later work.
The existing single environment's activation prompt was changed from the old
baseline name to `sign-language-recognition`; no second environment was created.
`scripts/check.ps1` passed: compilation, lockfile and four tests covering resumed
downloads/corrupt prefixes, path containment, skeleton rendering and camera-read
failure cleanup.

The user confirmed that both hands and fingers track well in the live preview.
The detector ran on the TensorFlow Lite CPU delegate. The 120.59-second preview
processed 2,585 frames (about 21.4 FPS): 1,827 with zero hands reported, 198 with
one, and 560 with two. These are aggregate detector counts, not recall or accuracy
measurements; the user was not instructed to keep hands visible for every frame.
Mean detector time was 17.56 ms. The camera closed normally at the time limit.
No webcam media or coordinates were saved and no classifier was trained.
Local report: `.cache/hands-preview-20261106-01.json`.

A CRC-checked official ASL Citizen training clip (`0022932153577568393-HELLO.mp4`)
was also fully decoded: 92/92 frames. MediaPipe reported hands in 48 frames, with
21 finite XYZ landmarks for every reported hand. This is one extraction smoke
check, not a tracking-accuracy measurement or recognition result. No feature
arrays were saved. Report: `.cache/asl-citizen-hand-smoke.json`.

The full download, CRC-checked extraction and file-presence/split audit have
completed successfully. Full-corpus video decoding, bulk landmark extraction,
classifier training and recognition evaluation have not been performed. Training
still requires the user's explicit authorization.

## Colab and classifier plan agreed on 2026-11-07

The user will perform the documented dataset download/extraction in a chosen
Colab notebook. The dataset belongs to that notebook's connected runtime, not to
the notebook file. A cloud preparation/training workflow is planned; no remote
runtime has been accessed or its execution verified by this project.

GRU is the planned first sign classifier; a small Transformer remains a possible
comparison. MediaPipe supplies landmarks and is not being retrained to recognize
sign meanings. OpenCV supplies camera frames. A future GRU would score the full
2,731-label vocabulary if trained on all dataset classes. No trained classifier,
bulk feature extractor, training loop or local sign-inference path currently
exists. Existing CLI commands cover data preparation and hand visualization.

The current hand tracker is CPU-based. A T4 assignment alone does not accelerate
it. A GPU-compatible MediaPipe delegate would need explicit configuration and
runtime verification. The repo's uv environment constraints and torch indexes
are currently Windows-specific; no Colab/Linux dependency environment is tested.

Runtime figures discussed are planning estimates, not Colab measurements:

- Hands-only extraction for the whole corpus: roughly 1–3 days as an initial
  budget, possibly longer on the allocated CPU or with body/face tracking.
  An assumed 90 frames per clip gives about 7.5 million frames; at the measured
  local detector-only 17.56 ms/frame, that is about 37 hours before other work.
  The 90-frame corpus average is an assumption, not a measured dataset property.
- Small GRU on a T4 with cached landmarks: roughly 1–6 hours for 50–100 epochs,
  assuming two layers, 128–256 hidden units, about 64 frames per sample and batch
  size 64–128. This is not a benchmark or a promised Colab session duration.
- Timing a representative extraction sample and several training/validation
  epochs after authorization is required to replace those budgets with forecasts.

Conservative augmentation is planned for training only: modest consistent
geometric transforms, timing variation and carefully bounded noise/masking.
Preserve sign motion and hand shape; validate horizontal flips and left/right
slot handling before using them. Validation/test preparation stays deterministic
and signer-disjoint. On-the-fly feature augmentation must not rerun MediaPipe on
every epoch or manufacture train/test overlap.

Colab processing must save completed feature batches and resumable training
checkpoints to private persistent storage. Use the runtime's local disk for
active data reads, check disk capacity and plan for session termination. The
local export bundle must include trained weights, model configuration, exact
label mapping, landmark/normalization/sequence settings, extractor versions and
evaluation metadata. Optimizer/scheduler state and random states are needed for
resuming training, rather than for local inference alone.

The detailed data-only cells and future-work checklist are in [colab.md](colab.md).
Updating these documents or pushing the repository does not authorize training.
