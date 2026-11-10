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
training loop or local sign-inference path currently exists. Raw batch
extraction was subsequently implemented as recorded below.

The current hand tracker is CPU-based. A T4 assignment alone does not accelerate
it. A GPU-compatible MediaPipe delegate would need explicit configuration and
runtime verification. The repo's uv environment constraints and torch indexes
are currently Windows-specific. The separate Colab extraction environment was
subsequently user-tested on a pilot and comparison as recorded below.

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

## Landmark extraction implementation on 2026-11-07

The user has completed the first three Colab data-preparation steps and now
requests extraction. They confirmed the dataset root
`/content/asl_citizen/ASL_Citizen` and availability of Google Drive. This authorizes
the extraction stage; it does not authorize GRU training.

Implemented a CPU raw extractor using the same MediaPipe hand tracker plus
pose-lite (33 joints) to retain hand placement relative to the body/head. Trackers
reset at every clip boundary. Native frames, timestamps, raw XYZ, handedness
category/score, pose confidence fields and absence/validity masks are preserved.
Hand arrays are detection-order observations, not stable anatomical slots;
future preprocessing must canonicalize them before GRU input. Hand and pose z
values have different origins and are not a shared metric coordinate system.

The extractor writes compressed numeric NPZ entries into split-specific ZIP
shards and commits a checksummed marker only after a verified copy. It validates
configuration, model/code/library/CSV fingerprints on resume, rejects mixed runs,
recomputes damaged/incomplete shards and records failed clips explicitly.
Default shards hold 100 clips. A local file lock prevents duplicate writers on
one host/work directory; separate VMs must not write the same Drive output.
Raw video files are assumed unchanged within a run and are not content-hashed.

Added a builder for a self-contained copy/paste setup cell and notebook. The
cell preserves the existing raw-data runtime, uses an isolated Linux extraction
venv, mounts Drive, runs a ten-clip pilot and defines an explicit full-run call.
It embeds source only, not dataset files, weights or features. It never invokes
a trainer. The first full-run cell is commented out pending pilot review.

Validation: nine project tests passed, including absent/low-confidence landmark
handling, incomplete decode rejection, duplicate-writer locking, resume,
corruption repair, explicit failures and configuration guards.
Two actual ASL Citizen training clips (220 frames) were extracted with hand and
body pose using both serial and spawned-worker paths. Both clips completed
without failure; arrays were finite with expected shapes and increasing
timestamps. Repeating the same run reused the shard with zero new decoded
frames. These are local functional checks, not tracking/recognition accuracy,
cloud performance or full-corpus validation. Subsequent user-run Colab checks:
the CPU pilot reported 4.95 wall seconds/clip with two workers; a separate
one-worker-per-delegate comparison completed ten clips/844 frames with no
failures, CPU 52.5284 seconds versus GPU 22.2775 seconds (2.36x). NVIDIA EGL
libraries loaded privately and all ten GPU clips reported NVIDIA rendering.
The tensor write error appeared in every GPU clip and none on CPU; it remains
unresolved. GPU output agreement and timing are preliminary, not tracking
accuracy or full-corpus validation. Full extraction still explicitly selects
CPU; training remains unauthorized. The follow-up 100-clip comparison completed
all pairs (9,000 frames): CPU 533.3548 seconds versus GPU 245.1075 seconds
(2.176x). All GPU clips used NVIDIA rendering and logged the tensor write error.
Hand-count agreement was 96.61%; one clip's hand p95 distance reached 0.39614
in normalized XY. A saved-feature audit cell now targets worst frames and
possible opposite-hand matching; full GPU extraction is not selected. The
compute-only extrapolation is 56.78 hours, with storage and interruption
overheads excluded.

See [landmark-extraction.md](landmark-extraction.md) for schema, pilot review,
resume, failures and current limitations. Canonicalization, GRU implementation,
augmentation, training, evaluation and local classifier export still need work.

## Local CPU benchmark on 2026-11-07

At the user's request, benchmarked the local i5-9400F (six logical CPUs) using
the same 100 seeded training clips/9,257 frames with hands and pose. Fresh
one-/two-/four-worker runs took 314.28/161.35/103.89 wall seconds, including
local shard commits and process startup. Every run succeeded, with no tensor
write errors. All 100 feature NPZs were byte-identical across worker settings.
Two-worker overall CPU utilization averaged about 37% in a 30-second sample,
which justified the additional four-worker run; its sample averaged about 75%.
Four workers were fastest tested, roughly 24.1 active hours extrapolated across
the corpus. Added a reusable bounded `scripts/benchmark_landmarks.py` runner.
Full local extraction/background setup and classifier training were not started.
See [the detailed report](local-cpu-benchmark-2026-11-07.md) for scope, raw-report
locations and limits. Use separate output identities for Windows and Colab.

## Full local extraction started on 2026-11-07

The user explicitly requested starting extraction with two workers. Launched
`scripts/local_landmark_job.py run --workers 2` in a hidden Windows background
process at 19:04 Asia/Tashkent. The child CLI uses CPU hands plus pose, all
train/val/test clips, no sample limit, 100-clip shards and a separate full-run
output folder. Its manifest lists 83,399 clips/836 shards, run ID
`f55c6cdb4e7e711d7d850cd8c3cbd0776e91ac5a873f98dc2476257b32659cf2`.
Verified a live heartbeat, advancing video progress and the first committed
100-clip shard with zero failures. Full completion is pending.

The supervisor provides `status`/`stop` commands, per-attempt stdout/native
logs, single-instance locking and a scoped Windows idle-sleep prevention
request. It changes no power-plan settings and allows the display to sleep.
Stopping terminates its child process tree; committed batches remain resumable.
No classifier training is started. See [local-extraction.md](local-extraction.md).

## GRU sequence limit agreed on 2026-11-09

The user selected a **128-frame maximum** for the initial GRU recognizer,
following the limit in the official
[ASL Citizen ST-GCN loader](https://github.com/microsoft/ASL-citizen-code/blob/main/ST-GCN/asl_citizen_dataset_pose.py).
This adopts the sequence limit, not the ST-GCN architecture or all of its
preprocessing choices.

Longer isolated signs will be uniformly sampled to 128 observations across
their full duration, preserving order and endpoints. Shorter signs retain
their frames and true sequence length; batch padding must be excluded from
GRU state updates using packed sequences. Per-landmark detection masks remain
separate from temporal padding. The same policy must be used for captured
webcam signs and recorded in the exported preprocessing configuration.

The limit is configurable and still requires implementation and validation.
It does not choose a fixed capture duration, solve sign segmentation or
authorize training. Existing raw landmarks remain usable without re-extraction.
Earlier training-time estimates with roughly 64 frames must be replaced by a
benchmark of the actual selected configuration.

## Local GPU training authorized on 2026-11-09

The user requested trying training on the local GPU after extraction completed.
Implemented the ASL normalization/hand-slot adapter, resumable prepared array
cache and two-layer 256-unit GRU with a 2,731-class linear head. The first run
uses the agreed 128-frame maximum, official signer-disjoint splits, batch 64,
AdamW and cross-entropy, up to 50 epochs with validation-loss early stopping.
The initial run uses no augmentation and does not evaluate the held-out test set.

Installed the existing locked CUDA extra and verified a CUDA forward/backward
operation on the RTX 4060 with PyTorch 2.11.0+cu128. Raw extraction archives are
retained unchanged. Separate smoke and baseline outputs prevent diagnostic
training from seeding the full run. Atomic epoch checkpoints, source/config
guards, status/stop controls and a hidden sleep-preventing supervisor support
interruption and resume. See [local-training.md](local-training.md) for details.
This authorization supersedes older setup-only restrictions for ASL training;
it does not claim that RSL training or webcam sign recognition is implemented.

## Regularized signer-grouped CV authorized on 2026-11-09

After the baseline's validation plateau and training/validation gap, the user
requested cross-validation training and regularization. Selected three
StratifiedGroupKFold folds within official training clips only, grouping by
signer and validating disjoint subjects, complete held-out coverage and class
coverage. The candidate increases dropout from 0.3 to 0.5 and weight decay from
0.0001 to 0.001, adding small training-only scale/roll perturbations that keep
masks, padding, temporal order and local depth intact. CV is a development
robustness check, not itself an overfitting remedy.

The background study trains three independent fold models, then a fresh
regularized model on the official training/validation split for a comparable
baseline evaluation. All use up to 50 epochs, patience 10 and epoch-level
resume. The test set, original model and feature cache remain untouched.
All 37 checks and a regularized CUDA smoke run passed before launch.
See [cross-validation.md](cross-validation.md) for protocol, timing and controls.
