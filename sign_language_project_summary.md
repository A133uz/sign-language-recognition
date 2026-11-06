# Real-Time Sign Language Recognition Using Computer Vision and Deep Learning

Project summary updated 2026-11-07. This document describes the current direction,
implemented components and remaining work. The pre-reset reference version was
preserved during implementation resets and remains available in Git history.
The current request to refresh all related documents supersedes keeping that
old reference text unchanged.

## 1. Objective and scope

Develop an application that recognizes individual American Sign Language (ASL)
signs from a laptop camera and displays their textual meanings. The minor project
covers isolated signs. Continuous signing, sentence generation and paragraph
context are later major-project research goals, not implemented functionality.

ASL is the primary language. The supervisor accepts ASL-primary/RSL-secondary
scope. A Russian Sign Language experiment would require its own labeled dataset
and evaluation; ASL Citizen contains ASL, not RSL. The application and university
report take priority; publication is optional. Project code uses ordinary Python
modules, with a Colab notebook acting as a possible launcher.

## 2. Current implementation and model responsibilities

| Component | Role | Current state |
| --- | --- | --- |
| OpenCV | Capture webcam frames and decode videos | Implemented |
| MediaPipe Tasks Hand Landmarker | Locate up to two hands and their finger joints | Implemented; local CPU preview tested |
| Dataset utilities | Acquire ASL Citizen, unzip, audit official splits and build label mapping | Implemented; local setup completed |
| GRU | Learn sign classes from landmark sequences | Planned baseline; not implemented or trained |
| Small Transformer | Possible classifier comparison | Planned option |
| Local sign output/history | Display recognized meanings using a trained classifier | Not implemented |
| Segmentation and language generation | Turn continuous signing into sentences | Future research |

MediaPipe does not currently classify sign meanings. It supplies hand landmarks.
The future GRU will analyze hand shape, relative positions and motion over time;
its classification layer will score sign classes. A saved class-ID/gloss mapping
will convert the selected output index into text. Training MediaPipe itself is
not the planned sign-classification step.

Implemented preview:

```text
Webcam -> OpenCV frames -> MediaPipe hand landmarks -> hand skeleton display
```

Planned isolated-sign application:

```text
Webcam -> shared hand/body landmark extraction -> normalization and sequence
preparation -> trained GRU -> sign label -> local word display/history
```

The shared hand/body schema, sequence preparation, missing-landmark masks and
recognition/rejection rules still need implementation and validation. Current
tracking uses hands only. The WLASL/I3D implementation, local clips and weights
were removed at the user's request. Earlier SVM, I3D and other experiments are
historical work, not active fallback classifiers.

## 3. Dataset: ASL Citizen

The current dataset is [Microsoft ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/):
83,399 videos covering 2,731 labeled signs from 52 participants. The full
vocabulary is the eventual training target; label presence does not establish
recognition accuracy.

| Official split | Videos | Participants |
| --- | ---: | ---: |
| Train | 40,154 | 35 |
| Validation | 10,304 | 6 |
| Test | 32,941 | 11 |

Keep the official signer-disjoint splits. Train on training examples, select
hyperparameters on validation, then evaluate the chosen model on held-out test
examples. Do not randomly mix videos from the same signer across splits.

The full archive and extracted files were prepared locally on 2026-11-07.
ZIP CRC verification and the file-presence/split audit passed for all declared
videos. Full-corpus decoding and bulk landmark extraction have not been run.
One training clip was fully decoded as a MediaPipe extraction smoke check.

Archive size: 45,924,134,223 bytes (42.77 GiB). Extracted entries total
49,604,368,459 bytes; reserve about 100 GB for both plus working space.
Data, models and feature arrays remain outside Git and subject to the
[dataset research license](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/).

The official pose-based baseline uses legacy MediaPipe Holistic and ST-GCN.
Our planned GRU/Transformer workflow is an adaptation of the dataset preparation
approach, not a reproduction of that benchmark. WLASL remains related work only.

## 4. Planned Colab preparation and training

The user plans to download the dataset directly from Microsoft inside a chosen
Colab notebook. Files belong to its connected VM, not the notebook file. The
copy-paste data-only steps, disk checks and persistence guidance are in
[docs/colab.md](docs/colab.md). No Colab session has been accessed or benchmarked
by this project, and the user has not reported completing that download yet.

After authorization:

1. Finalize and quality-check the shared landmark schema and preprocessing.
2. Benchmark representative video extraction in the actual Colab runtime.
3. Extract and save resumable landmark batches with split/signer membership,
   failure records, masks and extractor settings.
4. Train a small GRU on cached sequences; benchmark actual epoch time.
5. Compare conservative training-only augmentation using validation results.
6. Evaluate the final chosen model on the official held-out test set.
7. Export weights, architecture settings, exact labels, preprocessing schema and
   version/evaluation metadata; implement and test the local load/inference path.

Completed feature batches and resumable training checkpoints must survive VM
termination using private persistent storage. Use runtime-local disk for active
processing. The Windows uv configuration is not a tested Colab/Linux dependency
environment; cloud extraction/training setup still needs to be implemented.

Training remains on hold until explicitly authorized. Bulk landmark extraction,
GRU training and local word recognition commands do not exist yet.

## 5. Resources and runtime estimates

Local hardware includes an RTX 4060 with 8 GB VRAM. The tested hand tracker uses
the CPU. A two-minute live preview processed 2,585 frames at about 21.4 FPS;
the user confirmed that both hands and fingers tracked well. These observations
validate the preview, not sign-recognition accuracy.

Colab T4 training is a planned option. The current extractor does not request a
GPU delegate, so selecting a T4 does not automatically speed up MediaPipe.

- Whole-corpus hands-only extraction: roughly 1–3 days as an initial planning
  budget, potentially longer depending on the CPU, I/O and added body/face work.
- Small GRU on a T4 with cached landmarks: roughly 1–6 hours for 50–100 epochs,
  assuming two layers, 128–256 hidden units, about 64 frames per sample and batch
  size 64–128. No T4 training benchmark has been run.
- These are estimates, not commitments. A representative extraction sample and
  several training/validation epochs are needed for reliable forecasts.

The local detector averaged 17.56 ms/frame before decoding/saving overhead.
Assuming 90 frames per video would imply about 37 hours of detector processing
for the full corpus at that local rate. The 90-frame average is hypothetical.
Extract once and reuse cached landmarks across training experiments whenever
the extraction schema and settings stay compatible.

## 6. Augmentation and evaluation plan

Compare a baseline with modest geometric changes, timing variation and bounded
landmark noise/masking. Preserve sign-defining motion, hand shape and relative
positions. Validate mirroring and left/right slot handling before using it.
Avoid reversed motion, aggressive finger distortions and incomplete-sign crops.
Augment training samples on the fly; keep validation/test inputs deterministic
and preserve signer separation. MediaPipe does not need to rerun each epoch.

Report vocabulary size alongside top-one/top-five accuracy, per-class results
and confusion matrices. Measure latency/FPS on named hardware and test new
signers under varied framing, backgrounds and lighting. Check idle/unknown-sign
behavior before describing the application as a reliable live recognizer.
No current sign-recognition accuracy or trained-model GPU throughput is claimed.

## 7. Later major-project direction

Continuous recognition needs sign-boundary handling or an appropriate sequence
recognizer and continuous-signing data. ASL gloss sequences cannot simply be
joined word by word to guarantee grammatical English. Sentence generation and
context handling require separate implementation, data and evaluation. A limited
vocabulary demonstration is a possible research stage; open-domain paragraph
translation is not an implemented or promised result.

## 8. Report and working records

The university report can cover the problem, related work, dataset/splits,
architecture, preprocessing, training protocol, quantitative results, live tests,
limitations and future work. Keep planned features separate from measured
results and record model/configuration changes between experiments.

- [README.md](README.md): local setup and commands.
- [docs/project-decisions.md](docs/project-decisions.md): decisions, actual checks
  and implementation status.
- [docs/colab.md](docs/colab.md): cloud preparation guide, resource assumptions,
  training plan and export requirements.
- [Official ASL Citizen code](https://github.com/microsoft/ASL-citizen-code):
  authors' model-specific preparation and baseline references.
- [Official datasheet](https://www.microsoft.com/en-us/research/project/asl-citizen/datasheet/):
  dataset composition and collection/split information.
