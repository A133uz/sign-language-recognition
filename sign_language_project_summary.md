# Real-Time Sign Language Recognition Using Computer Vision and Deep Learning

Updated 2026-11-07. This summary records the agreed design, verified work and
remaining decisions. Historical implementations remain in Git history; they
are not active fallback recognizers.

## 1. Objective and scope

Develop a webcam application that recognizes an individual sign and displays
its corresponding word/gloss on screen, supporting **American Sign Language
(ASL)** and **Russian Sign Language (RSL)**. ASL is the primary implementation;
RSL is the planned extension. The supervisor is aware of this language split.
The application and university report take priority; publication is optional.

The minor project covers isolated signs. A single sign can span many video
frames, so temporal modeling is needed even when the output is one word.
Continuous signing, sentence generation and paragraph context remain later
major-project research goals, not implemented or promised functionality.

Use ordinary Python modules. Google Colab is an optional training environment;
the webcam application and the current ASL landmark extraction run locally.

## 2. Agreed datasets

| Language | Dataset | Feature preparation | Status |
| --- | --- | --- | --- |
| ASL | [Microsoft ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) | Run pretrained MediaPipe over the videos and cache hands/body landmarks | Raw extraction complete; first local GRU training authorized and implemented |
| RSL | [Slovo](https://github.com/hukenovs/slovo) | Start with the authors' supplied MediaPipe hand landmarks and label/split metadata | Selected; download, schema audit and integration pending |

### ASL: Microsoft ASL Citizen

Our verified download contains **83,399 videos, 2,731 labeled signs and
52 participants**. The full vocabulary is the eventual training target;
presence of a label is not evidence that the system already recognizes it.

| Official split | Videos | Participants |
| --- | ---: | ---: |
| Train | 40,154 | 35 |
| Validation | 10,304 | 6 |
| Test | 32,941 | 11 |

Keep these signer-disjoint splits. Train on training examples, select settings
using validation, then evaluate the chosen model on the held-out test split.
Do not mix the same signer across splits or tune using test results.

The complete archive was downloaded and CRC-checked during local extraction;
file-presence, vocabulary and signer-split audits passed. Archive size is
45,924,134,223 bytes (42.77 GiB), with 49,604,368,459 extracted bytes. Dataset
materials and derived features stay outside Git and remain subject to the
[dataset license](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/).
Full-corpus raw extraction is complete: the saved summary reports 83,399
successful clips, zero failed clip records and 836 shards. Feature archives
occupy 5,372,618,913 bytes (about 5.00 GiB). Training-cache preparation verifies
each archive's checksum and record mapping before use.

**The official ASL Citizen release does not supply a downloadable cache of
pre-extracted MediaPipe landmarks like Slovo does.** It supplies extraction
code and pretrained I3D/ST-GCN recognition weights. For our selected GRU route,
we therefore generate the training landmark sequences ourselves. See the
[official pose workflow](https://github.com/microsoft/ASL-citizen-code/tree/main/ST-GCN)
and [checkpoint release](https://github.com/microsoft/ASL-citizen-code/releases/tag/checkpoints_v1).

### RSL: Slovo

The authors report **20,400 videos**, **1,000 gesture classes** and **194 signers**,
with 15,300 training and 5,100 test examples split by subject. They list a
roughly **1.2 GB JSON download of pre-extracted MediaPipe hand landmarks** for
the trimmed videos. The trimmed-video archive is about 16 GB; using the supplied
landmarks need not require extracting every raw video again.
Sources: [dataset repository and downloads](https://github.com/hukenovs/slovo#downloads),
[landmark JSON](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo_mediapipe.json),
[dataset paper](https://arxiv.org/abs/2305.14527).

Planned preparation is to obtain the landmarks and annotations, align video IDs
with labels and splits, and inspect coordinate dimensions, hand identities,
missing detections, frame timing and detector/preprocessing compatibility.
Verify the actual label set, including any background/no-sign label, before
setting the RSL classifier's output size. Create a signer-separated validation
subset from the training pool; reserve the official test split for final use.

The supplied annotations are **hand landmarks**, whereas our ASL cache includes
**hands and body pose**. The initial RSL model will use a hand-based input schema
unless additional pose features are explicitly added later. Do not assume the
provided JSON matches our ASL NPZ schema or fill absent body features into an
ASL-trained model and call that an RSL recognizer. The RSL webcam preprocessing
must reproduce the RSL training representation. Supplied landmarks require
compatibility validation, but do not need to be regenerated if they are usable.

## 3. Agreed application pipeline

The selected baseline is **MediaPipe landmarks -> sequence preprocessing ->
GRU -> linear classification layer -> sign label -> displayed text**.

```text
Webcam -> OpenCV frames -> pretrained MediaPipe landmark detection
                                  |-> hand-skeleton preview
                                  |
                         User selects ASL or RSL
                           /                 \
              ASL preprocessing          RSL preprocessing
              (hands + body)             (supplied hand schema)
                       |                       |
                   ASL GRU                 RSL GRU
                       |                       |
                ASL linear head         RSL linear head
                       |                       |
                ASL label mapping       RSL label mapping
                           \                 /
                     Predicted word/gloss on screen
```

The first version uses **separate ASL and RSL recognizers, weights, label maps
and recorded preprocessing configurations**, selected explicitly in the UI.
Automatic language identification and a shared multilingual encoder are not
part of the agreed baseline. The general code structure and compatible detector
components can be shared; the two languages need not have identical input or
output dimensions. Translating an ASL output label into Russian is not RSL
recognition.

### Responsibilities

- **OpenCV** captures frames and decodes videos.
- **MediaPipe** uses pretrained vision models to locate/track hands and body
  joints. It is not the sign-meaning classifier, and we are not retraining it.
- **Preprocessing** establishes stable left/right hand identities, normalizes
  coordinates, handles missing points with masks and constructs a bounded
  sequence containing an isolated sign. Temporal resampling/padding is now
  implemented for the ASL trainer; webcam capture/windowing still needs work.
- **GRU**, a type of RNN, learns hand-shape, position and motion patterns across
  the frames of one sign. It is not reserved for sentence recognition.
- **A trainable linear layer** maps the GRU representation to one raw score
  per label. The selected class ID is converted to text using that model's map.

The proposed starting head is PyTorch `nn.Linear(gru_representation_size,
num_classes)`. For a 256-dimensional GRU representation, an ASL example is
`nn.Linear(256, 2731)`. The RSL output size comes from its audited label map.
The first local ASL run selects a two-layer, 256-hidden-unit GRU and this
linear head. Alternative hyperparameters still require validation experiments.
The input size must match the actual GRU representation.

Train the GRU and linear layer jointly with `CrossEntropyLoss` on raw class
scores; do not add softmax before that loss. At inference, choose a class from
the scores; softmax may be used for normalized scores when useful. Unknown-sign,
idle and acceptance-threshold behavior still needs a separate design and test.
A closed-set output is not a guarantee that any arbitrary motion is recognized.

SVM, ST-GCN, temporal CNNs, Transformers and pretrained video models were
discussed as alternatives. None replaces the selected GRU + linear baseline
at this stage. A pretrained ST-GCN would require its exact keypoint graph and
preprocessing; pretrained I3D consumes video pixels instead of our landmark
cache. Neither should be described as a drop-in trained GRU.

## 4. Offline preparation, training and evaluation

```text
ASL Citizen videos -> pretrained MediaPipe inference -> cached hand/body sequences
Slovo supplied hand landmarks + annotations -> validated data-format adapter
                                        |
                         Language-specific preprocessing
                                        |
                     Train GRU + linear head on training data
                                        |
                     Tune on validation; final held-out test
                                        |
               Export weights + architecture + labels + preprocessing
                                        |
                         Load in the webcam application
```

Training uses cached landmarks, so MediaPipe does not run again every epoch.
The offline and webcam paths must share the same feature preparation for each
language. Keep official splits, signer IDs, class mappings, missing-point masks
and feature provenance throughout training and export.

### Agreed sequence limit: 128 frames (2026-11-09)

Use **at most 128 time steps per isolated sign** for the initial GRU baseline.
This follows the sequence-length limit of the authors'
[ASL Citizen ST-GCN data loader](https://github.com/microsoft/ASL-citizen-code/blob/main/ST-GCN/asl_citizen_dataset_pose.py),
while retaining our own GRU architecture and preprocessing.

- For signs longer than 128 frames, uniformly sample 128 observations across
  the entire sign in chronological order, including its beginning and end.
  Do not keep only the first or latest 128 frames.
- For shorter signs, retain all original time steps and right-pad for batching.
  Keep true sequence lengths and use packed GRU sequences so padding does not
  update the recurrent state or become the representation used for classification.
- Keep landmark-detection masks separate from temporal padding: a real frame
  with missing hands is different from a nonexistent padded frame.
- Export the 128-frame limit, sampling rule and padding policy with the model;
  apply the same preparation to each captured webcam sign. This is an observation
  limit, not a fixed capture duration or a sign-boundary detector.

These settings are implemented in the ASL preprocessor and trainer. The raw cache
retains all decoded frames, so no landmark re-extraction is required. The
selected normalization needs recognition evaluation; webcam capture/window
boundaries remain unimplemented. Training time must be benchmarked at this length;
earlier estimates assuming about 64 frames do not establish a 128-frame budget.

Compare a baseline with conservative training-only augmentation, such as modest
geometric changes, timing variation and bounded masking/noise. Preserve
sign-defining motion and finger shape. Mirroring, reversed movement and
incomplete-sign crops must not be assumed to preserve labels. Fit any learned
normalization on training data only; keep validation/test preparation deterministic.

GRU training is planned on the local RTX 4060 (8 GB) or a Colab GPU such as T4,
subject to a verified PyTorch/CUDA setup. CPU DataLoader workers can prepare
batches while one GPU training process runs. Benchmark worker counts such as
0, 2 and 4; extraction's multiprocessing speedup does not predict training's.

**Local GPU training was authorized on 2026-11-09.** The ASL preprocessor,
prepared cache, GRU trainer and resumable background launcher are implemented;
see [the local training guide](docs/local-training.md) for exact settings and
live status commands. No trained accuracy is claimed by implementation alone.
Measure warm-up and several full epochs,
including validation, before promising a duration. Earlier hour-scale estimates
were unmeasured whole-run planning budgets for roughly 50-100 epochs, not
hours per epoch. Current epoch times and validation metrics are measured in
the local run's JSON reports. No held-out test or webcam accuracy is established.

## 5. Current implementation and running work

| Component | Current state |
| --- | --- |
| ASL Citizen download, extraction and split audit | Completed locally |
| Live MediaPipe hand-skeleton preview | Implemented; user confirmed both hands/fingers track well |
| Offline raw hands/body extraction | Completed: 83,399 clips, 836 shards, zero failed clip records |
| Slovo landmark/annotation acquisition and audit | Planned; not performed |
| Stable hand-slot canonicalization and model preprocessing | ASL baseline implemented; heuristic hand assignment and normalization require recognition evaluation |
| GRU + linear head, trainer and checkpoint export | First ASL run completed: 33 epochs; selected epoch 23; validation top-1 55.73%, top-5 82.50%; test not evaluated |
| ASL/RSL language selector and trained webcam recognition | Planned; not implemented |
| Continuous recognition and sentence generation | Later research |

Current raw ASL features retain both hands (21 joints each), 33 pose joints,
XYZ coordinates, native-frame timestamps, reported handedness and missing/
confidence masks. Hand slots are in **detection order**, not fixed anatomical
left/right order. Hand and pose depth use different origins; they are not a
shared metric 3D coordinate system. No dense face mesh, normalization, temporal
resampling, padding or augmentation is included in the raw extraction stage.
The current live preview draws hands only; the offline extractor also runs pose-lite.

The first corrected local GPU training run completed on 2026-11-09 in about
58 minutes, with early stopping after epoch 33. The lowest-validation-loss
checkpoint is epoch 23: validation top-1 55.73%, top-5 82.50%, macro F1 54.67%.
Training top-1 at that epoch was 90.77%, showing a substantial generalization
gap. These are not held-out test or webcam results. See
[the results and learning curves](docs/training-results-2026-11-09.md).

A three-fold signer-separated regularized follow-up was authorized and started
on 2026-11-09. CV uses only the official training split; a subsequent fresh
official-split run provides a comparable validation result. The candidate uses
dropout 0.5, AdamW weight decay 0.001 and small training-only geometric
perturbations. Baseline weights and cached features are preserved. Improvement
is not assumed; the test set remains unused. See
[the CV protocol and controls](docs/cross-validation.md).

On 2026-11-10 the user selected a separate **two-layer, 64-hidden-unit** GRU
capacity comparison, queued after the CV study completes successfully. It uses
263,915 parameters and the original baseline's settings (dropout 0.3, weight
decay 0.0001, no augmentation, identical official splits and feature cache),
changing hidden size only. The ongoing 256-unit CV engine was frozen to
preserve its source/configuration identity. See
[the small-model comparison and queue controls](docs/small-gru-comparison.md).

Both follow-up experiments completed successfully on 2026-11-10. On the same
official validation split, the selected regularized 256-unit model reached
**61.06% top-1**, **85.90% top-5**, and **60.14% macro F1**, improving top-1 by
5.34 percentage points over the original baseline. The selected 64-unit model
reached 45.89% validation top-1 (47.82% training top-1) after a 50-epoch budget,
suggesting underfitting or slower convergence. The three-fold regularized CV
mean was 38.29% +/- 2.42 percentage points top-1 on different signer partitions;
it is not directly comparable to the official-validation score. See
[the completed comparison and curves](docs/model-comparison-results-2026-11-10.md).

The validation-selected regularized epoch-37 checkpoint was subsequently
evaluated on all **32,941 official test clips / 11 unseen signers** on
2026-11-10: **49.92% top-1**, **76.24% top-5**, **49.47% macro F1** and test
loss **2.44304**. No training, augmentation or weight updates occurred; the
checkpoint hash is unchanged. Original and 64-unit models have not been tested.
See [the held-out test report](docs/test-evaluation-2026-11-10.md).

Full local extraction was explicitly authorized with **two CPU workers** and
started on 2026-11-07 at 19:04 Asia/Tashkent. Its manifest covers all 83,399 clips
in 836 resumable batches, with output under
`data/landmarks/asl-citizen-raw-v1-cpu-hands-pose/`. A hidden supervisor writes
logs and holds a temporary Windows automatic-idle-sleep prevention request;
no permanent power-plan setting is changed. Classifier training never starts
automatically after extraction.

Verified snapshot on 2026-11-07 at approximately **21:57 Asia/Tashkent**:
**6,500 clips committed, 65/836 shards, zero failed clips**, and the supervisor
was running. This is a dated observation, not a live progress counter. Use
`scripts/local_landmark_job.py status` through the project venv for current
progress. See [background job controls](docs/local-extraction.md) for status,
stop/resume and log locations. Preserve committed shards and the extraction
configuration when resuming; use separate output folders for incompatible
Windows/Colab or feature-schema runs.

## 6. Hardware and measured performance

Local hardware: **Intel Core i5-9400F**, six logical CPUs, and **RTX 4060 with
8 GB VRAM**. Current landmark extraction uses the CPU.

A local benchmark processed the same 100 complete training clips (9,257 frames)
with hands and pose, including model initialization, decoding, compression and
verified local shard writes:

| Workers | Average seconds/clip | Extrapolation across 83,399 clips |
| --- | ---: | ---: |
| 1 | 3.143 | 72.8 active hours |
| 2 | 1.613 | 37.4 active hours |
| 4 | 1.039 | 24.1 active hours |

All configurations completed without failures or tensor-write errors, and all
100 feature NPZs matched exactly across worker counts. Four was fastest among
the tested settings; the user chose **two** for the actual full run. These are
sample-based forecasts, not completed full-dataset durations or sign-recognition
accuracy. See [the detailed CPU benchmark](docs/local-cpu-benchmark-2026-11-07.md).

A user-run Colab CPU/GPU comparison on 100 clips/9,000 frames found a 2.18x GPU
extraction speedup after an experimental private NVIDIA EGL setup. All GPU clips
logged a tensor-write error, and a large hand-coordinate disagreement remained
unresolved. That candidate was not promoted into the full extractor. See
[the extraction record](docs/landmark-extraction.md) for details. This MediaPipe
GPU issue does not establish the performance or compatibility of future PyTorch
GRU training. No GRU GPU throughput or recognition result is claimed.

## 7. Performance metrics to demonstrate — discussion pending

Use standard metrics appropriate to **isolated sign classification** and a
webcam application. **The final metric list, averaging methods, protocols,
thresholds and target values have not been agreed and must be discussed
separately.** The following are candidates, not commitments or measured results:

| Area | Candidate measurements for discussion |
| --- | --- |
| Recognition quality | Top-1 accuracy; possibly top-k accuracy; precision, recall and F1 with suitable macro/weighted averaging |
| Error analysis | Per-class results and confusion matrices |
| Live performance | Inference latency, end-to-end response time and processed FPS on named hardware; distinguish inference time from sign-capture/window delay |
| Robustness | Held-out signers, lighting/background/framing changes, and idle or untrained-sign behavior under a defined protocol |
| Resource/training cost | Time per epoch and total training time; model size and CPU/GPU memory/use where relevant |

Report ASL and RSL separately, including vocabulary size, dataset split, number
of evaluated examples, input schema, model configuration and hardware. Landmark
counts, finite arrays and extraction FPS do not establish sign-recognition
accuracy. Do not claim full-vocabulary or reliable webcam recognition before
appropriate evaluation.

## 8. Remaining work and later direction

1. Complete and quality-check ASL landmark extraction, including missing-hand
   cases and the intended left/right canonicalization.
2. Obtain and audit Slovo's supplied hand landmarks and annotations; define its
   compatible live-input schema and signer-separated validation split.
3. Finalize and implement each language's preprocessing, isolated-sign capture
   policy, GRU + linear classifier, training configuration and export format.
4. Discuss and agree the demonstration metrics and evaluation protocol.
5. After explicit training authorization, benchmark epochs, train/tune the
   models and perform final held-out evaluation.
6. Implement the language selector and webcam prediction/text display, then
   evaluate latency and new signers using the agreed protocol.

Continuous recognition later requires sign-boundary handling or a continuous
sequence recognizer, suitable data and separate evaluation. Joining isolated
ASL/RSL gloss labels word by word does not guarantee grammatical sentences.

## 9. Technology and reference links

- [OpenCV documentation](https://docs.opencv.org/4.x/): camera/video input and frame processing.
- [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
  and [Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker): pretrained landmark detection.
- [PyTorch](https://pytorch.org/), [GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html),
  [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) and
  [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html): planned recognizer and training framework.
- [NumPy](https://numpy.org/doc/stable/): numeric landmark arrays and feature archives.
- [Microsoft ASL Citizen project](https://www.microsoft.com/en-us/research/project/asl-citizen/),
  [datasheet](https://www.microsoft.com/en-us/research/project/asl-citizen/datasheet/),
  [paper](https://arxiv.org/abs/2304.05934) and
  [baseline repository](https://github.com/microsoft/ASL-citizen-code).
- [Slovo dataset, models and downloads](https://github.com/hukenovs/slovo)
  and [paper](https://arxiv.org/abs/2305.14527).
- [README.md](README.md): setup and commands.
- [Project decisions](docs/project-decisions.md): decisions and implementation history.
- [Extraction guide](docs/landmark-extraction.md), [local job controls](docs/local-extraction.md)
  and [local CPU benchmark](docs/local-cpu-benchmark-2026-11-07.md).
- [Colab guide](docs/colab.md): optional cloud preparation/training plan.

Earlier WLASL/SVM/I3D implementations were removed at the user's request.
Those experiments and alternative models are historical context, not the
current ASL Citizen + Slovo, MediaPipe + GRU + linear design.
