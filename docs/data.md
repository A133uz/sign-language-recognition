# Dataset and experiment records

WLASL's official source is https://github.com/dxli94/WLASL. Read its Computational
Use of Data Agreement before obtaining or using the data. The repository states
academic/computational use and excludes commercial use. Missing URLs may require
the maintainers' video-request process. This project does not bypass that process
or redistribute dataset clips.

Keep raw clips under `data/raw/` and derived landmarks under `data/processed/`.
Both are ignored by Git. Do not assume the published sample count equals the
number of downloadable, successfully processed examples.

## Inspect a gloss

With the official `WLASL_v0.3.json` metadata in `data/raw/`, run from the project root:

```powershell
.\.venv\Scripts\python.exe -m sign_language inspect-wlasl "thank you"
```

Use `--metadata PATH` to choose a different local JSON file. The output reports
the gloss, its listed clip count, and its distinct known signer IDs. It does not
download videos or check whether their URLs work. Unknown glosses, missing files,
and invalid metadata produce an error. The functions in `src/sign_language/wlasl.py`
can also be imported from Python for further exploration.

## Audit the full metadata

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m sign_language audit-wlasl --output runs/wlasl_metadata_audit.json
```

The terminal shows totals and a SHA-256 fingerprint of the source JSON. The
optional output file contains those totals plus counts for every gloss; it is
overwritten when the command is rerun. `runs/` is ignored by Git. Omit `--output`
to see just the summary, or use `--metadata PATH` to select another input file.

The audit counts distinct recorded signer IDs and their overlap between the
metadata's original split labels. An overlap flags that those splits cannot be
assumed signer-disjoint; verify the meaning and consistency of signer IDs before
constructing new splits. Missing signer IDs are excluded from distinct counts.
Missing URLs are counted, but URLs present in metadata are not checked by this
command and do not establish that a video is accessible or usable.

For the local v0.3 metadata with SHA-256
`31ba5a5c421bdbff87494ee22ca6d4e276a8dfac44f43cc9a58b76ed1c4cd684`,
the audit found 2,000 glosses, 21,083 listed clips, and 119 distinct signer IDs.
Recorded IDs overlapped across the original train/test (104), train/validation
(108), and validation/test (98) splits. These are metadata counts, not a trained
model evaluation or proof of usable video coverage.

On 2026-11-01, a manual HTTP HEAD check of the first listed YouTube URL for each
of `hello`, `thank you`, and `goodbye` returned HTTP 200. This only confirms that
the pages responded to those requests; video playback and downloading were not tested.

## Probe direct MP4 examples

After reviewing the WLASL agreement and relevant source terms, run:

```powershell
.\.venv\Scripts\python.exe -m sign_language probe-wlasl "hello" --limit 3
```

The command selects up to three direct MP4 links from that gloss, downloads each
temporarily, and asks OpenCV to decode its first frame. It deletes each temporary
file afterward. `--limit` accepts 1 to 10, and each file is limited to 16 MiB;
larger files are reported as `too_large`. Use `--metadata PATH` for another JSON
file. YouTube pages are skipped because a webpage response is not a video file.

The result `decoded` means one frame was decoded, not that the complete video,
annotated sign interval, label, or every other video is usable. HTTP errors,
network errors, and decode failures describe a particular attempt and can have
different causes. The command reports each result in JSON and exits successfully
when the probe ran, even if individual examples could not be decoded.

On 2026-11-01, probing all five direct MP4 examples for `hello` produced:
`27184` (ASL Search) `decode_failed` after receiving 114 bytes; `27172`
(SignSchool) `decoded` at 1920×1080; and `27173` (ASLdeafined) `decoded` at
640×480. The remaining examples, `27175` (HandSpeak) and `27177` (Signing Savvy),
returned HTTP 404. These are point-in-time results, not a representative availability rate.

## Interpreting signer IDs

The [WLASL paper, section 3.2](https://arxiv.org/html/1910.11006) says the authors
used face embeddings to match signers across videos and manually checked the
results. The repository identifies `signer_id` as the signer identifier. These
annotations support grouping by ID, but matching or manual annotation can still
be imperfect; ID-disjoint splits do not by themselves prove person-disjointness.
The local metadata shows that many recorded IDs occur in more than one original
split, so the original split is not ID-disjoint.

## Provisional pilot split

Create a new report from the local metadata and draft vocabulary:

```powershell
.\.venv\Scripts\python.exe -m sign_language split-wlasl --output runs/my_pilot_split.json
```

The output path must not already exist; use a new name to preserve earlier reports.
`--metadata PATH`, `--vocabulary PATH`, and `--seed NUMBER` select other inputs or
the random seed (default 42). Both input files' SHA-256 hashes and vocabulary order
are recorded. Reports under `runs/` stay out of Git.

The algorithm joins examples that share a signer ID, exact URL, or video ID into
groups; a seeded shuffle assigns about 60% of the groups to training and 20% each
to validation and test. Percentages apply to groups, not clips. It checks and reports
signer-ID, exact-URL, and video-ID overlap, plus counts and missing glosses for each
split. It makes one random assignment; it does not guarantee every gloss appears in
every split. It rejects missing IDs and mixed signer-ID types so equivalent numeric
and string IDs are not silently treated as separate signers. Different URLs for the
same underlying video may remain undetected.

Each record retains the original source split, URL, signer/video IDs, and frame
annotations. `availability: not_checked` means this manifest is based on metadata,
not verified video files. It is **not ready for model training or evaluation**.
Check actual files and sign intervals, reassess signer-ID reliability, regenerate
the split from usable clips, and audit class coverage before using it.

For the local v0.3 metadata and draft 10-sign vocabulary, seed 42 produced 107/34/29
listed clips and 26/8/8 signer IDs in train/validation/test. Every pilot gloss is
represented in all three partitions, with zero recorded ID or exact-media overlap.
These counts can change when inaccessible clips are removed. Some evaluation
classes have only one recorded signer and one clip, so their metrics would be
highly uncertain even if the files are all usable.

## Inventory local clips

After obtaining clips in accordance with their source terms, put MP4 files in
`data/raw/videos/`, named `<video_id>.mp4`, and run:

```powershell
.\.venv\Scripts\python.exe -m sign_language inventory-wlasl --videos data/raw/videos --output runs/video_inventory.json
```

The directory must exist and the output file must be new. `--metadata` and
`--vocabulary` select different inputs. The report contains one record per
selected metadata instance, input and local-file SHA-256 hashes, annotated frame
bounds, and `missing`, `decode_failed`, or `decoded` status. For decoded files,
it also records the number of sequentially decoded frames, the container-reported
frame count and FPS, and the first frame's dimensions. Duplicate or unsafe video
IDs are rejected. No network requests or file downloads are made.

OpenCV stops when it cannot read another frame, so a `decoded` result does not
prove a file is complete or that the sign interval or label is correct. Frame
annotations may refer to a source video or an already trimmed clip; establish
that interpretation before extracting features. A hash is a snapshot: rerun the
inventory if files change. The current split command does not consume this report;
the usable-clip review and inventory-based splitting are still to be implemented.

Track the following small records under `data/manifests/` once preparation exists:

- Dataset release, source URL, retrieval date, and metadata checksum.
- Video ID, exact source gloss, signer ID, source split, and assigned split.
- Relative local path, clip frame bounds/FPS, download status, and extraction status.
- Exclusions with reasons, per-class counts, and signer coverage.
- Split-generation seed/method and a signer-overlap audit across all splits.

Do not silently merge similar labels or replace dataset labels with English
phrases. Keep any reviewed mapping explicit and versioned. Label order in
`configs/vocabulary.yaml` will define class IDs; save it with every trained model.
PyYAML treats unquoted yes/no as booleans, so those labels are quoted in the config.

Before training, establish a feature schema shared by offline extraction and the
webcam app: landmark ordering, coordinate space, normalization, handedness/mirror
handling, missing-hand masks, sequence sampling/padding, and timestamps. Keep
location relative to the body available where needed rather than removing all
position information through wrist normalization.

Signer-disjoint splitting and coverage need to be audited for the chosen subset;
do not assume every upstream split meets the project's evaluation requirements.
With few samples, some classes may need additional consented recordings or a
smaller evaluation scope. Add idle/non-target footage and a rejection strategy
before treating a closed-set classifier as a live recognizer.

## Experimental isolated-sign pipeline

Install the CPU or CUDA profile and place Google's official
[Hand Landmarker model](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
at `assets/models/hand_landmarker.task`. Put *already prepared* WLASL clips in
`data/raw/videos/<video_id>.mp4`. The [WLASL preprocessing script](https://github.com/dxli94/WLASL/blob/master/start_kit/preprocess.py)
cuts annotated YouTube intervals into clips and copies other sources. The command
below treats every local file as a complete prepared example; it does not trim it.

WLASL's README specifies that interval frame indices refer to decoding at 25 FPS.
The interval extractor now samples source frames at annotation timestamps before
applying the one-based inclusive bounds; merely writing native frames at 25 FPS
does not align them. An audit on 2026-11-04 found five downloaded interval sources
at 29.97 FPS (`70017`, `70220`, `70225`, `70234`, `70259`). Earlier local cuts of
those five used native frame indices. They have been regenerated and fully
decoded for frame-count readback; the earlier cuts are preserved under
`data/raw/superseded_native_fps_intervals/`, with hashes recorded in
`runs/fps25_interval_corrections_20261104.json`. Earlier feature archives and
locally trained checkpoints include those superseded cuts and need regeneration
before further comparison. This correction does not change the externally
pretrained I3D weights or prove the cause of webcam errors.

```powershell
.\.venv\Scripts\python.exe -m sign_language prepare-wlasl --output data/processed/pilot.npz
.\.venv\Scripts\python.exe -m sign_language train data/processed/pilot.npz --output checkpoints/pilot.pt --device cuda
.\.venv\Scripts\python.exe -m sign_language webcam checkpoints/pilot.pt
```

Use new output paths for each run. Press **Space**, perform one sign during the
two-second capture, then read the displayed label; press **Q** to quit. Use
`--device cpu` for CPU training or `--camera 1` to select another camera. An
unmirrored frame is used for features, while the preview is mirrored. During
capture, yellow dots show detected hand landmarks; at least four original frames
with a detected hand are required before making a prediction.

For small-data comparison, `train --method svm` uses a standardized linear
support-vector classifier on the same sequences. It runs on CPU and supports
3–100 classes; GRU remains the scalable default. Its internal vote fraction is
the proportion of pairwise classifiers voting for a label, not a probability;
the webcam window displays its output as an experimental guess. Its standardizer
floors per-feature scales at 0.001 to limit amplification of tiny variations;
`--svm-scale-floor` can change that experimental value for a new training run.

Preparation samples 32 frames per clip, extracts up to two hands with MediaPipe,
sorts them left-to-right by image position, and saves normalized image XYZ plus
presence flags. Missing hands are zeros. Hands can swap order when they cross;
facial expressions are not included. An optional experiment adds seven body
landmarks and each hand's position relative to the nose and shoulder width. Use
Google's [Pose Landmarker Lite model](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)
and pass `--pose-model assets/models/pose_landmarker_lite.task` to both
`prepare-wlasl` and `webcam`. Pose and hand model hashes are checked at inference;
hand-only checkpoints still work without a pose model. A clip must decode at least two frames and
have detected hands in at least four sampled frames. A clip that passes is not
necessarily complete or correctly labeled. The split uses recorded signer IDs,
exact URLs, video IDs and duplicate file hashes; ID or source errors can still
cause leakage. Preparation tries 256 seeded assignments and selects the one with
the fewest missing training classes, then the fewest missing validation/test
classes, breaking ties by clip balance. This uses labels and signer groups, not
model predictions, and the selected seed is recorded. Missing classes in evaluation are reported and make evaluation
incomplete. Training selects a checkpoint by validation loss and reports raw
correct/total counts. The displayed softmax score is not calibrated confidence;
the classifier shows "Uncertain" when the highest score is below the experimental
`--min-score` threshold (default 0.5). This threshold is not calibrated and may
reject valid signs or accept unknown signs. It is not continuous translation or
a production application.

For 66 obtained clips and a coverage-selected split, the hand-only and hand-plus-
pose GRUs each achieved 2/15 on validation. A linear SVM with hand-plus-pose
features achieved 7/15 validation and 12/21 test on that same split, but got
neither held-out `hello` example right. The partitions are too small for a
reliable accuracy estimate, and these results do not validate webcam performance.

Fourteen additional annotated YouTube intervals were extracted using the upstream
one-based inclusive frame indices, producing 80 local MP4 files in total. These
intervals have not been manually checked; changes in an upstream video could
invalidate the annotations. Preparation accepted 79 files and excluded one with
insufficient detected hands. The selected split has 46/15/18 clips; validation
covers all ten classes, while test has no `goodbye` clip. The hand-plus-pose SVM
scored 9/15 on validation and 10/18 on test. On held-out `hello`, it got 1/1
validation and 0/1 test; on held-out `thank you`, it got 0/1 validation and 1/3
test. These tiny counts and unreviewed clips do not establish reliable webcam
recognition. The downloadable source videos and trained checkpoints are ignored
by Git; the retrieval reports are local files under `runs/`.
At the default threshold, the SVM accepted all 15 validation clips, including six
wrong predictions. No unknown-sign set was tested, and the vote threshold is not
evidence that out-of-vocabulary signs can be rejected.

On the same 79-clip split, retraining with the 0.001 SVM scale floor gave 8/15
validation and 13/18 test. The floor was chosen to mitigate very small training
feature scales, not calibrated for webcam deployment. `hello` was 1/1 validation
and 0/1 test; `thank you` was 0/1 validation and 1/3 test. The first webcam test
of the 79-clip SVM used the earlier, unfloored checkpoint.

An academic-use [WLASL mirror](https://huggingface.co/datasets/Voxel51/WLASL)
provided 42 additional local pilot clips. Of 122 local MP4s, 121 passed the
hand-presence check. A coverage-selected split had 74/24/23 clips in train,
validation, and test, with all ten labels represented in each split. The SVM
with the scale floor scored 10/24 validation and 17/23 test. Since these splits
and source clips differ from earlier runs, the figures are not a controlled
comparison. Per-class counts remain small and labels and intervals are unreviewed.

On 2026-11-04 a first direct-MP4 pilot attempt produced 44 files that decoded
at least one frame, from 102 direct URLs. The rest returned errors or failed the
first-frame check. Preparation subsequently decoded and extracted hand landmarks
from these 44 files. With seed 42 the split contains 30/4/10 clips from 10/3/3
recorded signers in train/validation/test. Seven of the ten glosses are absent
from validation and four from test; the clips and labels have not been manually
reviewed. A 100-epoch CUDA run selected epoch 63 by validation loss and scored
17/30 training, 0/4 validation and 4/10 test. These tiny, incomplete partitions
do not support a reliable accuracy estimate and should not be treated as proof of
generalization. A camera frame passed through the feature extractor and model,
but no hands were detected in that frame; this is not a live-recognition test.

In a subsequent user webcam test, "hello" was labeled "sorry", "thank you" was
labeled "good", and an out-of-vocabulary sign was labeled "good". These examples
demonstrate real errors, not an accuracy measurement. An uncertainty threshold
was added afterward; it has not yet been validated on these examples.

In a later webcam test of the 79-clip SVM, the user saw hand landmarks but
reported that `hello`, `thank you`, and `I` were still not recognized correctly.

## Pretrained WLASL2000 video experiment

An alternative uses the I3D architecture from
[pytorch-i3d](https://github.com/piergiaj/pytorch-i3d), licensed under Apache-2.0;
its license is included beside the vendored implementation. The
[published WLASL2000 checkpoint and class mapping](https://huggingface.co/spaces/cedssama/I3D_Sign_Language_Classification/tree/main)
are separate third-party assets subject to their original terms. Place the
checkpoint at `assets/models/wlasl2000_i3d.pt` and its corresponding class list
at `assets/models/wlasl2000_classes.txt`; neither is committed to Git.
The files used here are the [checkpoint](https://huggingface.co/spaces/cedssama/I3D_Sign_Language_Classification/resolve/main/weights/asl2000/FINAL_nslt_2000_iters%3D5104_top1%3D32.48_top5%3D57.31_top10%3D66.31.pt)
and [class list](https://huggingface.co/spaces/cedssama/I3D_Sign_Language_Classification/resolve/main/wlasl_class_list.txt).
Their respective SHA-256 hashes are `243a19e6deef3becffbfc5b7dd8adb32916c8bee482565ca243c082584732620`
and `2c131fb7f052340aa7ae8579c8cae9746028740b8e62efeb957a7790dbefcf4a`;
the program checks these to prevent mismatched class mappings or weights.

```powershell
.\.venv\Scripts\python.exe -m sign_language webcam-i3d --device cuda
```

Press Space and perform one isolated sign, then wait for up to three guesses.
Frames are resized to a 256-pixel short side, center-cropped to 224 pixels, and
sampled to at most 64 frames. The preview box approximates the area the model
sees; keep your face and signing hands inside it. Webcam frames are sampled by
capture time so uneven frame arrival does not distort their temporal spacing;
saved clips are sampled by frame order. This resizing is an experimental departure from the
original WLASL evaluation code. On one local WLASL test clip per pilot class,
the model's top guess matched 4/10 labels; `goodbye` produced `bye`. These clips
may overlap its original training data or signers, and this tiny sample is not
an independent accuracy evaluation. It cannot reject unknown gestures and has
not been validated on the user's webcam signing.

In a webcam test of the unrestricted WLASL2000 model, the user reported unrelated
guesses including `caption` and `cross` for `hello`, and `pillow` and
`responsibility` for `thank you`. Thus the unrestricted webcam model failed that
test; the labels are recorded as reported, without a full ranked-output log.

To restrict this experiment explicitly to the draft pilot vocabulary, add
`--vocabulary configs/vocabulary.yaml`. The window labels this as pilot-only;
it will always choose from those labels, including when the signer performs a
different sign. On 19 locally available clips from the original WLASL test split,
top-one matched 7/19 labels over all 2,000 classes and 15/19 after restricting
guesses to the ten pilot labels. This is an exploratory closed-set comparison,
not independent webcam validation. The original WLASL split is not guaranteed
signer-disjoint and these 19 clips should not be treated as a final benchmark.

An optional [twelve-label configuration](../configs/vocabulary_12_signs.yaml)
adds the distinct WLASL labels `i` and `me`. A small visual check of available
examples suggests that `i` is the fingerspelled letter, while `me` uses a gesture
toward the signer; the interpretation still needs sign-language review. In an
exploratory check, this configuration returned the correct top label for 15 of
the same 19 pilot test clips, plus one available `i` and one `me` validation
clip. These two extra examples are far too few to validate either label, and
the pretrained model may have seen related examples or signers during training.
This configuration does not replace the ten-sign pilot.
In a subsequent user webcam test of the twelve-label mode, the pronoun gesture
produced `hello` first and `me` third. This is not a successful top-one recognition.

To inspect an already saved, isolated sign clip without opening a webcam, run:

```powershell
.\.venv\Scripts\python.exe -m sign_language predict-i3d data/raw/videos/27172.mp4 --device cuda --vocabulary configs/vocabulary.yaml
```

The command decodes at most 600 frames and prints the top three glosses and
uncalibrated scores as JSON. With a vocabulary restriction the scores are
relative only to the selected classes; without it, predictions cover the model's
2,000 labels. A score is not proof that a sign is present or recognized correctly.

For an exploratory per-gloss report on available local WLASL examples, run:

```powershell
.\.venv\Scripts\python.exe -m sign_language evaluate-i3d --device cuda --restricted --source-split test --output runs/i3d_pilot_report.json
```

The output must be a new file. It lists missing and decoded clips, top-one and
top-three counts per gloss, individual guesses, signer IDs and file hashes.
Omit `--restricted` to select from all 2,000 classes; `--vocabulary` changes the
glosses included in the report. `--limit` stops after the first listed examples,
including missing files, and therefore does not produce a representative sample.
The upstream split is not signer-disjoint, and the pretrained model may have seen
the same videos or signers. These reports are exploratory, not independent model
evaluation or evidence of webcam performance.

In a user webcam test of the pilot-only I3D mode, `hello`, `thank you`, `please`,
and `sorry` were each reported as the first guess for the corresponding sign.
The model did not put `yes` or `no` first; `no` appeared third. These reports
are evidence for those attempts, not a measured accuracy or proof that the
remaining classes or unfamiliar signers are recognized reliably.
The user later reported that both `yes` and `no` were labeled `thank you` in a
guided-crop retry. The guide changed the preview, not the trained model.

For each experiment, record commit, lockfile/config identifiers, data and feature
versions, vocabulary size, seed, split counts, metrics, hardware, and latency.
Curate small results under `reports/`; keep generated output under ignored `runs/`.

## Corrected-clip rebuild and guided webcam diagnostics

After the 25-FPS interval correction, all 132 local clips passed the landmark
preparation check. The regenerated archive is
`data/processed/wlasl_expanded_pose_fps25_20261104.npz`; its report contains current
video hashes, all checked against the files on disk. The new twelve-label SVM is
`checkpoints/wlasl_expanded_pose_fps25_20261104_svm.pt`. It scored 11/21 validation
and 17/31 test, with zero correct validation predictions for `me` and `i`.
It is not a validated webcam replacement. Old archives remain available for
comparison and must not be mistaken for the corrected features.

An I3D comparison on the same 21 validation clips scored 20/21 with the current
aspect-preserving resize and mean temporal pooling. Following the reference
test path's spatial crop, all-frame input and max temporal pooling scored 7/21;
square resizing with 64 sampled frames and max pooling scored 18/21. These are
restricted twelve-label, exploratory results, with possible pretrained exposure;
they do not explain or resolve the user's live errors. The current default stays
unchanged. See [the experiment record](../reports/2026-11-04-webcam-diagnostics.md).

The guided test collects repeated, randomized prompts and compares aspect/square
resizing and mean/max pooling on each capture:

```powershell
.\.venv\Scripts\python.exe -m sign_language test-webcam-i3d --device cuda --vocabulary configs/vocabulary_12_signs.yaml --sign hello --sign "thank you" --sign yes --sign no --sign me --repeats 3 --output runs/webcam_guided_01.json
```

Space starts a two-second countdown. Begin signing when `SIGN NOW` appears;
capture lasts two seconds by default. Space after the result advances to the
next trial. Q, Escape or closing the window saves the completed trials as a
partial report. A failed camera or inference saves completed trials with a
failure status when possible. The output path must be new.

Frames are retained only in memory at a 256-pixel short side, then sampled by
timestamps. The square comparison resizes these retained frames to 256x256;
it is a diagnostic option, not an exact reproduction of the upstream training
loader. No video, landmarks or embeddings are saved. The JSON contains prompts,
predictions, frame counts, capture spans, comparison latency, asset/config/lock
hashes and per-sign agreement counts. Scores are uncalibrated and relative to the
entire selected vocabulary, even when only some signs are prompted. `me` remains
separate from the fingerspelled `i`. Failed and uncompleted prompts remain visible;
a correct third guess does not count as correct top-one agreement.

Prompts are intended labels rather than independently reviewed ground truth.
Same-capture comparisons help diagnose preprocessing, but any chosen change
needs fresh trials across the intended vocabulary. This command has passed
simulated camera lifecycle tests and real CUDA inference on saved `yes` and `no`
clips before its first human test.

That first human session has now completed: all four variants matched `hello`
and `thank you` on 3/3 prompted attempts each, and `yes`, `no` and `me` on 0/3
each. The user confirmed the prompts were followed and capture seemed normal.
No video or features were saved, and no comparison variant was selected as a fix.
See [the guided results](../reports/2026-11-04-guided-webcam-results.md) for counts,
timing and the subsequent, unsuccessful body-relative landmark probe.

## Optional ASL Citizen baseline

WLASL remains the main dataset and full-vocabulary target. A separate diagnostic
backend uses the [published ASL Citizen I3D checkpoint](https://github.com/microsoft/ASL-citizen-code/releases/tag/checkpoints_v1).
Microsoft describes its dataset as recordings by everyday signers in everyday
settings; this motivates a webcam comparison, but does not establish better
recognition on this project's camera. The backend does not merge datasets or
relabel WLASL examples for training.

Obtain the official `ASL_citizen_I3D_weights.zip` release and place its
`ASL_citizen_I3D_weights.pt` member at `assets/models/aslcitizen_i3d.pt`.
Generate `assets/models/aslcitizen_classes.json` from the official dataset's
`splits/train.csv`: read UTF-8 CSV, take stripped `Gloss` values, sort the unique
strings, and save with Python `json.dumps(labels, indent=2)` using UTF-8 and no
trailing newline. This reproduces the source loader's class order, with 2,731
entries. The program checks both asset hashes, recorded in
[the baseline experiment report](../reports/2026-11-04-aslcitizen-baseline.md).
The official archive's `use.txt` contains Microsoft Research License Terms for
non-commercial research, including redistribution restrictions. Keep these
dataset assets and derived class metadata local and ignored; they are not covered
by the repository's code licensing choices.

```powershell
.\.venv\Scripts\python.exe -m sign_language test-webcam-i3d --backend citizen --device cuda --sign HELLO --sign THANKYOU --sign YES --sign NO --sign ME --repeats 3 --seed 43 --output runs/webcam_citizen_01.json
```

This backend defaults to `configs/vocabulary_aslcitizen.yaml` and its own model
assets. The twelve native labels include separate `FINE1`/`FINE2`; no alias to
WLASL `fine`, `goodbye`, or `i` is applied. The fingerspelled `i` is not in this
experimental vocabulary. All candidates remain available during every prompted
trial; selecting five prompts does not restrict guesses to those five signs.

The source-derived preprocessing retains BGR, applies its resize and 224 center
crop, and selects up to 64 frames using the source loader's centered temporal
stride. Short sequences repeat their last frame to reach 64; the source loader
randomly pads with either first or last, so this is an explicit deterministic
departure. Prediction takes the maximum over linearly upsampled temporal logits.
The guided camera requests 640x480, retains native frames only in memory with a
256-MiB limit, resamples irregular frame arrival by timestamps, and draws the
corresponding wider crop guide. The actual capture shape is recorded in the JSON.

Real CUDA checks matched the reference-derived input tensors and top-five
predictions exactly on all ten downloaded source test clips. On this deliberately
small sample (two signers per prompted sign), full-class top-one was 6/10 and
restricted top-one 8/10; both `THANKYOU` clips were labeled `GOOD`. On ten WLASL
validation clips of the same five glosses, the model scored 3/10 full-class and
7/10 restricted. These unrepresentative probes are not a claim of generalization
or a demonstrated improvement over the WLASL model. Its subsequent human test
completed 15 prompts: HELLO 2/3, THANKYOU 1/3, YES 1/3, NO 0/3 and ME 1/3
top-one agreement (5/15 total). The alternative is not selected as a fix.

## Inspecting and comparing body framing

`test-webcam-i3d --quality` adds an actual spatial model-crop preview. It retains
native camera frames only in memory and extracts 32 hand/pose samples after
capture. The report saves aggregate detection counts, hand bounding-box sizes
in model pixels, fraction of detected hand points within the crop, and median
nose/shoulder framing. It does not save pixels, landmark coordinates or embeddings.
It uses the local hand and pose Tasks files, whose hashes are included in the
report. The measurements cover all detected hands, including possibly inactive
hands; they are not a sharpness test or recognition-quality guarantee.

A measured framing difference motivated an optional WLASL body-alignment
comparison. Build a local reference from a prepared hand/pose archive:

```powershell
.\.venv\Scripts\python.exe -m sign_language build-framing-reference data/processed/wlasl_expanded_pose_fps25_20261104.npz --output assets/models/my_framing_reference.json
```

Only training records contribute. Each source video hash is checked against the
feature manifest before measuring its dimensions; changed clips are rejected.
The output must be new and remains ignored with the local model assets. This
implementation reproduced the exploratory reference exactly using 80 training
clips: median nose (116.39, 56.63), shoulder span 103.03 model pixels.

```powershell
.\.venv\Scripts\python.exe -m sign_language test-webcam-i3d --device cuda --vocabulary configs/vocabulary_12_signs.yaml --align-body assets/models/my_framing_reference.json --sign hello --sign yes --sign no --sign me --repeats 1 --output runs/framing_comparison.json
```

The flag enables the quality measurements and compares `body_aligned` with the
original guesses on the same capture. One fixed scale and translation map the
capture's median nose and shoulder span to the reference; the transform does
not track individual hands or change motion over time. It produces 256x256
frames for the existing 224 center crop. Uncovered pixels use neutral gray;
padding above 35% or relative zoom outside 0.5–2 is rejected. Alignment needs
visible nose and shoulders in at least eight sampled frames. A rejected alignment
remains visible as an error and a missing variant prediction, while the original
guesses remain in the report. The right panel shows the aligned last frame after
each successful comparison. No aligned video or per-capture matrix is saved.

The comparison passed a complete saved-clip replay with real CUDA/MediaPipe and
the local checks (88 tests). On the existing 21 validation clips it matched 21
labels, versus 20 originally. These are small, repeatedly inspected validation
data with possible pretrained exposure, not independent accuracy. Human framing
measurements and current test status are in [the framing record](../reports/2026-11-04-framing-diagnostic.md).
The same-capture human comparison then scored 1/4 top-one for both original
and aligned inputs: hello matched, yes/no/me did not. It is not a validated fix.

For a subsequent native hand-shape model, the twelve-label ASL Citizen subset
is now local at `data/raw/aslcitizen/videos`, with original split CSVs under
`data/raw/aslcitizen/splits`. It contains 177/46/146 train/validation/test videos
from 28/5/11 selected signer IDs, with no signer-ID overlap across these splits.
The acquisition checked archive CRCs and wrote
`runs/aslcitizen_pilot_acquisition_20261104.json`; full decoding is recorded in
`runs/aslcitizen_pilot_inventory_full_20261104.json`. All 369 files decoded to
their container frame counts. One training YES clip (`9953610281521326-YES.mp4`)
contains 664 frames and exceeds the existing preparation bound; do not silently
truncate it or describe that bound as a codec failure. Labels and intervals are
not independently reviewed by these checks. The native feature/training pipeline
described below is now implemented; data availability alone does not establish
webcam improvement.

## Native world-hand model

```powershell
.\.venv\Scripts\python.exe -m sign_language prepare-citizen --output data/processed/my_citizen_world.npz
.\.venv\Scripts\python.exe -m sign_language train data/processed/my_citizen_world.npz --method svm --representation world_summary --skip-test --output checkpoints/my_citizen_world.pt
```

Preparation preserves native class names and source memberships, rejects signer
overlap and identical video bytes across partitions, and records video/model/CSV
hashes. It samples up to 32 distinct frames, requires hands and world hands in
at least four distinct samples, and repeats shorter sequences to 32 only after
that check. The existing 600-frame bound rejects the 664-frame source clip rather
than silently truncating it. Native source labels and intervals remain unreviewed.

The schema `hands-pose-world-wrist-normalized-v1` has 288 values per frame:
the previous 160 image/body values plus two blocks of 63 world coordinates and
one validity bit each. Each estimated 3D hand is centered at its wrist and scaled
by wrist-to-middle-MCP distance. World hands are paired with the image-x-sorted
image hands; no independent world-x sorting is used. These are estimates from
[MediaPipe](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python),
not measured physical hand geometry.

`world_summary` uses normalized world-hand shape, wrist-to-nose body-relative
offsets and image hand-presence bits: 134 values per frame. Four temporal-bin
means and whole-sequence standard deviation produce 670 classifier inputs.
The scaler is fitted on training data only. SVM exports contain numeric tensors
for linear or RBF inference rather than a scikit-learn pickle. Exported inference
was checked against scikit-learn predictions, including multiclass RBF voting.

Use `--skip-test` while selecting a model. After freezing the chosen checkpoint,
score one explicit partition and write a new report:

```powershell
.\.venv\Scripts\python.exe -m sign_language evaluate-landmarks data/processed/my_citizen_world.npz checkpoints/my_citizen_world.pt --split test --output runs/my_citizen_test.json
```

The local prepared archive accepted 173/46/146 clips from 27/5/11 source signer
IDs, with all twelve native classes in each partition. Four validation-only
candidates scored 32/46 raw-linear, 40/46 compact-linear, 34/46 compact-RBF and
27/46 GRU. Selection used validation macro accuracy, then match count. The chosen
compact linear checkpoint scored 104/146 on its first held-out test evaluation;
per-class counts and limitations are in [the experiment report](../reports/2026-11-04-world-hand-model.md).

```powershell
.\.venv\Scripts\python.exe -m sign_language test-webcam --backend landmarks --weights checkpoints/aslcitizen_world_linear_20261104.pt --device cpu --sign HELLO --sign THANKYOU --sign YES --sign NO --sign ME --repeats 3 --output runs/my_world_webcam.json
```

The backend reads class names from the checkpoint and uses all its classes unless
an explicit matching vocabulary is supplied. `test-webcam-i3d` remains an alias
for `test-webcam` for existing video-model commands. The landmark backend requests
640x480 and captures first, then extracts the same max-dimension-480 RGB MediaPipe
features as preparation. It uses the whole frame rather than an I3D center crop.
The SVM runs on CPU; the GRU training comparison exercised CUDA. Predictions,
timing and tracking counts are saved, without video or landmark coordinates.
Insufficient tracked hands produce a recorded rejection and no forced guess.
This is not an unknown-sign detector: tracked untrained gestures can still be
misclassified. SVM vote fractions are not calibrated probabilities.

On three saved source yes/no/me clips, capture-path features matched the prepared
features exactly, and predictions agreed. The complete guided path passed a
saved-clip replay using real CPU MediaPipe/classification. Local checks passed
98 tests. The first human session matched YES and NO on 3/3 attempts each,
HELLO 2/3, THANKYOU 1/3 and ME 0/3 (9/15 total), with tracking present in every
trial. This supports progress on yes/no and records continuing failures; it
does not establish reliable recognition across the intended vocabulary.

An optional timing-augmented candidate can be trained with
`--temporal-augmentation --method svm --representation world_summary`. It
augments training sequences only, repeating endpoints and varying time sampling;
the fitted scaler uses augmented training data only. Validation/test clips remain
unaugmented. The setting is experimental and disabled by default. To compare
compatible landmark models on one shared extraction, add
`--compare-weights CHECKPOINT` to `test-webcam --backend landmarks`. The report
hashes both weights and labels the variants `landmarks` and
`comparison_landmarks`. See [the timing experiment](../reports/2026-11-05-timing-augmentation.md)
for commands, validation regressions, runtime checks and human-test status.

The optional `--representation body_geometry` is another 288-value-archive
transform, retaining all fingers relative to the nose/shoulders and derived
finger joint angles/distances. It produces 745 summary inputs and also supports
`--temporal-augmentation`. The raw extractor and native labels do not change;
compatible models can use the same-capture comparison. See
[the geometry experiment](../reports/2026-11-05-body-geometry.md) for its definition,
commands, regressions and actual webcam status. It is not a default replacement.

`sign_language.gesture_review.review_gestures(tracking=True)` is an optional
preview-only diagnostic with colored finger markers and estimated world-hand
views. It does not classify signs, save images/features or collect training
examples. A separately supplied loopback-only endpoint may display two preview
boards to the reviewing assistant only with explicit user consent. This is not
the normal recognition flow. Findings and limits are in
[the tracking review](../reports/2026-11-05-tracking-review.md).
