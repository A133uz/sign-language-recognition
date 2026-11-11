# Five-sign GRU live NO predictions — focused investigation

## Findings and scope

**Follow-up:** the [controlled cause experiments](recognition-cause-experiments-2026-11-11.md)
now include 94 raw/cache comparisons, frozen-head probes, six confirmed live
trials, temporal/depth controls and a reproduced public-video scale failure
isolated to pose observations/validity masks. These establish specific failure
mechanisms; they do not establish one exclusive cause of every historical error.
The sections below preserve the earlier checks in their original scope.

The frozen GRU is sensitive to hand visibility and context at capture endings.
This is measured on existing training/validation features, but is not yet a
confirmed cause of the user's live predictions. Separately, the new FPS limiter
introduced a real Windows-clock regression. That is corrected in code, and
the subsequent paired check confirms that the severe FPS halving is gone.
Recognition remains wrong. The regression cannot explain the original failure,
which predates the limiter.

No training, model selection, video decoding, data cleaning or full audit restart
occurred. The epoch-18 checkpoint and all five-sign cache hashes remain unchanged.

## Checks and recorded actions

1. Verified actual model SignGRU / epoch 18 / five labels, evaluation mode, label
   dictionary and checkpoint identity against the live session. SHA256:
   `78b1a4b02ad7ed9118d9b63ee2f9fe0f5c1d91c6c54c66c41e4c18695d18ca7c`.
2. Read the first completed session: four captures, all NO first, scores
   80.1–94.3%, hands present 343/409 frames and body anchors 409/409. Presence
   does not establish correct joint geometry. Intended labels were not logged,
   so no formal live accuracy is calculated.
3. Read the existing frozen test predictions, without rescoring test: NO appears
   only 6/60 times. This is not a globally constant-output classifier.
4. Ran six prespecified controls on all 75 training / 19 validation clips. The
   unchanged input reproduces the recorded 66/75 and 15/19 correct.
5. Read the newer user-launched session
   `.cache/webcam-recognition-20261111-211947.json`: ten captures, NO first in
   eight, YES in one, THANKYOU in one. Thus its symptom is mostly NO rather than
   literally a constant output. Intended labels are absent here too.
6. Reproduced and corrected the FPS regression using the machine's clock
   resolution; added regression tests and capture-ending diagnostics.
7. Prepared a two-capture HELLO comparison and requested user availability.
   No camera was opened during the investigation.

## Model sensitivity controls

| Control | Train correct / 75 | Validation correct / 19 | Validation NO predictions |
|---|---:|---:|---:|
| Original | 66 | 15 | 1 |
| Repeat final frame throughout sequence | 17 | 3 | 0 |
| Zero pose coordinates/masks | 47 | 12 | 1 |
| Zero hand geometry and hand flags | 18 | 3 | 0 |
| Crop to first/last detected-hand frame | 36 | 7 | 6 |
| Hold last hand features through trailing missing-hand frames | 32 | 7 | 5 |

**74/75 training clips and all 19 validation clips end without a detected hand.**
Median per-clip detected-hand fractions on the prepared grid are 39.84% and
43.48%. These include real rest and are not detector accuracy measurements.
The older webcam log's whole-clip counts do not reveal its ending hand visibility.

All four validation HELLO examples are originally correct; none remains correct
after cropping or trailing-hand filling. A cropped HELLO becomes NO at 89.7%,
and two cropped YES examples become NO at 90.6% and 86.9%. This partly reproduces
the direction/confidence of the live symptom on known dataset examples.

These are **destructive controls**, not alternate benchmark scores, safe
augmentations or approved fixes. Cropping can remove genuine motion when a hand
is missed; filling creates artificial postures. They demonstrate sensitivity,
not that the same changes occurred in the user's input. Do not add fake zeros
or held joints to force a desired answer.

Static-final-frame and hand-removal controls perform poorly, so this evidence
does not show that the model ignores all motion or hands. Completely zero
inputs at the four original lengths predict THANKYOU, not NO; this invalid-input
control alone does not establish valid live geometry.

Initial command (refuses existing output):

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_five_sign_no_collapse.py
```

Local evidence: `.cache/five-sign-gru-no-collapse-20261111-v1/report.json`.
[Portable control results](assets/five-sign-gru-no-collapse-2026-11-11.json).

## Confirmed FPS regression

The newer session delivered **27.76–30.14 FPS**, but retained **15.76–18.09 FPS**.
The initial limiter used `time.monotonic()`: on this Python 3.12 Windows machine
that is GetTickCount64 with **15.625 ms** resolution. Its minimum-interval gate
repeatedly discarded native approximately 30-FPS frames.

Replay of 301 nominal 30-FPS frames over ten seconds using that quantization:
the old sampler retained **161**, the corrected sampler **300**. The GUI now
uses **perf_counter / QueryPerformanceCounter**, reported resolution **100 ns**.
The sampler retains fractional rate credit instead of rejecting every short
interval. Credit is bounded/reset after long stalls; no frames are duplicated.
This is an average-rate target, not perfectly uniform intervals.

The original synthetic tests missed this platform-specific clock case. Tests
now include Windows-quantized 30-FPS timestamps and larger camera jitter.
[Clock replay and pre-fix session evidence](assets/five-sign-gru-fps-regression-2026-11-11.json).
This corrects a demonstrated sampler bug; it is **not a verified recognition
fix**. The paired physical check below retained 29.53 / 27.84 FPS instead of the
previous 15.76–18.09 FPS. The earlier NO-only session already ran at approximately
30 FPS without this limiter.

## Prepared live check

The GUI now saves `tracking_summary`: hand presence by quarter, last-frame hand
presence and the count of hand-present frames in the final ten. No coordinates
or media are saved. `--expected-sign HELLO` optionally displays/logs a prompt;
it is never used by the classifier. Actual completion still needs confirmation.

Planned captures, with the same person, model, camera and intended sign:

1. HELLO once; end capture as usual.
2. HELLO once; return hands to a relaxed position, wait one second, then end.

The plan is stored in `.cache/gru5-ending-context-plan-20261111-01.json`.
After the user approved the comparison, opened
`webcam-gru5-ending-context-20261111-01`: the session confirms SignGRU / epoch 18 /
CUDA / five labels, expected HELLO and `perf_counter` resolution 100 ns. The user
confirmed completing both conditions. The window subsequently closed cleanly
with exactly two recorded trials. The prompt is metadata only.

```powershell
.\.venv\Scripts\python.exe scripts/recognize_webcam.py --expected-sign HELLO --report .cache/webcam-gru5-ending-context-20261111-01.json
```

All **84 project tests** pass, including the integrated synthetic camera loop,
clock regression, prompt tagging and aggregate ending measurements. The matched
capture results are recorded below; no live improvement was assumed in advance.
Model weights and source features are unchanged; the audit stays stopped.

## Completed two-HELLO comparison

The user confirmed the two planned conditions. The checkpoint SHA256 is unchanged.

| Capture / condition | First guess / score | HELLO score | Capture FPS / source FPS | Final ten frames with detected hands |
|---|---:|---:|---:|---:|
| #1, finish as usual | NO / 93.67% | 0.77% | 29.53 / 29.98 | 10 / 10 |
| #2, lower hands and wait one second | YES / 82.68% | 0.94% | 27.84 / 29.25 | 10 / 10 |

Both HELLO attempts are misclassified; HELLO ranks third in both. This is a
two-attempt diagnostic, not a population webcam-accuracy estimate. The prediction
changed, but did not become correct. The corrected sampler retained 67/68 and
100/105 delivered frames: the severe frame-rate halving did not recur.

The intended **no-detected-hand ending was not achieved**: the tracker reports a
hand in every final-ten-frame sample of both clips. The user followed the neutral
hand instruction, but lowering hands did not create the detector-defined condition
seen in the dataset. This does not confirm or disprove that ending's causal role.
Counts cannot identify which hand/pose was detected; no geometry or video was saved.

[Confirmed session and measurements](assets/five-sign-gru-ending-check-2026-11-11.json).
The camera is closed. No model update, retraining or data changes were made.

A focused follow-up is prepared, pending user availability: one automatic
four-second HELLO capture, followed by both hands fully below the image until
the timer ends. Do not press Space to finish. Avoiding an end-of-capture keypress
removes one possible source of extra hand motion; that source has not been
established in the saved captures. Verify the final mask rather than assume it.

Plan: `.cache/gru5-ending-auto4-plan-20261111-01.json`. The user approved and the
window was opened with a four-second limit, expected HELLO, actual SignGRU /
epoch 18 / CUDA / five labels and the corrected high-resolution capture clock.
The user subsequently confirmed completion; the recorded results follow below.
No recognition success is assumed.

## Completed automatic-capture session

The user confirmed the requested HELLO procedure. The session actually contains
**three captures**, not the single capture originally planned. All have expected
HELLO metadata; individual repeats/stop triggers were not separately confirmed.
The first is shorter than the four-second configuration, and stop reasons are
not logged, so automatic termination is not independently established for each.
The window closed cleanly after three results.

| Capture | First guess / score | HELLO score | Duration | Retained / delivered FPS | Final ten frames with hands |
|---|---:|---:|---:|---:|---:|
| #1 | NO / 93.49% | 0.75% | 3.638 s | 26.39 / 29.14 | 10 / 10 |
| #2 | YES / 89.16% | 1.87% | 4.029 s | 26.06 / 28.54 | 10 / 10 |
| #3 | NO / 90.51% | 3.82% | 4.014 s | 26.66 / 29.15 | 5 / 10 |

No capture predicts HELLO first. The third finally has no hand in its last frame,
but only **5/108 frames in the entire capture** lack a detected hand. This does not
verify a whole second of absent-hand ending. Counts do not reveal whether hands
were still visible, the detector retained/hallucinated landmarks, or the intended
motion/ending occurred at a different time. No raw frame/coordinate evidence exists
for distinguishing these explanations. The user confirmation is preserved without
assuming any one of them.

The rate is no longer halved, but retained FPS is still lower than delivered FPS;
the limiter is an average-rate budget, not a fixed-30-FPS resampler. The original
failure at approximately 30 FPS remains independent evidence against FPS being the
sole explanation. No useful recognition improvement or confirmed live cause was
established by these ending tests. Do not keep requesting the same trial without
collecting more informative input evidence.

[Automatic-session snapshot](assets/five-sign-gru-auto-ending-check-2026-11-11.json).

## Fresh public-video pipeline verification

Ran **all four HELLO validation videos** through actual video decoding and the
same `SignRecognizer.predict_frames` function used for webcam captures: fresh
MediaPipe hand/pose extraction → unchanged normalization → frozen epoch-18 GRU.
All four predict HELLO first, with scores **91.89%, 55.42%, 90.20%, 92.37%**.
The complete label rankings match their cached-feature predictions, with maximum
score difference **0.0**. The checkpoint hash is unchanged and no webcam was opened.

This verifies the full frame-to-model path on those dataset examples, beyond
the earlier raw-feature comparison. It does not prove correct landmarks on live
frames or exclude camera-specific capture differences. It also does not establish
that the user performed a wrong sign. Remaining hypotheses concern live landmark
geometry, framing/visibility, hand assignment, timing and model generalization.

Command:

```powershell
.\.venv\Scripts\python.exe scripts/verify_five_sign_video_path.py
```

[Fresh video-path evidence](assets/five-sign-gru-video-path-2026-11-11.json).
The script refuses to overwrite existing evidence. It reads only the four public
validation videos and cached reference inputs; no training, cleaning or test-set
rescoring occurs. A native logging-network warning did not prevent completion or
change any of the four predictions.

The next useful evidence is a direct view of annotated live input, with explicit
permission because current diagnostics promised metadata-only storage. Such images
must remain out of the training data; no recognition fix is assumed in advance.

## Approved two-still landmark review

The user approved reviewing two annotated webcam stills: one during HELLO and one
at the ending. Added `scripts/review_live_landmarks.py` and the bounded
`StillReviewRecognizer` wrapper. This uses the actual recognizer's observations
for both prediction and overlays, exports only the frame closest to one second
and the final retained frame, and rejects additional captures in the same session.
Images remain inside ignored `.cache`, with no video, coordinate-array files,
training-data writes or model changes. Cyan/magenta indicate assigned hand slots,
not independently verified anatomical identity; the image is unmirrored.

Validated automatic finishing despite an early Space press, two-image limits,
unchanged source pixels/observations, and truthful media flags. The complete
85-test suite passed, along with lockfile, compilation and dependency checks.
Launched `.cache/gru5-annotated-review-20261111-01`, verified its ready message
identifies GRU epoch 18, and instructed a four-second automatic capture: HELLO
during the first two seconds, hands below the image for the remaining time.
The user confirmed capture completed as instructed. Capture #1 stopped at its
time limit (4.063 s), retaining 107 frames at 26.09 FPS from a 29.05 FPS source.
The frozen GRU predicted **NO 75.56%, HELLO 17.01%**, then THANKYOU 5.58%, YES
1.47%, PLEASE 0.38%. No model or training-data changes were made.

Reviewed both exported images directly:

- At 1.026 s, one raised open hand is visible and the finger/wrist skeleton
  follows it without an obvious gross displacement. Upper-body pose edges also
  broadly follow the visible shoulders and raised arm. This is one frame, not a
  verification of joint accuracy or identity throughout the motion.
- At 4.063 s, both hands are outside the visible signing area and no hand
  skeleton is drawn. The pose estimator still places wrist/elbow estimates near
  the desk/occluded forearms; hand absence does not imply zero pose features.
- The final ten retained frames all have **zero hand detections**, unlike the
  earlier unsuccessful ending tests. There are 84 hand-positive frames out of
  107 overall. Ten frames do not independently verify a whole second of absence.

This provides an actual no-hand ending while the prediction remains incorrect,
weakening the explanation that hands detected at the ending alone cause NO.
There is no obvious gross hand-tracking failure in the sampled gesture image.
The stills cannot verify motion, continuous hand-slot identity, or normalization
throughout the sequence. Model generalization, timing and sequence-level feature
differences remain hypotheses; neither a definitive cause nor a fix is established.
The two PNGs and session metadata remain only in ignored `.cache`; do not copy
these webcam images into tracked documentation or training data.

## Final classification-layer hypothesis: read-only inspection

The user asked whether the fully connected classification layer could cause the
poor live predictions. Inspected `gru_model.py`, `five_sign_training.py`,
`training.py`, `recognition.py` and the existing epoch-18 test report. No training,
new test-set scoring or model modifications were performed.

- The GRU produces a 256-value final hidden state; dropout feeds a jointly trained
  `Linear(256, 5)` head (1,285 parameters including bias).
- AdamW receives `model.parameters()`, including the head. Raw logits go into
  cross-entropy during training, with softmax applied only for inference scores.
- The runtime loads state strictly, checks the adjacent label mapping, and sets
  evaluation mode, disabling dropout.
- The existing 60-example test confusion matrix contains only six NO predictions
  across all examples. HELLO is correct in 12/14 examples and THANKYOU in 12/12;
  the overall recorded top-1 is 42/60. These are small-sample dataset results,
  not webcam accuracy. They contradict a classifier that always emits NO on
  every input, without excluding head-specific failure on live representations.

No obvious head wiring/loss/activation error was found by inspection. A learned
decision boundary can still fail on unfamiliar hidden representations, as can the
upstream GRU representation itself. Separating these explanations would require
a controlled comparison with the GRU frozen and alternate heads evaluated on the
same inputs. This is a proposed experiment only; existing test results must not
become repeated model-selection feedback, and no extra live coordinates/images
are authorized by this conceptual question.

```powershell
.\.venv\Scripts\python.exe -u scripts/review_live_landmarks.py --output .cache/gru5-annotated-review-20261111-01
```

```powershell
.\.venv\Scripts\python.exe scripts/recognize_webcam.py --max-seconds 4 --expected-sign HELLO --report .cache/webcam-gru5-ending-auto4-20261111-01.json
```
