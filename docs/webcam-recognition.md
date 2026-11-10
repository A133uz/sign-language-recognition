# ASL webcam test: ST-GCN and GRU

Run with the project environment:

```powershell
.\.venv\Scripts\python.exe scripts/recognize_webcam.py
```

The default checkpoint is now
`models/asl-stgcn27-v1/best.pt` (ST-GCN, epoch 39).
To select the preserved GRU instead, pass
`--checkpoint models/asl-gru-128-cv-regularized-v1/official-refit/best.pt`.
It predicts across all 2,731 ASL Citizen labels. There is no restricted pilot
vocabulary, fine-tuning, or automatic unknown-sign rejection. This is a test
of manually bounded isolated signs, rather than continuous signing.

## Controls

1. Keep your face, shoulders and hands visible, with room for the full motion.
2. Focus the camera window and press **Space**. After the brief preparation
   countdown, wait for **SIGN NOW**, then perform one complete sign.
3. Press **Space** again immediately after the sign. Capture automatically
   ends after eight seconds. **R** cancels a capture or clears previous guesses.
4. Wait for landmark processing and the five ranked predictions. **Q** or
   Escape closes the window. Repeat with another isolated sign.

Use `--camera 1` to select another webcam, `--device cpu` to explicitly use CPU
classification, or `--max-seconds 5` to change the capture cap (1–10 seconds).
Scores are softmax outputs, not calibrated confidence. Numbered gloss variants
are the saved dataset labels. Record the sign you actually performed when
judging predictions; the application does not know a trial's true label.

## Data path

The UI captures unmirrored raw frames at the camera's delivered rate. A separate
worker draws a hand-skeleton preview without making capture wait for detection.
Only the displayed preview is mirrored. The preview may lag the camera slightly.

After capture ends, the worker creates fresh MediaPipe hand and pose-lite
trackers for that sign, using the offline extractor's CPU VIDEO settings and
the actual capture timestamps. It observes every captured frame, calls the
shared `frame_observation` and **unchanged** `normalize_sequence`, reproduces
the prepared cache's float16-to-float32 rounding, then runs the frozen GRU.
This is 254 features per frame, uniform whole-sign sampling to at most 128
frames, and the true sequence length passed to the packed GRU.

Inference uses the saved weights, label order, preprocessing identity and
model source hashes. The classifier runs on CUDA by default; the checkpoint
configuration selects cuDNN (enabled for ST-GCN, disabled for GRU). A capture without any
hand detections or valid shoulder anchors is rejected with a retry message.
Other poor captures can still generate an incorrect closed-set prediction.

Live video and landmark coordinates stay in memory and are discarded. Only
predictions, aggregate tracking counts, timing and checkpoint identity are
written to a unique `.cache/webcam-recognition-*.json` session report (or an
explicit `--report` path). No training or
model updates occur. The original and archived checkpoints are preserved.

## Verification and limits

On 2026-11-10, all **48 project tests** passed, including checkpoint restoration,
prepared-feature precision, inference without weight/gradient updates,
missing-tracking rejection and camera cleanup. A complete training video
(`REGISTER`, 57 frames) passed through fresh hand/pose extraction and the
recognizer; its top-five labels/scores matched the prepared-cache inference
on the same clip. This checks input compatibility, not independent accuracy.

The first webcam session was launched with epoch 37 on CUDA. The user reported
three incorrect first guesses: THANKYOU → FROWN1, HELLO → NONSENSE and
NO → THUNDER. These are reported sign/result pairs, not a fully labeled trial
set or a statistically meaningful estimate of webcam accuracy. Session reports
contain multiple attempts without ground-truth prompts, so do not assign the
remaining trials a true label or count them as successes/failures automatically.

Hands and shoulder anchors were present in all 95 frames of a FROWN1 capture,
and throughout several THUNDER captures. Camera delivery was approximately
30 frames/second. Other attempts had missing hands. Presence counts do not
establish correct finger geometry, hand assignment or sign boundaries; they
only show that complete detector dropout does not explain every failure.

A read-only review of the existing frozen official test predictions found:

| Label | Test examples | Top-1 correct | Top-5 correct |
| --- | ---: | ---: | ---: |
| HELLO | 14 | 8 | 11 |
| THANKYOU | 12 | 9 | 10 |
| NO | 11 | 7 | 8 |

These small per-class samples show that the model learned the labels, with
imperfect generalization even within dataset testing. They do not identify
the cause of the webcam errors. The single-video input-compatibility check
above also does not rule out framing, tracking, timing or signer differences
on live input. A controlled next diagnostic should label each prompted trial
and compare capture/feature characteristics with training examples before
choosing another training change. No retraining, vocabulary restriction or
checkpoint replacement was performed in response to this feedback.

The offline test accuracy remains 49.92%
top-1; it does not establish accuracy for a new webcam signer. This version
processes landmarks after each capture, so results have a processing delay;
it does not claim continuous or low-latency real-time recognition.

The subsequent [documented investigation](recognition-investigation-2026-11-10.md)
contains checkpoint/data controls, a confirmed six-trial live comparison,
input-size and hand-visibility-context experiments, and their limitations.
All timing/mirroring variants failed those six trials. The strongest measured
model weakness was sensitivity to hand-visibility/rest context; additional
tracking failures and smaller hand images were observed. No single cause or
successful fix is claimed. Diagnostic code is separate from default inference;
no retraining or automatic default orientation/cropping change was made.

## Trained ST-GCN session

The first trained ST-GCN webcam window was opened on 2026-11-11 using epoch 39.
Preflight identified a physical input-length mismatch: ST-GCN inference now
uses the same 128-step padded grid as training/evaluation, retaining the true
length. A trained-model comparison then matched cached top-five scores exactly.
This did not change the model weights or offline test metrics. See
[the launch, compatibility checks and pending feedback](stgcn-webcam-test-2026-11-11.md).

The first session retained a misleading hardcoded GRU processing label despite
loading ST-GCN. The corrected UI derives all model labels from the loaded
class, shows its epoch/device persistently and numbers captures in both the
window and log. Session 02 is the corrected rerun; recognition quality remains
unconfirmed. Full details and the user's original observations are in the
live-test record above.
