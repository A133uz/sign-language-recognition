# ASL webcam test: ST-GCN and GRU

Run with the project environment:

```powershell
.\.venv\Scripts\python.exe scripts/recognize_webcam.py
```

The default checkpoint is now
`models/asl-gru-five-signs-early-v2/regularized/best.pt` (regularized GRU,
epoch 18, five labels), as requested for the current webcam test.
To select the preserved full-vocabulary ST-GCN, pass
`--checkpoint models/asl-stgcn27-v1/best.pt`; for the full-vocabulary GRU, pass
`--checkpoint models/asl-gru-128-cv-regularized-v1/official-refit/best.pt`.
Those full-vocabulary checkpoints predict across all 2,731 ASL Citizen labels.
They do not apply a restricted vocabulary or automatic unknown-sign rejection. This is a test
of manually bounded isolated signs, rather than continuous signing.

The explicit equivalent of the default **five-sign GRU pilot** is:

```powershell
.\.venv\Scripts\python.exe scripts/recognize_webcam.py --checkpoint models/asl-gru-five-signs-early-v2/regularized/best.pt
```

Its header must show **GRU, epoch 18, CUDA, 5 labels**: HELLO, NO, PLEASE,
THANKYOU, YES. It was trained with five outputs, not by hiding full-vocabulary
predictions. It remains closed-set: other signs can be assigned one of these
labels. The first five-sign GRU user check produced NO in all four logged captures;
reliable live recognition is not established. See
[pilot results and limitations](five-sign-early-training-2026-11-11.md).

The later [controlled diagnostic](recognition-cause-experiments-2026-11-11.md)
completed six confirmed HELLO/THANKYOU/NO trials, with 4/6 original first guesses
correct. It reproduced sensitivity to pose observations/validity masks and to
later sequence context; replacement heads and mirroring did not resolve the
pilot. These are small diagnostic samples, not a webcam accuracy benchmark or a
verified repair. The default checkpoint and preprocessing remain unchanged.

The [masked mean-pooling comparison](gru-mean-pooling-2026-11-11.md), completed
2026-11-12, improves fixed test results to 47/60 but scores 4/6 in the paired live
check versus the original's 5/6. It recognizes both HELLO/THANKYOU attempts but
calls both NO attempts HELLO. It remains experimental; the default stays the
original GRU. Select it explicitly with
`--checkpoint models/asl-gru-five-signs-mean-pool-v1/best.pt`; its header reads
**GRU mean-pool, epoch 4**.

The same five-sign experiment is also available with **ST-GCN27**:

```powershell
.\.venv\Scripts\python.exe scripts/recognize_webcam.py --checkpoint models/asl-stgcn27-five-signs-early-v1/best.pt
```

Its header must show **ST-GCN27, epoch 4, CUDA, 5 labels**. This is distinct from
the default five-sign GRU and the older epoch-39, 2,731-label ST-GCN. Offline pilot test top-1 is
65.0% (39/60), with particularly weak NO recall; no live result is established.
See [the ST-GCN pilot report](five-sign-stgcn-2026-11-11.md).

The [2026-11-12 mean-pooling request](stgcn-mean-pooling-repeat-2026-11-12.md)
confirmed that this ST-GCN already pools by the mean over valid time and all
nodes. A fresh preserved repeat has identical selected model tensors and test
predictions. Its explicit path is
`models/asl-stgcn27-five-signs-mean-pool-repeat-v1/best.pt`; it remains ST-GCN,
epoch 4, five labels. It is not a different pooling architecture or the default.

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
The capture buffer now targets **30 FPS** by default. Use `--capture-fps 30`
to specify the same limit explicitly (valid range 1–60).
Scores are softmax outputs, not calibrated confidence. Numbered gloss variants
are the saved dataset labels. Record the sign you actually performed when
judging predictions; the application does not know a trial's true label.

## Data path

The UI continuously reads unmirrored camera frames, requests 30 FPS from the
driver, and rate-limits frames retained for recognition to the requested budget.
A fractional-frame credit budget tolerates arrival jitter without repeatedly
dropping native 30-FPS frames. Faster frames are skipped without unbounded catch-up; slower input
is retained without duplication. Actual selected-frame timestamps are preserved.
A separate worker draws a hand-skeleton preview without making capture wait for detection.
Only the displayed preview is mirrored. The preview may lag the camera slightly.

The result view shows measured **capture FPS**, the target and skipped frames.
Session JSON records `camera_reported_fps`; each prediction records
`capture_fps_limit`, `source_frame_count`, `source_capture_fps`,
`fps_limit_dropped_frames` and `effective_capture_fps`. Reported driver FPS is
not assumed to equal measured delivery. These are aggregate measurements;
video and landmark coordinates are not saved.

The existing 128-step normalization still runs afterward. A long capture can
therefore be uniformly downsampled further; limiting capture FPS does not make
the model's final time steps a fixed 30-FPS grid or standardize the dataset.

The preceding four NO-only captures measured **30.45, 29.87, 29.36 and 30.24 FPS**
before this limiter existed. Their rates were already near the five-sign training
clips' approximately 30 FPS. This is a consistency guard, not an established fix
for the recognition failure. Synthetic 60→30, jittery-30, slower-15 and stall
tests and an integrated synthetic camera loop initially passed (81 tests), but
the later physical-camera session exposed a Windows clock regression: only
15.76–18.09 FPS was retained from roughly 30-FPS delivery. The corrected version
uses high-resolution `perf_counter` and fractional frame credit. It passes
**84 tests**, including a quantized Windows-clock replay. A subsequent paired
HELLO check retained 29.53 / 27.84 FPS, confirming no severe halving, but both
guesses remained wrong (NO then YES). Both endings still had detected hands.
See the [focused investigation](five-sign-no-collapse-investigation-2026-11-11.md).

Captures now include a `tracking_summary` of hand presence by quarter, last-frame
presence and a final-ten-frame count. These are counts/fractions, not coordinates.
For a labeled diagnostic use `--expected-sign HELLO`: the prompt is displayed
and logged, but never passed to the classifier or used to change its predictions.

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
