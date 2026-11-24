# Technical assignment: automatic isolated-sign recognition

**Status:** implementation in progress; experimental, ASL only. Updated 24 November 2026.

## Objective

Recognize individual ASL signs from a webcam without a capture key, using a
rolling landmark window and the selected face-aware 15-sign GRU. Preserve the
manual webcam workflow and the selected epoch-36 checkpoint. This is sign
spotting research, not continuous sentence recognition or translation.

## Scope and constraints

- Use the existing checkpoint, labels, MediaPipe models, 1,741-channel window
  normalization and geometry calculation. Do not train or replace models.
- In the original raw prototype, frames, coordinates and embeddings remain in
  memory; its local logs contain aggregate measurements and predictions. A
  separate [opt-in collection workflow](activity-landmark-collection-2026-11-24.md)
  was subsequently authorized to save hand/pose coordinates locally; it does
  not save video or face mesh. Prompts remain unconfirmed until user review.
- The first prototype emits **raw closed-set predictions**, clearly labeled as
  experimental. It does not establish an activity detector, unknown-sign
  rejection, stable captions, sign boundaries, or reliable recognition.
- ASL/RSL selection is not changed. A future RSL streaming implementation needs
  a compatible trained SLOVO checkpoint and its separate hand-only schema.
- Model architecture and training changes belong in notebooks and training is
  not authorized by this assignment. Do not modify `.gitignore` or `output/`.

## Work packages and acceptance gates

1. **Baseline protocol.** Use the existing manually triggered face-aware model.
   Collect intended labels for HELLO, THANKYOU, PLEASE, YES and NO at comfortable,
   slow and fast speeds, and record capture and prediction timing. Require user
   participation and consent before opening the camera. Until then the baseline
   is a protocol, not measured live accuracy.
2. **Streaming tracking and buffer.** Detect hand, pose and face landmarks once
   per processed frame, retain strictly increasing real timestamps and explicit
   masks, bound the buffer by duration and count, show tracked hands, and drop
   expired frames. Verify finite outputs, resets and window boundaries. Compare
   persistent tracking against fresh-per-clip tracking on the same held-out or
   validation video; do not assume equality.
3. **Raw sliding inference.** Default to a 2-second window updated at most
   twice per second, using actual timestamps. Normalize the full raw observation
   window via the existing model path, recompute geometry, reject insufficient
   detections, and skip scheduled windows while inference is busy. Display one
   raw gloss with explicit warnings; expire old predictions. Record bounded
   aggregate performance metrics, no frames or coordinates. Compare a given
   observation batch against the current preprocessing/model outputs.
4. **Temporal resampling experiment.** Offline, compare unchanged sampling,
   nearest existing observations and timestamp-based interpolation on identical
   validation captures. Synchronize modalities, preserve masks, do not bridge
   long gaps or ambiguous hand assignments, recompute geometry, and report
   timing and per-speed outcomes. Interpolation is not enabled by default.
5. **Rejection and duplicate handling.** Evaluate idle, unknown signs, repeated
   signs and natural transitions separately. Tune thresholds on validation only;
   do not treat confidence or hand presence as proof. Report false outputs per
   minute, missed signs and latency. Static signs must not require hand motion.
6. **Evaluation and decision.** Use signed, user-confirmed attempts and held-out
   signers where possible; report correct, missed and duplicate detections,
   false positives, throughput and end-to-end latency. Maintain the manual mode
   and no-key prototype side by side. Do not reuse the repeatedly inspected test
   set as a fresh final benchmark. Decide on further work using evidence.
7. **Optional subsequent model work.** Only if experiments justify it and after
   explicit authorization, develop fine-tuning, activity detection or continuous
   recognition in a notebook, using appropriately labeled validation data.

## Execution sequence

The current implementation begins work packages 1–3. Resampling, calibrated
rejection, automatic sign boundaries, multi-sign interpretation, RSL and model
training remain separate follow-on work. Results and known limitations must be
documented before a prototype is presented as usable recognition.

## Initial implementation and verification

Run the separate ASL-only prototype with:

```powershell
.\.venv\Scripts\python.exe scripts/webcam_sliding_experiment.py
```

It begins automatically, shows tracked hand landmarks and raw closed-set guesses,
and closes with Q. `--device cpu`, `--camera 1`, `--window-seconds 2` and
`--interval-seconds 0.5` select the respective options. Tracking runs synchronously
on the camera/UI loop; inference has one worker and no pending-work queue. This
prototype was later tried live by the user, who reported that automatic capture
worked but produced arbitrary glosses while hands rested; this is qualitative
feedback, not a benchmark. The local `.cache` session
report contains timing and aggregate counts only, no images or coordinates.

On 24 November, a 46-frame ASL Citizen validation THANKYOU clip was decoded
once and passed both to the existing fresh-tracker adapter and to a newly
initialized streaming tracker over the same full sequence on CPU. Both produced
GOOD with the identical top score (0.3448663652). This verifies one full-clip
path under matching tracker initialization, not the correctness of the label,
rolling-window behavior or equivalence after longer continuous tracking. The
streaming tracking plus scoring took 2.395 seconds in this one offline run,
including tracker creation; this is not a webcam FPS or end-to-end benchmark.

Automated tests check window bounds and data ownership, timestamp and frame-size
validation, missing-detection rejection, and exact equality with the existing
normalizer for an identical observation batch. The baseline protocol still needs
user-confirmed captures. There is no resampling, filtering, rejection calibration
or repeated-sign handling in this prototype.

The matched [guided filter diagnostic](sliding-filter-diagnostic-2026-11-24.md)
found that a provisional score threshold and stability rule missed some correct
labels while still emitting labels at rest. Neither filter was added to the
default prototype; further independent validation is required.
