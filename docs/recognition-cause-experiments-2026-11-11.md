# Recognition failure: controlled diagnostic experiments

The user requested experiments to identify the cause of poor webcam recognition.
This record contains procedures, evidence and limitations, not private reasoning.
No exact cause is assumed in advance. Existing checkpoint, extracted features,
official test split and the stopped corpus-cleaning job are preserved.

## Prespecified offline comparisons

Command: `scripts/investigate_recognition.py --output .cache/recognition-forensics-20261111-v1`.
Source: frozen five-sign regularized GRU epoch 18, 75 train / 19 validation clips.

1. Reproduce all predictions and raw-to-cache normalization for these 94 clips.
2. Measure hand visibility/slot changes, framing, source FPS, duration and feature
   clipping. Quality flags are not confirmed detector errors.
3. Compare temporal reversal, shuffling, static active poses, suffix-only inputs,
   temporal crops and fixed sequence lengths.
4. Compare hand-slot exchange, bilateral reflection, removal of feature groups,
   and replacement/appending of training-derived rest context. These are
   destructive sensitivity controls, not valid new benchmark scores or fixes.
5. Freeze the GRU and fit fixed linear and 64-unit MLP diagnostic heads to training
   embeddings only. Also fit simple linear classifiers to timing/visibility,
   pose summaries and hand summaries. No hyperparameter search; these probes
   are isolated from production and are not selected using the official test.

## Planned matched live comparisons

The user approved six guided trials: two each of HELLO, THANKYOU and NO. They
will compare original and diagnostic inputs from the same in-memory captures,
plus the separately fitted diagnostic heads. Save predictions and aggregate
measurements only: no video, stills, per-frame joint coordinates or live embeddings.
Wait for the user to confirm the prompted signs were performed normally before
interpreting results as labeled trials. Do not train on these diagnostic captures.

## Interpretation policy

A transformed input becoming correct demonstrates intervention sensitivity for
that capture, not a general fix. The same control must be considered on dataset
examples and repeated prompted captures. A small pilot can establish specific
failure mechanisms or implementation bugs; it cannot guarantee that one cause
explains all webcam errors. Unresolved alternatives must remain explicit.

## Completed results

All planned offline comparisons and six prompted live trials completed. The user
confirmed: "Completed all six; capture was normal." The window closed and released
the camera. Original source checkpoint and cache hashes were verified unchanged.
The full audit remains `stopped_by_user`, 214/836 shards; this investigation read
only the 94 selected train/validation examples and six selected public videos.
No new webcam media, landmark coordinates or embeddings were written to disk.
The two stills from the earlier, separately approved review were not copied.

**Conclusion:** there are demonstrated failures in robustness to pose observations
and temporal context. This is stronger than an unspecified distribution-gap
hypothesis, but is not proof of one universal cause for every earlier capture.
Most older live sessions contain only counts/scores, preventing reconstruction
of their precise failure mechanisms.

### Experiment 1: pipeline and cache integrity

- Normalized all 75 training / 19 validation raw landmark records and compared
  against the exact saved float16 feature representation: maximum difference
  **0.0 for every clip**.
- Confirmed no repeated video names or overlapping participant IDs across the
  three official subset partitions (75/19/60 clips, 24/4/11 signers). Only test
  index metadata was read for this identity check; test inputs were not scored.
- Reproduced original frozen-model predictions: 66/75 train, 15/19 validation.
- Runtime strict weight/config/label/source checks still pass. Neither cache
  corruption nor a simple live/offline feature conversion mismatch was found in
  these selected records. This does not validate landmark correctness.

### Experiment 2: isolate the final classification head

Kept the GRU frozen and fit fixed diagnostic classifiers to its final hidden
representations, using only training examples. Also compared linear classifiers
of aggregate pose, hand, or timing features. No production weights were updated.

| Diagnostic | Train / 75 | Validation / 19 | Confirmed live / 6 |
|---|---:|---:|---:|
| Existing GRU + existing linear head | 66 | 15 | 4 |
| Frozen GRU + new standardized linear classifier | 75 | 15 | 2 |
| Frozen GRU + new standardized 64-unit MLP head | 75 | 15 | 3 |
| Hand mean/std summary + linear classifier | 75 | 17 | 4 |
| Pose mean/std summary + linear classifier | 73 | 9 | 3 |
| Timing/hand-presence summary + linear classifier | 34 | 6 | 2 |

The small validation advantage of hand summaries did not transfer into a higher
live count. These are diagnostic fits, not independent evidence for selecting a
replacement system. Two new heads failing on the frozen representation weakens
the final-layer-only hypothesis; it does not prove every useful distinction has
been erased from that representation.

For failed HELLO #1, the NO-minus-HELLO logit margin was +4.369. The head's bias
contributed **-0.100** (favoring HELLO), whereas its interaction with the hidden
representation contributed about **+4.470**. Small residuals arise from AMP.
The five nearest training hidden representations by cosine distance all belong
to NO. Failed THANKYOU #5 also has five nearest NO examples. These observations
place the visible failure before/at the learned representation-to-score mapping,
not in label display or a constant NO bias.

### Experiment 3: confirmed matched live trials

Every variant used the same in-memory capture, with fresh tracking for mirrored
pixels. Actual retained FPS was 23.5–29.0; source FPS was 28.5–30.3. Severe earlier
FPS halving did not recur, though the sampler is not perfectly uniform. All
displayed probabilities are uncalibrated model scores.

| Capture | Performed | Original first guess | Original target score |
|---|---|---|---:|
| 1 | HELLO | NO 94.46% | 1.20% |
| 2 | THANKYOU | THANKYOU 91.86% | 91.86% |
| 3 | NO | NO 95.44% | 95.44% |
| 4 | HELLO | HELLO 46.09% | 46.09% |
| 5 | THANKYOU | NO 95.60% | 2.62% |
| 6 | NO | NO 94.81% | 94.81% |

This establishes variable live success, not an always-NO output. Six prompted
attempts by one signer are not a webcam-accuracy benchmark.

**Observed temporal failure in THANKYOU #5:** processing the first 25%, 50%, 75%
of the capture gives THANKYOU scores **70.7%, 87.9%, 78.2%** respectively. The
complete capture instead gives THANKYOU **2.62%** and NO **95.60%**. Thus useful
recognition evidence exists earlier and is overturned by subsequent input to
the final-state GRU. This is a concrete sequence-context failure for this trial;
it does not distinguish genuine pose variation from inaccurate late tracking.

Removing only local hand-depth channels from this exact capture changes its
first guess to THANKYOU **62.4%**. Across all six trials this control yields 5/6,
versus 4/6 original; it retains the original 66/75 and 15/19 offline counts, while
some individual predictions change. It does not correct HELLO #1. These depth
estimates are not demonstrated outliers against training ranges, so the result
is evidence of a harmful learned dependence in this case, not proof of bad z
coordinates or a validated depth-removal fix.

**Ending-context control:** appending 16 training-derived no-hand rest frames
changes HELLO #1 from NO to HELLO 80.9%. Appending 32 gives HELLO 91.2%. However,
32 rest frames also change both correctly recognized NO trials into HELLO, and
the failed THANKYOU becomes HELLO too. The rest manipulation therefore induces
a HELLO bias and is unsafe as an application fix. It demonstrates context
sensitivity, not a reliable correction. No fake frames were added to normal
recognition or to training data.

**Orientation/assignment controls:** live captures contain no adjacent
single-hand slot switches. Swapping slots gives 0/6 correct; reflecting anatomy
1/6; re-extracting mirrored pixels 2/6. A simple orientation/left-right switch
is not supported as a general fix. In contrast, 18/75 training and 5/19 validation
clips have slot-change flags; these remain heuristic quality flags, not confirmed
annotation errors. No cleaning was performed.

### Experiment 4: timing and visibility shift

Training HELLO/THANKYOU/NO median durations are 2.25/2.72/2.42 s. Training median
detected-hand coverage is 39.84%, whereas these six live captures have median
coverage **97.92%** (range 77.1–100%). Detected hands may include resting hands;
coverage is not movement duration or landmark accuracy.

Training-derived target-blind median pre/hand-span/post lengths are 12/31/26
frames. Phase matching preserves the 15/19 validation count, but cannot apply
to any of these six captures: all lack at least one detector-defined rest phase.
The tool explicitly leaves unsupported cases unchanged instead of inventing
neutral positions. This comparison therefore does not establish a live fix.

Simple uniform resampling does not solve the failures: 64 frames gives 2/6;
128 gives 4/6. Temporal reversal gives 3/6, shuffling 2/6. These destructive
controls support temporal sensitivity, not a claim that gestures should be
reversed or shuffled. Full results for all 24 base controls, two phase controls
and mirrored-pixel extraction are in the evidence JSON.

### Experiment 5: reproduce a camera-scale failure on public video

Selected the first two validation-index clips of each HELLO/THANKYOU/NO,
regardless of their predictions. Each was fully decoded and freshly extracted
under original pixels, 65% person size in a black-padded canvas of identical
dimensions, and half-resolution-downsample-then-upsample conditions: **18 full
frame-to-MediaPipe-to-GRU passes**. No webcam was opened for this experiment.

Original recognition is 4/6; reduced-person-scale 5/6; half-resolution 4/6.
Scale is not uniformly harmful: it corrects two NO errors while breaking a
HELLO. These aggregate counts must not be represented as a general accuracy
improvement from shrinking people.

The HELLO failure is reproducible on `05614933522741539-HELLO.mp4`:

| Same gesture / frozen model | First guess | HELLO score |
|---|---|---:|
| Original video | HELLO | 55.42% |
| Person scaled to 65%, fresh MediaPipe | NO | 8.19% |
| Original landmarks under exact known coordinate scaling | HELLO | 55.42% |
| Scaled fresh features with original pose restored | HELLO | 46.90% |
| Scaled fresh features with only original elbow/wrist pose groups restored | HELLO | 44.18% |
| Scaled fresh features with all original hand groups restored | NO | 10.15% |

The exact coordinate transform changes normalized features by at most
**1.19e-6**, showing that the mathematical scale normalization works for this
case. Fresh MediaPipe estimation instead changes normalized pose coordinates and
**124 elbow/wrist validity-mask entries** across the video; selected face and
shoulder mask entries do not change. Restoring pose coordinates alone or masks
alone is insufficient; restoring them together recovers the original top label.
This locates a specific reproducible failure in sensitivity to pose estimates
and their validity masks. Neither detector output is ground-truth joints, and
restoring public-video features is a diagnostic transplant, not a webcam fix.

## What is established, and what remains unresolved

Established mechanisms:

1. The learned recognizer is sensitive to pose estimation/validity changes caused
   by an innocuous image-scale transform, even with correct scale normalization.
2. Its final recurrent state can overturn a previously correct sign decision
   after later frames; local hand-depth features contribute to one confirmed
   failed THANKYOU decision.
3. Training and live hand-visibility/rest context differ substantially. Artificial
   rest padding can bias the model toward HELLO, confirming it is not a safe fix.
4. Replacing the final head, mirroring, slot swapping or uniform sequence-length
   changes do not resolve these six trials consistently.

Not established: one exclusive cause of all historical webcam errors, ground-truth
joint accuracy, whether each depth variation is a tracking error, or a validated
repair. A 75-example training set offers only 14–16 clips per class; improving
robustness requires measured generalization, not confidence in a sampled skeleton.

The next justified repair comparison is an isolated model variant that reduces
fragile depth/pose dependencies and/or aggregates evidence over the sequence
instead of relying exclusively on the last hidden state. Any new variant needs
matched fresh evaluation; these diagnostic captures must not become training
data, and repeatedly choosing changes from their scores would overfit the pilot.
No production variant has been trained or substituted in this investigation.

## Artifacts and verification

- [Complete aggregate evidence JSON](assets/recognition-forensics-2026-11-11.json)
- [Diagnostic graphs](assets/recognition-forensics-2026-11-11.png)
- Local scratch bank and individual execution reports:
  `.cache/recognition-forensics-20261111-v1/`. The bank contains only public
  training features/embeddings and fitted diagnostic heads, never live arrays.
- Source/cache hashes unchanged; no source deletions, training-data edits or
  production checkpoint changes. Corpus audit remains stopped.
- **90 project tests passed**, plus source compilation, lockfile and dependency
  checks. Tests include bilateral geometry/mask consistency, transformation
  non-mutation, phase resampling boundaries, and exactly six automatic captures
  with no media/coordinate exports and guaranteed camera release on completion.

Additional reproduction commands, each requiring a new output filename:

```powershell
.\.venv\Scripts\python.exe scripts/investigate_phase_timing.py --bank .cache/recognition-forensics-20261111-v1/offline-bank.pkl --output .cache/recognition-forensics-20261111-v1/phase-report.json
.\.venv\Scripts\python.exe scripts/diagnose_five_sign_live.py --bank .cache/recognition-forensics-20261111-v1/offline-bank.pkl --report .cache/recognition-forensics-20261111-v1/live-report.json
.\.venv\Scripts\python.exe scripts/investigate_camera_conditions.py --output .cache/recognition-forensics-20261111-v1/camera-report.json
.\.venv\Scripts\python.exe scripts/isolate_scale_failure.py --camera-report .cache/recognition-forensics-20261111-v1/camera-report.json --output .cache/recognition-forensics-20261111-v1/scale-isolation-detailed.json
.\.venv\Scripts\python.exe scripts/report_recognition_forensics.py --source .cache/recognition-forensics-20261111-v1
```
