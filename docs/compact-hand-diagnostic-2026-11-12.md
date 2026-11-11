# Compact hand shapes: NO and YES observation

## User feedback and scope

In the interactive mean-pooling session the user reported matching first guesses
for HELLO, THANKYOU and PLEASE, incorrect guesses for NO and YES, and overlapping
hand points for the latter signs, whose ending skeletons looked fist-like.
This is user-reported visual evidence, not an assistant-reviewed image or a
confirmed tracking failure. No new images or landmark coordinates were captured.

The session `.cache/webcam-mean-pool-20261112-01.json` identifies SignMeanPoolGRU,
epoch 4, SHA256
`909e90daffb8d244c7492d1303ab8dfeeddd8b325c0335ad74207711385466c9`.
At the first feedback inspection it contained ten logged captures and remained
open. Intended labels/capture-number mappings were not supplied, so do not assign
the user's sign descriptions to particular capture IDs or calculate formal live
accuracy from this feedback. Log predictions are compatible with the reported
successful labels, but do not establish the mapping independently.

## Preview versus classifier input

`hands.py:draw_hands` draws only x/y image positions, with an outer dot radius of
four pixels. Points separated by several pixels can visually overlap. The
recognizer observes a separate fresh tracker pass after capture, so the preview
does not verify the exact per-frame joint estimates used for recognition.
The mean-pool GRU receives normalized pose/hand geometry including wrist-relative
hand xyz; the preview does not show depth. Projected overlap is not equivalent
to identical xyz features, and compact/folded hand anatomy can legitimately
place several landmarks close together. Estimated z is not ground-truth depth.

ASL YES normally uses an S/fist handshape and wrist motion; a fist-like ending
alone is therefore not evidence of a bad estimate. NO must be considered over
its movement, not only the ending.

References: [ASL University YES](https://www.lifeprint.com/asl101/pages-signs/y/yes.htm),
[ASL University NO](https://www.lifeprint.com/asl101/pages-signs/n/no.htm),
[MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker).

## Read-only dataset feature check

Ran `scripts/inspect_compact_hand_shapes.py` on existing prepared training and
validation features for HELLO/NO/YES: 45 training and 12 validation clips. No
test examples, training, webcam capture or source edits were involved.
For the dominant canonical hand slot, measured thumb-tip/index-tip separation
and its within-clip p90-minus-p10 range over frames with valid local hand shape.
Coordinates are palm-normalized; measurements are not pixels or metres.

| Median across clips | Train NO | Train YES | Val NO | Val YES |
|---|---:|---:|---:|---:|
| Median thumb/index XY separation | 0.383 | 0.250 | 0.442 | 0.191 |
| Within-clip thumb/index XY p90-minus-p10 range | 1.410 | 0.269 | 1.105 | 0.335 |

This shows that the cached examples are not universally identical closed-hand
representations: their estimated finger separation varies differently over time.
It does not establish real-joint accuracy, semantic separability of every clip,
or preservation of these distinctions in the user's webcam input. Detected rest,
finite sampling, hand-slot errors and perspective can affect these summaries.

The next informative comparison is the same scalar finger-opening and wrist-
movement measurements on prompted live NO/YES captures, alongside enlarged hand
views if separately authorized. That would distinguish plausible projected
overlap from lost articulation more directly than another classifier swap.
Max pooling can retain brief learned responses but cannot reconstruct missing
or inaccurate landmark information; no new pooling/model change was made here.

[Full scalar feature-check evidence](assets/compact-hand-feature-check-2026-11-12.json).

Reproduce with a new output path:

```powershell
.\.venv\Scripts\python.exe scripts/inspect_compact_hand_shapes.py --output docs/assets/compact-hand-feature-check-2026-11-12.json
```

## Focused live diagnostic prepared and opened

The user requested another diagnostic. Added a scalar articulation helper shared
by a reference-preparation command and the actual live inference path. This
measures thumb/index and thumb/middle separation in xy/xyz, finger PIP angles,
index/middle tip-to-wrist distance, estimated palm-axis excursions, body-relative
wrist excursion, usable-frame counts and slot-switch flags. Comparisons use
training-only per-class scalar ranges; validation records are retained as
reference evidence but do not set thresholds or select a model.

The xyz palm-axis measurement is an estimated orientation proxy, not a physical
wrist angle. Orientation changes are not bridged across detection gaps. Real
detected resting hands remain included, consistently with the reference. The
detector and both recognition models are unchanged.

All **100 project tests passed** before launch, including synthetic projected
overlap with retained depth, opening versus closed-hand measurements, gap-aware
motion calculations, non-mutation, and exactly six automatically ended trials
with scalar-only reports. Reference: 45 train and 12 validation examples at
`docs/assets/hand-articulation-reference-2026-11-12.json`.

Opened `.cache/compact-hand-check-20261112-01.json` with prompts **NO, YES, HELLO**
twice. HELLO is a control. The actual ready messages identify SignGRU epoch 18
and SignMeanPoolGRU epoch 4; both score identical observations from the same fresh
tracker pass. Each capture is automatic four seconds; the window closes after
six trials. Requested completion confirmation; no results assumed yet.

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_five_sign_live.py --bank .cache/recognition-forensics-20261111-v1/offline-bank.pkl --candidate-checkpoint models/asl-gru-five-signs-mean-pool-v1/best.pt --compact-reference docs/assets/hand-articulation-reference-2026-11-12.json --report .cache/compact-hand-check-20261112-01.json
```

## Completed focused diagnostic

The user confirmed **"Completed all six; capture was normal."** The session
contains six successful four-second captures and closed its camera normally.
Both checkpoint hashes remain unchanged. No model, preprocessing, training data
or default checkpoint was changed. No additional media/coordinates were saved.

| Capture | Prompt | Original GRU first guess | Mean-pool GRU first guess |
|---|---|---|---|
| 1 | NO | NO 92.2% | NO 67.7% |
| 2 | YES | NO 81.8% | NO 57.7% |
| 3 | HELLO | NO 92.1% | HELLO 88.6% |
| 4 | NO | NO 94.4% | NO 70.8% |
| 5 | YES | NO 61.2% | NO 58.4% |
| 6 | HELLO | HELLO 75.4% | HELLO 93.6% |

Original: **3/6**; pooled: **4/6**. These are same-capture comparisons for this
session, not general live accuracy or grounds to replace the default. In the
previous paired session, the pooled model missed NO; it recognizes both NO
attempts here. Capture-specific behavior and the small sample remain important.

### Finger opening is preserved in the measured live features

Within-clip thumb/index XY separation range is p90 minus p10 in normalized palm
units, across frames with valid local shape in the dominant canonical slot:

| Sign | Live attempt 1 | Live attempt 2 | Training median | Training p10–p90 |
|---|---:|---:|---:|---:|
| NO | 1.294 | 1.236 | 1.410 | 0.792–1.745 |
| YES | 0.263 | 0.310 | 0.269 | 0.241–0.674 |

The numerical opening-range distinction survives extraction and normalization
in this session. It contradicts a universal explanation that these inputs are
identical collapsed fists. It does not prove full joint accuracy or that every
distinction required for recognition is preserved.

YES index-tip-to-wrist medians are **1.100 / 1.024**, within its training central
range **0.877–1.406**. Estimated xyz palm-axis excursions are **9.18° / 26.72°**;
training median is 16.45°, p10–p90 9.40–60.78°. The first is near the low end
(13th empirical percentile among 15 training clips), and the second is inside
the central range. Thus some key shape/motion measurements are dataset-like
even though both recognition models predict NO. These marginal distributions
do not establish that the joint multivariate sequence is in-distribution.

### Remaining limitations

- Features include detected resting hands and up to 128 sampled frames, as in
  the reference. Timing, handshape and motion need not occupy matching phases.
- Slot-switch flags are 1/1/3/0/0/2 across captures 1–6; they are review flags,
  not independently confirmed physical-hand swaps. The dominant-slot summary
  can omit observations assigned elsewhere, although useful separation remains.
- Compact/occluded joints can be inaccurate while aggregate distances look
  plausible. No images or raw joint sequences are available for checking these
  particular observations against real anatomy.
- Mean-pool YES predictions remain NO after artificial rest controls. Original
  YES predictions become YES after appended rest, but adding 32 rest frames also
  changes correctly recognized original NO attempts to YES. Rest padding is
  therefore still not a safe repair.

The strongest updated conclusion is **difficulty using/generalizing available
shape and motion evidence**, rather than proven total loss of finger articulation.
Tracking noise, pose/visibility context, feature coupling and temporal learning
remain possible contributors. Do not claim MediaPipe is fully correct, that one
root cause is isolated, or that changing pooling alone resolves recognition.

### Evidence and report reproduction

- [Complete scalar reference and confirmed live report](assets/compact-hand-live-2026-11-12.json)
- [Reference distributions and paired predictions](assets/compact-hand-live-2026-11-12.png)
- **100 tests** passed before capture, including this capture mode and scalar
  measurement edge cases. The reporting command executes without retraining or
  rescoring any model and contains no raw landmark/video export.

```powershell
.\.venv\Scripts\python.exe scripts/report_compact_hand_check.py --report .cache/compact-hand-check-20261112-01.json --reference docs/assets/hand-articulation-reference-2026-11-12.json
```
