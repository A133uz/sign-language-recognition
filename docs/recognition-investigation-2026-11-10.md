# Webcam recognition investigation — 2026-11-10

This is an investigation record: actions, reproducible experiments, evidence,
decisions and limitations. Proposed explanations remain hypotheses until a
check supports them. No training, checkpoint replacement or vocabulary
restriction is authorized as part of this diagnosis.

**Current conclusion:** the evidence points to a training/webcam input gap,
especially detected-hand/rest context, with additional tracking and image-size
problems. The current GRU is demonstrably sensitive to those changes. A wrong
checkpoint, label order or shared-preprocessing mismatch was not found. The
experiments do not identify one exclusive cause or establish a working fix.

## Reported symptom and preserved baseline

The user reported THANKYOU → FROWN1, HELLO → NONSENSE and NO → THUNDER in
the first full-vocabulary webcam test. These are qualitative observations,
not an exhaustively labeled evaluation. The session's saved predictions do
not identify the intended label for every attempt.

- Selected model: `models/asl-gru-128-cv-regularized-v1/official-refit/best.pt`,
  epoch 37, two-layer 256-unit GRU, 2,731 labels.
- Expected SHA-256:
  `6e09a1b3c62ea364e2a86b5551eaefe71651332f75fb64a060ed184de7032b39`.
- The archived weights and original training/evaluation artifacts must remain
  unchanged. New machine-readable diagnostic outputs go under `.cache/`.
- The working tree already contains the newly implemented webcam recognizer,
  its tests and documentation. Those are preserved during this investigation.

## Questions to distinguish

1. Are the wrong weights, label order, detector assets or preprocessing used?
2. Does a fresh video-to-prediction pass agree with the cached evaluation path?
3. Are frame capture, tracking, sign boundaries or camera geometry damaging inputs?
4. How much does performance vary across dataset signers and these three labels?
5. Is the last-hidden-state GRU sensitive to timing, idle frames, framing or
   handedness changes that the current augmentation does not address?

## Step 1 — inspect implementation and existing evidence

Actions: inspect Git status; read `recognition.py`, `recognition_webcam.py`,
the training epoch/inference code, the prepared-cache identity, the prior
webcam guide and session report. Read project design memory as historical
context, while treating current code/artifacts as authoritative.

Observations:

- The current recognizer explicitly restores `state['model']` with strict
  loading, checks preprocessing/model source hashes and adjacent config/labels,
  and uses `eval()` plus inference mode. The initial implementation's missing
  weight-loading call was corrected **before** the user webcam session; its
  checkpoint-restoration test and recorded REGISTER smoke check passed.
- Both training/evaluation and webcam classification use CUDA AMP and cuDNN
  disabled. MediaPipe inputs are unmirrored; only the UI preview is mirrored.
- Captures are manually bounded, and the GRU classifies its last valid hidden
  state. There is no automatic active-sign trimming or temporal augmentation
  in the selected training profile.
- The earlier REGISTER training-video smoke check matched the prepared-cache
  prediction exactly. This checks one example, not all labels or live inputs.
- The first session saved predictions/tracking counts/timing only. Frames and
  landmark arrays were deliberately discarded, so past failed features cannot
  be reconstructed or inspected from that report.
- A read of `.../raw.../manifest.json` failed because that filename does not
  exist. Locate the extractor's actual config/index files before checking
  detector provenance; do not infer the raw cache is missing.

Decision: test checkpoint/cache identity, batch inference and controlled input
perturbations before recommending another model or another training run.

## Experiment results

### Step 2 — reproducible offline audit

Command:

```powershell
.\.venv\Scripts\python.exe -u scripts/diagnose_recognition.py --output .cache/recognition-investigation-20261110-v1 --fresh-videos
```

Machine-readable results: `.cache/recognition-investigation-20261110-v1/audit.json`;
the chosen validation rows are saved in `validation-sample.json`. Console output
is in `.cache/recognition-investigation-20261110-v1.log`.

Checks completed:

- Correct epoch-37 checkpoint; raw/prepared/checkpoint labels and cache IDs
  agree. Both hand/pose task SHA-256 values match the raw extractor's
  `run-manifest.json`. MediaPipe, OpenCV and NumPy versions agree with extraction.
- For one HELLO, THANKYOU and NO video from **each** of train and validation,
  raw-NPZ normalization reproduces prepared float16 features bit-for-bit.
  Fresh video decoding/extraction reproduced **all six** top-five label lists
  and scores exactly (maximum score difference zero).
- Evaluated every training/validation example of the three signs. Training:
  HELLO 14/14, THANKYOU 16/16, NO 16/16 first guesses correct. Validation:
  HELLO 4/4, THANKYOU 2/3, NO 2/3. These small counts are descriptive, not
  evidence of reliable live recognition.
- Deterministic seed-42 sample of **512 validation clips**, all 2,731 outputs:

| Feature-level experiment | Top-1 vs original labels | First guesses changed |
| --- | ---: | ---: |
| Original | 60.35% | — |
| Repeat final frame for 25% extra time steps | 59.96% | 14.45% |
| Repeat initial frame for 25% extra time steps | 59.77% | 13.67% |
| Duplicate all time steps, then enforce 128 limit | 60.35% | 17.58% |
| Keep every second time step | 53.52% | 27.93% |
| Keep middle 80% of time steps | 58.20% | 17.19% |
| Horizontal reflection plus anatomical slot exchange | 38.87% | 58.98% |

The perturbations act on already-normalized features. They isolate sensitivity;
they do not recreate image-level detector behavior. Cropping can remove
sign-defining motion, and mirroring is not guaranteed to preserve every ASL
label. Therefore these figures are **not** claims of a universally correct
augmentation or measurements of the webcam root cause.

- FP32 vs AMP first guesses agree for 511/512 clips; mean absolute logit
  difference 0.00351. FP32 top-1 is 60.16%. Single-clip vs batched inference
  agrees on all 32 checked first guesses (max logit difference 0.015625).
  Ordinary AMP/batching differences do not explain the broad failure pattern.
- Existing test results already show signer variation: 30.40–69.39% top-1.
  No test inputs were used for perturbation experiments or model selection.

Decision: test actual image reflection and temporal windows on the **same
live captures**, with ground-truth prompts and aggregate feature measurements.
Orientation is a supported sensitivity, not yet an established cause. Small
static endpoint holds alone did not produce a comparable population-wide drop.

### Step 3 — prepare a controlled live diagnostic

The user approved two attempts each of HELLO, THANKYOU and NO. Added
`scripts/diagnose_webcam.py` and aggregate-only `recognition_diagnostics.py`.
Factored the existing extractor's in-memory observation step into
`SignRecognizer.observe_frames`, preserving detector settings and the selected
preprocessor/model. No preprocessing or trained weight files were changed.

Protocol: Space begins a 1.5-second preparation countdown; capture then lasts
three seconds automatically. Perform one sign during SIGN NOW and avoid the
keyboard until processing finishes. Every capture is scored with its full
2,731-label output under five predefined conditions: original; first 80%;
last 80%; middle 60%; actual horizontally reflected pixels re-extracted with
fresh trackers. The prompt's score/rank and top-five labels are retained.

Saved measurements include tracking fractions, aggregate shoulder/palm size,
feature/depth magnitude distributions, clipping and hidden-state saturation.
No webcam frames, landmark arrays or full feature vectors are saved. Prompt
compliance must be confirmed by the user before interpreting accuracy counts.
These variants are diagnostic comparisons, not a replacement inference policy.

Command (launched as a hidden console with a visible OpenCV test window):

```powershell
.\.venv\Scripts\python.exe -u scripts/diagnose_webcam.py --report .cache/webcam-diagnosis-20261110-01.json
```

### Step 4 — inspect the completed, confirmed live comparison

All six trials completed and the window closed normally. The user confirmed:
"Completed all six; capture was normal". That confirmation is recorded
separately in `.cache/recognition-investigation-20261110-v1/user-confirmation.json`;
the original session output remains unchanged.

Every condition produced **0/6 top-1 and 0/6 top-5 matches**: original,
first-80%, last-80%, middle-60% and mirrored-pixel inputs. These six trials of
one signer are a diagnostic result, not a population accuracy estimate.

| Trial/prompt | Original first guess | Target rank minimum | Frames with hands | Median palm size, pixels | Median shoulder span, fraction of image width |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 HELLO | PHILOSOPHY | 9 | 82.6% | 90.8 | 0.469 |
| 2 THANKYOU | STOMACH | 1683 | **0%** | No detections | 0.454 |
| 3 NO | HUMMINGBIRD | 490 | 100% | 43.1 | 0.242 |
| 4 HELLO | GRENADE | 56 | 98.9% | 39.5 | 0.236 |
| 5 THANKYOU | SANDWICH2 | 79 | 85.7% | 42.3 | 0.231 |
| 6 NO | CIRCLE | 214 | 91.2% | 62.7 | 0.308 |

Rank minimum counts logits strictly greater than the target logit plus one;
ties can give a lower bound on its position. The actual top-five membership
check uses the returned top-five labels.

Confirmed tracking issue: original trial 2 detected **zero hands in 92 frames**;
reflecting those same pixels recovered hands in **81.5%** of frames. It still
did not recognize THANKYOU. This shows orientation-dependent detector failure
for that capture, not a universal mirror fix. Diagnostic scoring deliberately
retains pose-only results to measure the failure; the normal recognizer's
existing zero-hand guard would reject this capture instead of accepting a sign.

Other failed trials had abundant hand detections. Neither complete dropout
nor mirroring explains all six. All frame shapes were 480×640, captures were
approximately three seconds, and no features hit the configured clipping limit.
Presence and finite ranges still do not establish correct fine finger geometry.

### Step 5 — quantify training/live context differences

Command:

```powershell
.\.venv\Scripts\python.exe -u scripts/diagnose_input_statistics.py --output .cache/recognition-investigation-20261110-v1/dataset-input-statistics.json
```

This read the raw landmarks for all **46 training and 10 validation** examples
of the three signs. It saved aggregate input statistics and frozen predictions,
not landmark arrays. Of the full official training split, 40,154/2,731 is
approximately **14.7 examples per class**; the large total dataset size should
not be mistaken for hundreds of training examples of each sign.

| Training sign | Examples | Median fraction of frames with hands | Mean hand presence in final quarter | Median palm size, pixels |
| --- | ---: | ---: | ---: | ---: |
| HELLO | 14 | 34.1% | 4.0% | 92.5 |
| THANKYOU | 16 | 35.2% | 4.5% | 76.0 |
| NO | 16 | 39.8% | 5.7% | 85.7 |

The mean training hand-presence pattern across clip quarters was:
HELLO **28.8%, 79.8%, 31.6%, 4.0%**;
THANKYOU **20.4%, 83.7%, 32.7%, 4.5%**;
NO **30.2%, 81.6%, 44.3%, 5.7%**.
By contrast, both live HELLO and both live NO trials had detected hands
throughout their final quarter. This is a measured difference in detector
output/context; absent detections do not establish whether a physical hand
was outside the image, occluded or simply missed.

Additional validation-only ablations were first run interactively and saved
to `endpoint-context-ablation.json`, then reproduced through this CLI:

```powershell
.\.venv\Scripts\python.exe -u scripts/diagnose_recognition.py --output .cache/recognition-investigation-20261110-context-v2 --context-variants
```

For each anatomical slot, extend its first/last valid hand-feature vector over
the previously missing leading/trailing frames. Keep body-pose features and
time-step counts unchanged. This is a **synthetic feature-level counterfactual**,
not a detector repair, an accepted augmentation, or a proposal to falsify live
missingness. It cannot recreate all physical hand/body relationships.

| Same 512 validation clips | Top-1 | Top-5 | First guesses changed |
| --- | ---: | ---: | ---: |
| Original | 60.35% | 85.94% | — |
| Keep last detected hand state visible through trailing gaps | **30.86%** | 53.12% | 63.48% |
| Keep first detected hand state visible through leading gaps | **38.87%** | 64.45% | 54.88% |
| Extend both ends | **16.02%** | 33.59% | 82.03% |

This is stronger evidence of **hand-visibility/rest-context sensitivity** than
the small static-frame holds in Step 2. It supports a mechanism for failure
when a new signer keeps a hand visible near clip boundaries. It does not
isolate missingness masks from the coordinate changes or prove that context
alone caused each live error. The last-hidden-state architecture and lack of
training coverage for these context changes are plausible contributors; a
different sequence head has not been trained or compared here.

### Step 6 — isolate smaller hand images on dataset videos

Command:

```powershell
.\.venv\Scripts\python.exe -u scripts/diagnose_video_scale.py --audit .cache/recognition-investigation-20261110-v1/audit.json --output .cache/recognition-investigation-20261110-v1/video-scale.json
```

For the six Step-2 videos, resize the entire image to 60% of its original
dimensions and center it in a black canvas of the original size. Re-extract
with fresh detectors and the same frozen recognizer. This changes pixel
detail and borders while preserving aspect ratio; it is not an exact webcam
simulation and has only six examples.

Native predictions were correct for **5/6**; smaller-image predictions for
**3/6**. Both native-correct THANKYOU examples became incorrect (INSPIRING and
GOOD, with THANKYOU ranked third and second). The already incorrect validation
NO example remained incorrect; it is not counted as a newly caused failure.
This supports sensitivity to reduced image detail/detection differences.

The later live trials' median palms were about **39–43 pixels**, compared
with about **76–93 pixels** for these labels' median training clips. Several
live shoulder spans were also much smaller. Ideal body normalization removes
uniform scale from exact coordinates; it cannot recover finger detail that
the image/detector did not resolve. The first HELLO trial had a normal-size
palm yet still failed, so image size is not a complete explanation.

## Conclusions, confidence and next decisions

| Finding | Evidence level | Consequence |
| --- | --- | --- |
| Weight/label/cache/preprocessor mismatch | Not found in identity checks and six exact fresh-video reproductions | No basis to replace the selected checkpoint or label map |
| Numeric precision/batch size | Negligible effect in tested controls | Switching off AMP is not a supported remedy |
| Hand-visibility/rest context | Large measured training/live gap and large validation ablation effect | Strongest supported model weakness and likely live contributor; not an exclusive root-cause proof |
| Hand tracking failure | Confirmed for one original THANKYOU trial | Some inputs are unusable despite a working camera; inspect detector quality per capture |
| Smaller hand images/framing | Measured in later live trials; controlled video probe caused two new errors | Additional supported contributor; correct body scaling alone cannot repair detector error |
| Mirroring or simple temporal crop as a fix | All five variants failed all six confirmed trials | Do not silently promote either as a solution |
| Generalization | 100% on these training examples, imperfect validation/test and failed new-signer trials | Dataset recognition metrics do not transfer directly to this webcam signer |

The exact contributions of gesture variation, occlusion, fine landmark accuracy
and context cannot be separated retrospectively: webcam images and landmarks
were not retained. No live frames were reviewed by the assistant. This is an
explicit limit of the diagnosis, rather than a claim that the user signed
incorrectly or a reason to keep changing models blindly.

Recommended order for a subsequent change, **not executed here**:

1. Define a repeatable complete-sign capture convention with a neutral start
   and finish, and camera framing that retains sufficient hand image detail.
   Validate quality from actual detector outputs; do not manufacture missing
   landmarks or assume a visible skeleton is accurate.
2. If further training is authorized, compare training coverage/augmentation
   for realistic start/end context, duration, safe handedness changes and
   camera scale. Preserve complete sign-defining motion. Compare a temporal
   pooling/attention head against the current last-state head as a controlled
   experiment; no improvement is assumed.
3. Test on prompted examples from new signers, alongside official validation.
   Keep the frozen reported test result and archived weights for comparison.
   Quality/unknown-sign rejection is a separate evaluation problem; suppressing
   guesses alone does not improve recognition accuracy.

## Durable evidence and verification

- [Compact machine-readable evidence](assets/recognition-investigation-2026-11-10.json)
  includes exact comparison counts, aggregate dataset/live measurements,
  numeric controls and SHA-256 values for the scripts and raw diagnostic
  reports used. Full local reports remain under the `.cache/` paths above.
- `scripts/check.ps1`: **50 tests passed** after adding the diagnostic path,
  including checkpoint restoration, frozen inference, aggregate-only output,
  observation preservation and hook cleanup. The model preprocessor and
  architecture source files were not changed.
- No training, checkpoint replacement, vocabulary reduction, video saving or
  landmark-coordinate saving occurred. Original model/archive hashes are
  checked again at completion. No automatic Git push was performed.

Final completion checks: `scripts/check.ps1` passed **50 tests** (3.324 seconds
for the unit-test portion); dependency imports and offline lockfile checks
passed. The selected checkpoint and 22,202,898-byte ZIP archive still match
their recorded SHA-256 values. Every file hash in the compact evidence record
and every local Markdown link checked successfully; `git diff --check` passed.
The six-trial diagnostic reported complete and the user closed its window.
All investigation source and documentation changes remain local for review.
