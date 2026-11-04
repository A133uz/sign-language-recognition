# Finger/body geometry experiment — 2026-11-05

The timing candidate's same-capture human test matched 9/15 prompts, versus
7/15 for the original, but ME matched only 1/3 and THANKYOU 0/3. The existing
world summary represents the wrist relative to the nose, but not each fingertip
relative to the body. This experiment retains that location information and
uses derived finger shape rather than camera-oriented world coordinates.
No personal webcam samples are retained or used for training.

## Shared representation

`body_geometry` derives 149 values per sampled frame from the same 288-value
raw archive, without a new extractor or schema:

- 84 image values: both hands' 21 xy points, centered on the nose and divided by
  shoulder width. Missing hands and insufficient visible body landmarks mask
  these offsets to zero.
- 60 world-hand shape values: fifteen adjacent-joint cosines, five fingertip
  distances to the wrist and ten inter-fingertip distances per hand. Raw world
  coordinates are already wrist-centered and palm-size normalized. Missing
  world hands mask these values to zero.
- Five validity values: two image-hand masks, two world-hand masks and one body
  mask (visible nose/shoulders and sufficient shoulder width).

Four temporal-bin means and whole-sequence standard deviation produce 745
SVM inputs. This preserves temporal order coarsely and discards camera-oriented
world vectors. Image/body offsets still depend on viewpoint, tracking accuracy
and frame aspect ratio. Derived angles do not recover occluded fingers or make
the classifier an expert judge of sign correctness.

## Training-only candidate selection

Four predefined candidates used the same three-fold, signer-grouped training
splits as the timing probe (seed 42), with train-fold-only scaler fitting and
linear SVMs (C=1, balanced weights). Selection ranked out-of-fold macro accuracy
then total matches. No source test predictions or scores were computed.

| Candidate | Training-signer CV | Macro accuracy | ME |
| --- | --- | --- | --- |
| Existing world summary + timing | 133/173 | 0.766 | 9/11 |
| Body geometry + timing | 140/173 | 0.81042 | 11/11 |
| World + geometry, no timing | 130/173 | 0.752 | 8/11 |
| World + geometry + timing | 140/173 | 0.80989 | 10/11 |

The chosen geometry-only candidate was fitted on all training examples with
the five timing versions (865 sequences). On the existing validation partition
it scored 39/46 (macro 0.819), versus the original 40/46 (0.854): HELLO 4/4,
THANKYOU 1/3, BYE 1/3, PLEASE 4/4, SORRY 3/3, YES 5/5, NO 3/3, HELP 5/5,
FINE1 3/4, FINE2 2/3, GOOD 3/4 and ME 5/5. BYE remains a regression. These are
small, repeatedly used partitions; training-CV selection is exploratory and
does not establish independent webcam accuracy. The original remains frozen.

Source report: `runs/geometry_probe_20261105.json` (ignored).

```powershell
.\.venv\Scripts\python.exe -m sign_language train data/processed/aslcitizen_world_20261104.npz --method svm --representation body_geometry --temporal-augmentation --skip-test --device cpu --output checkpoints/aslcitizen_geometry_linear_20261105.pt
.\.venv\Scripts\python.exe -m sign_language test-webcam --backend landmarks --weights checkpoints/aslcitizen_world_linear_20261104.pt --compare-weights checkpoints/aslcitizen_geometry_linear_20261105.pt --device cpu --sign HELLO --sign THANKYOU --sign YES --sign NO --sign ME --repeats 2 --seed 44 --output runs/webcam_geometry_comparison_20261105_01.json
```

New output paths are required. Checkpoint SHA-256:
`8459cb39192abe56ba67666f566d8876867f055c8787523b17759167bfb5b42d`.
The native twelve-class order is preserved; no label merging or exclusion is used.

## Runtime checks

The production Torch transform matched the NumPy selection-probe transform on
all 219 training/validation sequences, maximum absolute difference 3.58e-7.
Real CPU MediaPipe extraction followed by both classifiers matched archived-input
predictions on three public validation clips (HELLO, THANKYOU and ME). Both
still predicted GOOD for that THANKYOU clip. The full guided saved-YES replay
completed, produced both variants and released its simulated camera. Reports:
`runs/geometry_runtime_parity_20261105.json` and
`runs/guided_geometry_replay_20261105.json` (ignored).

All 104 tests and the full local check script passed. Geometry tests cover absent
or degenerate tracking, image translation/scale invariance, world rotation
invariance and retained fingertip location. These are implementation checks,
not recognition validation. No dependency or environment changes were made.

## Same-capture human result: rejected as a replacement

All ten trials completed, with both models predicting every capture. The user
confirmed following the prompts with normal capture. Each capture contained
60–62 frames at 640x480, with hands in 22–32 of 32 sampled frames. No trial was
rejected. Shared extraction and both classifiers took median 1999 ms, range
1914–5164 ms after each two-second capture, not continuous recognition FPS.

| Prompt | Original top-one | Geometry candidate top-one | Original top-three | Candidate top-three |
| --- | --- | --- | --- | --- |
| HELLO | 2/2 | 2/2 | 2/2 | 2/2 |
| THANKYOU | 1/2 | 0/2 | 2/2 | 0/2 |
| YES | 1/2 | 0/2 | 1/2 | 0/2 |
| NO | 1/2 | 0/2 | 2/2 | 1/2 |
| ME | 1/2 | 1/2 | 1/2 | 2/2 |
| Total | 6/10 | 3/10 | 8/10 | 5/10 |

The geometry candidate is rejected as a replacement. Its training-signer CV
improvement did not transfer to this session, and it regressed on YES, NO and
THANKYOU. Discarding world orientation while adding image/body offsets is not
a validated fix. The original weights and defaults are preserved. The failed
candidate and implementation remain available for reproducible diagnostics;
they are not promoted to normal webcam use. The goal remains incomplete.

Report: `runs/webcam_geometry_comparison_20261105_01.json` (ignored), SHA-256
`502293db5fcfba3d3e215c3a09992b4d19bd567dae1e342e4c1b13e99fa39ab6`.
Only predictions, timing, dimensions/counts and aggregate tracking counts were
saved; no personal video, coordinates or embeddings were retained or used for
training. The process exited normally and released the camera.

There were two displayed top-score ties for the original and four for the
candidate. Vote ties follow native label order; this is prompt agreement with
a small sample, not independent or calibrated accuracy. Remaining native labels
and unknown-sign rejection were not checked in this session.
