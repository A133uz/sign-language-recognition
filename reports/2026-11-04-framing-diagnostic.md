# Webcam framing investigation — 2026-11-04

The ASL Citizen guided session completed all 15 randomized prompts on CUDA with
the native twelve-label vocabulary. It scored 5/15 top-one and 10/15 top-three
prompt agreement. HELLO matched 2/3, THANKYOU 1/3, YES 1/3, NO 0/3 and ME 1/3.
The camera returned 480x640 BGR frames; captures contained 60–62 frames over
2.000–2.047 seconds. Median model inference latency was 437.1 ms, range
304.9–557.7 ms. These are separate attempts from the WLASL run, with a different
native candidate set; this is not a controlled accuracy comparison. The session
does not demonstrate an overall webcam improvement, so no default was changed.

Report (ignored): `runs/webcam_citizen_20261104_01.json`, SHA-256
`fec349fdeced4d7823a90a2f3f69fa0c71f7748eb0926e1244692af62b03884e`.

## Crop measurements

The user then completed one WLASL prompt each for yes, hello, no and me using
`--quality`. All four captures completed. The default guess matched hello only;
yes and me produced thank you, and no produced hello. No raw video or landmark
coordinates were saved. Measurements below describe 32 MediaPipe samples per
capture and all detected hands, rather than identifying an active signing hand.

| Prompt | Frames with hands | Detected hands wholly inside crop | Median hand short side | Shoulder span | Nose y |
| --- | --- | --- | --- | --- | --- |
| yes | 32/32 | 100% | 36.1 px | 123.6 px | 94.2 px |
| hello | 32/32 | 96.9% | 37.0 px | 121.8 px | 93.1 px |
| no | 29/32 | 100% | 32.2 px | 123.6 px | 93.1 px |
| me | 26/32 | 100% | 35.7 px | 120.7 px | 91.0 px |

The 224x224 model crop therefore contained the detected hands, with sizes
comparable to the dataset. This does not measure blur, label correctness or hand
detail that MediaPipe failed to detect. A simple missing-hand/crop explanation
is not supported by these detected samples. Body framing differed: across the
80 training clips, median nose y was 56.63 (10th–90th percentile of per-clip
medians 45.76–69.22), and median shoulder span 103.03 (86.26–115.37).
This is a hypothesis for an alignment experiment, not proof of the error cause.

Report (ignored): `runs/webcam_view_quality_20261104_01.json`, SHA-256
`1bcf219f86291942074da6454be052f492cc827a70f0423276095d441688c8f2`.
The geometry implementation matched existing prepared-feature measurements on
saved yes/no/me clips with real CPU MediaPipe, within 0.001 pixel for the tested
medians. Generated details: `runs/view_quality_runtime_check_20261104.json` and
`runs/wlasl_model_view_reference_20261104.json`.

## Fixed body alignment experiment

The optional transform maps the median nose location and shoulder span to a
reference fitted on training clips only. It applies one isotropic scale and
translation to every frame of the capture, preserving the sign's motion. It
does not rotate hands, alter labels, update model weights or use a hand-coded
sign rule. Missing pose, extreme zoom or excessive padding rejects alignment
and leaves the original predictions available, with the missing variant counted.

The package reference builder verified all 80 source training video hashes and
reproduced the initial prototype reference exactly. Calibration is a local,
ignored artifact: `assets/models/wlasl_framing_reference_verified.json`, SHA-256
`0a6d0bda0e1e12c200784d3ebd3a4eaabdbde4ac522454a4f582f0de8b98eb58`.
Its source feature hash is
`cd0d2dec59961cd62d7bd0f4a92ae4069e5f326c1de6f3b8a89395d75e9b8909`.
Target nose is (116.39, 56.63), shoulder span 103.03 in the 224-pixel model view.

On the same 21 validation clips, the original predictions matched 20 labels;
the aligned predictions matched 21. A synthetic low-face/large-shoulder condition
inspired by the live measurements matched 19/21. The synthetic condition changed
one no prediction to yes and retained the existing good/thank-you error. Thus
that perturbation alone did not reproduce all the live failures. Small sample
size, repeated validation inspection, original signer overlap and possible
pretrained exposure prevent treating these counts as independent accuracy.
Generated report: `runs/body_alignment_validation_20261104.json`.

A complete guided-path replay of a saved yes clip used actual CUDA and MediaPipe,
produced both original and aligned predictions, released its simulated camera,
and saved aggregate metrics only. This is integration evidence, not webcam
recognition proof. Report: `runs/guided_alignment_replay_20261104.json`.
Local verification passed 88 tests, lint, formatting, all vocabulary checks,
base diagnostics and Git whitespace checks.

## Human alignment outcome

The four-prompt comparison completed on CUDA. All four alignments were available;
original and aligned predictions each matched only hello (1/4 top-one, 1/4
top-three). Original guesses for me, yes and no were thank you, goodbye and hello;
aligned guesses were thank you, hello and fine. Relative zoom was 0.94–1.05 and
padding 19–24%. This later capture had nose y around 118–122, so its framing was
different from the preceding diagnostic; each comparison still used the same
capture for original and aligned predictions. The transform did not resolve
these live errors and is not selected as a fix.

Generated report (ignored): `runs/webcam_body_alignment_20261104_01.json`, SHA-256
`b988d70e258f98cbf8dd98e507c20844db5aef2182b57031cddf63ff4eb856dd`.
The dataset improvement therefore does not transfer to these observed webcam
attempts. The next investigation is stronger hand-shape modeling using more
public examples and signer variation, rather than claiming framing solves it.

## Data for the next model

The bounded native ASL Citizen acquisition completed 369/369 clips with zero
download failures and source ZIP CRC verification. It retained the original
177 training / 46 validation / 146 test memberships. The selected signer-ID sets
are disjoint (28/5/11 IDs); this is not independent identity verification.
Transfer reservation was about 205.8 MiB including the archive index and bounded
range overhead; ten existing probe files were copied after verifying CRCs.

All 369 files subsequently decoded to their declared frame counts. One training
YES clip has 664 frames, beyond the current 600-frame preparation bound, and
requires explicit handling rather than silent truncation. The first inventory
reported it under a preparation-limit error; the full inventory supersedes that
result after sequentially decoding all 664 frames. Availability and frame counts
do not certify sign labels or visual quality. No new native feature/model
training has been implemented or claimed at this point. Generated reports:
`runs/aslcitizen_pilot_acquisition_20261104.json` and
`runs/aslcitizen_pilot_inventory_full_20261104.json`.
