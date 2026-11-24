# Guided sliding-window activity diagnostic — 24 November 2026

## Protocol and scope

This follow-up evaluated raw ASL predictions, the previous 0.60-score and
two-consecutive-window candidate, and aggregate motion measurements on the same
windows. The checkpoint and production webcam defaults did not change. The
measurements are exploratory proxies, not a trained or calibrated activity
detector, and no thresholds are selected from this session.

The user confirmed normal capture and all 14 prompts. UNKNOWN trials #5 and #10
were both HOPE; the checkpoint's 15-label map does not include HOPE. Thus they
are two attempts of one unknown sign, not evidence across unknown signs. HOLD_MY
was an already formed MY pose held throughout capture, not a verified complete
isolated sign with annotated boundaries.

Each trial ran for about seven seconds; the instruction for sign attempts was
to sign in the first three seconds and relax. Windows were two seconds long,
sampled about every half-second; they overlap strongly. The first and last
portions of a trial may contain preparation, relaxation, or transitions. Actual
sign boundaries were not annotated. A prompt match at any time is not proof of
correct sign localization. Trials #3 and #9 HELLO had 3 and 4 unscored windows;
#6 REST had 10, #8 MOVE had 6, and UNKNOWN #5 and #10 had one each. The
diagnostic caught a ValueError without saving its category, so the actual
reason for any individual unscored window cannot be established from this log.
Unscored windows are missing measurements, not successful rejections.

The motion measurements are the 75th percentile of valid consecutive-frame
speeds in each window: wrist displacement in shoulder-widths per second, and
mean finger-joint displacement relative to the wrist in palm-widths per second.
Only consecutive detections and time gaps <=0.25 seconds contribute. The input
uses the existing per-window normalization and hand-slot assignment. Detector
jitter, assignment changes, occlusion, signer movement, and short intervals can
affect these measurements. They are not verified anatomical velocities.

The original local JSON `.cache/sliding-activity-guided-20261124-v1.json` has
SHA-256 `d983487756426a609453f73a1062666e1117509e99060fbd6aeb2041064cdef7`.
A separate local confirmation file records the user's response and the hash.
Only labels, scores, timestamps, frame counts and aggregate speed/coverage
values were persisted; video, landmark coordinates and embeddings were not.

## Findings

Fourteen completed trials produced 115 scored and 25 unscored windows.
The following counts concern overlapping scored windows, not independent trials
or continuous output durations.

| Condition | Raw outputs | Threshold + stability outputs | Unscored windows |
|---|---:|---:|---:|
| REST #1, #6, #14 | 20/20 scored | 0/20 scored | 10 |
| Unrelated MOVE #2, #8 | 14/14 scored | 2/14 scored | 6 |
| HOPE #5, #10 | 18/18 scored, forced known labels | 0/18 scored | 2 |

The two MOVE false outputs were YOU and ME. The zero filtered rest and HOPE
outputs in this session do not establish reliable rejection: the previous guided
session produced high-scoring repeated false outputs at rest. One unknown sign
and one signer cannot support a general open-set claim.

| Attempt | Raw windows matching prompted label | Threshold + stability matching |
|---|---:|---:|
| HELLO #3 | 6/7 | 0/7 |
| HELLO #9 | 3/6 | 0/6 |
| YES #4 | 9/10 | 8/10 |
| YES #11 | 10/10 | 9/10 |
| THANKYOU #7 | 0/10 | 0/10 |
| THANKYOU #12 | 0/10 | 0/10 |

For the MY held-pose trial, raw outputs were MY in five windows and ME in five;
the combined filter emitted MY in two windows and ME in four. This is a pose
control, not a complete-sign accuracy result.

The median per-window wrist-speed proxy was 0.53 and 0.86 shoulder-widths/s
in the two scored REST trials. It was 0.57 and 0.52 in the two YES trials;
the held MY pose was 0.17. The whole-trial windows include relaxation and do
not isolate the actual sign, but the observed overlap illustrates why a fixed
motion threshold cannot safely separate rest from signing in this session.
The second unrelated MOVE trial contained windows with much higher wrist and
finger motion, showing that motion alone also cannot establish that a known sign
is being performed. These measurements do not identify whether the source is
real movement or landmark noise.

## Limitations and next step

This is one signer with two attempts of each selected sign, no exact boundaries,
no second unseen signer, and 25 missing scored windows. The diagnostic performs
model scoring and motion calculations synchronously, which can perturb frame
delivery. Sampled scores and two-second overlapping windows cannot establish
false events per minute, precise end-to-end sign latency, or generalization.

Do not install a motion gate or the 0.60/stability candidate as a default from
these data. The next useful measurement is to capture and retain only aggregate
tracking failure reasons, counts and timing, then examine carefully bounded
sign/idle examples from additional signers. A separate sign-activity model or
rejection method would require appropriately labeled data and a genuinely
held-out evaluation. Model training is not authorized by this diagnostic.
