# Sliding-window tracking coverage diagnostic — 24 November 2026

## Purpose and method

Investigate unscored windows in the earlier activity experiment. That earlier
report did not save failure categories, so this new session cannot prove the
causes of its 25 unscored windows. This session records separate model-input
coverage counts for hands, shoulder anchors, and face, together with fixed
failure categories. Counts are not proof that a tracked landmark is accurate.

The user confirmed normal capture and all six prompts: hands resting visibly,
hands held below frame, HELLO then relax, unrelated hand movement, HELLO then
relax, and natural rest. Each trial lasted about seven seconds with overlapping
two-second windows scheduled about every half-second. A full window is unscored
when the existing model adapter rejects it before inference. The diagnostic
computes coverage and performs inference synchronously, so its timing need not
match the asynchronous prototype. Captures have no verified sign boundaries.

The original local report `.cache/sliding-health-guided-20261124-v1.json` has
SHA-256 `ad2c8c236d3c98a95718bfcde312bc6a33baf10c1b04d6a63182a9d40412e277`.
A separate local confirmation record attaches user feedback to this hash. Only
predictions, timing, failure categories and aggregate coverage counts were
saved, not video, landmark coordinates, or embeddings.

## Results

| Trial | Scored windows | Unscored | Recorded unscored reason |
|---|---:|---:|---|
| Rest, hands visible | 10 | 0 | — |
| Rest, hands below frame | 0 | 10 | No hands in all 10 windows |
| HELLO #1, then relax | 8 | 2 | No hands in both windows |
| Unrelated hand movement | 10 | 0 | — |
| HELLO #2, then relax | 8 | 2 | No hands in both windows |
| Natural rest | 10 | 0 | — |

There were 46 scored and 14 unscored overlapping windows. In all 14 unscored
windows the normalized model input contained no detected hands; usable face
and shoulder anchors were present. The 10 windows in the hands-below-frame
trial are expected model-input rejection rather than evidence of a crashed
worker. The four HELLO-trial unscored windows ended around 6.2–6.7 seconds,
after the user was instructed to sign in the first three seconds and relax;
precise sign boundaries were not recorded. Hand-detection counts alone do not
verify exact tracking positions or explain earlier unrecorded failures.

When resting hands remained visible in trial 1, the raw classifier guessed MY
in all 10 windows and its score reached 0.86. The provisional 0.60 plus
two-window stability condition still emitted MY in seven of those windows. In
the unrelated MOVE trial the same condition emitted HELLO once. Therefore hand
presence alone does not distinguish signing from rest or unrelated movement,
and this filter should not be deployed as a validated fix.

In both HELLO trials raw HELLO appeared in multiple scored windows; the
provisional combined filter did not emit HELLO. Those observations are not a
general accuracy estimate: this was one signer with two HELLO attempts, and the
windows overlap. The diagnostic recorded no user-provided onset or offset times.

## Interpretation and follow-up

No-hands rejection is expected when hands leave the image and should remain a
quality guard. It is not a complete sign-activity or unknown-sign detector.
The earlier session's 25 unscored windows remain unexplained individually:
this matched check makes no-hands rejection plausible for some, but cannot
retroactively establish their reason. Additional labeled resting poses, unknown
signs, true held signs and multiple signers are needed to evaluate a rejection
strategy. Maintain the existing default and do not train a new model without
explicit authorization.
