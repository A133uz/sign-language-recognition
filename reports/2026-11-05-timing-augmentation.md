# Timing augmentation comparison — 2026-11-05

The world-hand model's first human test failed ME on 0/3 attempts. Consented
preview review found no established signing mismatch explaining that failure.
This experiment tests sensitivity to the position and duration of a gesture
within a capture. It does not use the user's images, coordinates or embeddings
for training and does not change the class vocabulary.

## Selection within training signers

The existing archive contains 173 train / 46 validation / 146 test clips. Four
predefined candidates were compared using three-fold training-only
`StratifiedGroupKFold`, shuffled with seed 42. Each source signer ID stayed in
one fold, and every fold's fitting partition contained all twelve labels.
Scalers were fitted on each fold's fitting examples only. Candidates were ranked
by aggregate out-of-fold macro accuracy, then correct count. This is exploratory
cross-validation, with source labels and signer IDs not independently reviewed.
It is not independent of previous work on this training corpus.

| Representation | Timing augmentation | Training-signer CV | Macro accuracy | ME |
| --- | --- | --- | --- | --- |
| Existing world summary | No | 117/173 | 0.671 | 5/11 |
| Existing world summary | Yes | 133/173 | 0.766 | 9/11 |
| Missing-hand-masked summary | No | 103/173 | 0.600 | 6/11 |
| Missing-hand-masked summary | Yes | 112/173 | 0.648 | 9/11 |

Timing augmentation produces five versions of each fitting sequence: original;
six frames delayed with the first frame repeated; six frames advanced with the
last frame repeated; 32 samples over original frames 0–24; and 32 samples over
frames 7–31. Temporal direction is preserved, without inventing coordinates.
It changes timing only approximately and may omit part of a gesture; it does
not model every real webcam capture or establish robustness to signing variants.

After selection, the baseline and chosen augmented candidate were fitted on
all 173 training clips and checked on the existing validation partition. The
candidate scored 39/46 (macro 0.826), versus baseline 40/46 (0.854). ME stayed
5/5; THANKYOU stayed 1/3; HELLO fell from 4/4 to 3/4 and NO from 3/3 to 2/3;
GOOD improved from 2/4 to 3/4. These regressions prevent declaring a general
improvement. No test scores were computed for the new candidate.

Report: `runs/timing_probe_20261105.json` (ignored). Grouped splitting follows
the [scikit-learn API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

## Reproducible candidate and same-capture comparison

```powershell
.\.venv\Scripts\python.exe -m sign_language train data/processed/aslcitizen_world_20261104.npz --method svm --representation world_summary --temporal-augmentation --skip-test --device cpu --output checkpoints/aslcitizen_timing_linear_20261105.pt
.\.venv\Scripts\python.exe -m sign_language test-webcam --backend landmarks --weights checkpoints/aslcitizen_world_linear_20261104.pt --compare-weights checkpoints/aslcitizen_timing_linear_20261105.pt --device cpu --sign HELLO --sign THANKYOU --sign YES --sign NO --sign ME --repeats 3 --seed 43 --output runs/webcam_timing_comparison_20261105_01.json
```

Output paths must be new. The candidate fits the same linear SVM and 670-value
representation, with 865 training sequences after augmentation. Existing weights
remain frozen. New weights SHA-256:
`fea3d267f4b7cef5c380cda5978225cf5da32aabdd00df13e7cd884fdc0a020a`.

The optional comparison requires identical label order, native dataset, feature
schema and extractor hashes. It extracts each capture once and scores that exact
sequence with both checkpoints. Reports retain predictions and aggregate tracking
counts, without camera pixels or coordinates. Variant `landmarks` is the original
model; `comparison_landmarks` is the timing-augmented candidate. Both choose
among all twelve native checkpoint labels, even though the diagnostic prompts
five signs. Neither model reliably rejects unknown signs.

The SVM `--skip-test` path now bypasses test inference as well as omitting test
metrics; previously it computed unused test predictions internally. The earlier
candidate reports did not expose or use those predictions for selection.

## Runtime checks

Real CPU MediaPipe extraction and both numeric classifiers matched archived-input
predictions on three public validation clips (HELLO, THANKYOU and ME). The
THANKYOU clip was classified as GOOD by both, a continuing failure. A saved YES
clip replay exercised the full guided lifecycle, produced both variant results,
completed the prompt, and released its simulated camera. These are saved-clip
checks, not human webcam evidence.

Reports: `runs/timing_runtime_parity_20261105.json` and
`runs/guided_timing_replay_20261105.json` (ignored).

All 101 tests and the full `scripts/check.ps1` checks passed. New tests check
temporal direction, training-only scaler fitting, skipped test inference,
shared capture extraction, incompatible label rejection and comparison provenance.
Dependencies and environment arrangement did not change.

## Same-capture human comparison

All fifteen trials completed with predictions from both models. The user
confirmed following every prompt with normal capture. Captures contained
61–62 frames at 640x480, with hands in 20–32 of the 32 sampled frames. No trial
was rejected for tracking loss. Shared extraction plus both classifiers took
median 1957 ms (range 1906–5280 ms) after the two-second capture; this is batch
latency, not continuous recognition FPS.

| Prompt | Original top-one | Timing candidate top-one | Original top-three | Candidate top-three |
| --- | --- | --- | --- | --- |
| HELLO | 2/3 | 3/3 | 3/3 | 3/3 |
| THANKYOU | 0/3 | 0/3 | 1/3 | 0/3 |
| YES | 3/3 | 3/3 | 3/3 | 3/3 |
| NO | 2/3 | 2/3 | 3/3 | 3/3 |
| ME | 0/3 | 1/3 | 1/3 | 1/3 |
| Total | 7/15 | 9/15 | 11/15 | 10/15 |

This provides one same-capture example of improvement on ME and HELLO, with
THANKYOU still failing. It does not establish reliable recognition or justify
replacing the default. The candidate remains optional, and both checkpoints
are preserved. Earlier sessions used different captures, so their totals should
not be compared as controlled changes. Unknown gestures and the remaining
native labels were not tested in this session. The webcam goal remains active.

Report: `runs/webcam_timing_comparison_20261105_01.json` (ignored), SHA-256
`ddc1e8d0999ae17c2ec2d3bc2b452c7c707eed4da61935974198db1adccb4305`.
Only labels, timing, frame dimensions/counts and aggregate tracking counts were
saved; no user video, coordinates or embeddings were retained or used to train.
The process exited normally and released the camera. SVM vote ties are broken
by checkpoint label order, so top-one prompt agreement is not calibrated
confidence or an independent sign-correctness assessment.
The original had five top-score ties among its displayed guesses; the candidate
had two. These counts further limit interpreting small total differences.
