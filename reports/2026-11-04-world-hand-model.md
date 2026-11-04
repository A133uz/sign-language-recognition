# Native world-hand model — 2026-11-04

Prior WLASL and ASL Citizen video-model trials failed on the user's yes/no/me
signing, and body alignment did not fix those attempts. This experiment trains
a separate hand-shape classifier on the acquired native ASL Citizen subset.
It does not merge label systems or replace the full WLASL vocabulary goal.

## Preparation

369 source clips were decoded, with original splits retained. Preparation accepted
173 training / 46 validation / 146 test clips from 27/5/11 signer IDs, with all
twelve native labels present in each partition. Three training clips had hands
in fewer than four distinct samples; one training YES clip exceeded the 600-frame
bound. These four were excluded explicitly. Signer-ID overlap and identical
video bytes across splits are rejected; source signer identity and labels are
not independently reviewed.

The 288-value schema retains image-hand and upper-body context and adds estimated
world-hand coordinates, wrist centered and normalized by wrist-to-middle-MCP
distance. Image-x ordering determines the paired hand slots. Missing/degenerate
world hands remain zero with validity masks. Encoding uses unmirrored RGB input
through the existing CPU MediaPipe models. Dataset features and weights stay local.

The compact representation retains 128 world-hand values, four wrist-to-nose
offsets and two image-hand presence bits. Four temporal-bin means plus global
standard deviation yield 670 inputs. This reduces dependence on absolute image
position and represents hand shape at a normalized scale. It does not guarantee
robustness to rotation, occlusion, handedness or domain changes.

```mermaid
flowchart LR
    A[Full video frames] --> B[32 RGB samples]
    B --> C[MediaPipe hands and pose]
    C --> D[World hand shape and body-relative wrists]
    D --> E[Temporal summary]
    E --> F[Linear SVM votes]
    F --> G[Native class guesses]
```

## Validation-only selection

All candidates used seed 42 and the same source partitions. Standardization was
fitted on training only, with scale floor 0.001. SVMs used C=1 and balanced class
weights; the RBF comparison used scikit-learn's scale gamma. The GRU trained 150
epochs on CUDA and selected an epoch by validation loss. Test scoring was disabled
for every candidate. Selection ranked validation macro accuracy, then total
correct predictions.

| Candidate | Training | Validation | Validation macro accuracy |
| --- | --- | --- | --- |
| Raw 288-value sequence, linear SVM | 173/173 | 32/46 | 0.693 |
| Compact world summary, linear SVM | 173/173 | 40/46 | 0.854 |
| Compact world summary, RBF SVM | 150/173 | 34/46 | 0.708 |
| Raw world sequence, GRU | 148/173 | 27/46 | 0.568 |

The compact linear SVM was frozen before held-out scoring. Source report:
`runs/aslcitizen_world_candidates_20261104.json` (ignored).

## Selected model: first held-out evaluation

| Native class | Validation | Test |
| --- | --- | --- |
| HELLO | 4/4 | 11/14 |
| THANKYOU | 1/3 | 9/12 |
| BYE | 3/3 | 11/12 |
| PLEASE | 4/4 | 12/13 |
| SORRY | 3/3 | 6/12 |
| YES | 5/5 | 9/10 |
| NO | 3/3 | 10/11 |
| HELP | 5/5 | 8/13 |
| FINE1 | 3/4 | 8/16 |
| FINE2 | 2/3 | 2/8 |
| GOOD | 2/4 | 7/12 |
| ME | 5/5 | 11/13 |
| Total | 40/46 | 104/146 |

Test results were obtained once for the selected checkpoint, with no retraining
or candidate selection using those outcomes. These counts describe the source
test partition, not independently reviewed webcam accuracy. FINE2 and SORRY remain
weak, and untrained signs cannot be reliably rejected. FINE1/FINE2 are distinct
native classes; no fingerspelled I class or WLASL label alias is introduced.
Report: `runs/aslcitizen_selected_world_test_20261104.json` (ignored).

## Runtime checks and provenance

The capture-first backend retains frames only in memory, then extracts up to 32
distinct samples using the same preprocessing as dataset preparation. On three
saved source YES/NO/ME clips, feature arrays matched the archived features exactly
(maximum difference 0.0), and top-one predictions agreed and matched those clip
labels. This is implementation parity, not a new accuracy estimate.
`runs/world_capture_feature_parity_20261104.json` records the checks.

A saved-clip replay exercised the entire guided pipeline with real CPU MediaPipe
and the selected classifier, released its simulated camera, and saved predictions
and tracking counts only. A first looping fixture captured mostly rest frames and
correctly rejected insufficient hand tracking; presenting the complete source
sign during the capture interval passed. No model or threshold was changed to
make the replay pass. Report: `runs/guided_world_model_replay_20261104.json`.

All 98 local tests, lint/format checks, vocabulary checks, base diagnostics and
Git whitespace checks passed. SVM inference is CPU; CUDA was
actually exercised by the GRU training comparison. Existing `.venv` and lockfile
were used without dependency changes.

- Features: `data/processed/aslcitizen_world_20261104.npz`, SHA-256
  `544b8f978288e1c0018672e4829410dfb651f1c10cb8a34575dde2d98404b7bb`.
- Selected checkpoint: `checkpoints/aslcitizen_world_linear_20261104.pt`, SHA-256
  `dbcdc408ff5ebe38ebb709eb3bd6a5a87bc3fac631a646598476e4e164501719`.
- Schema: `hands-pose-world-wrist-normalized-v1`.
- Code is an uncommitted worktree; data, trained weights and feature archives
  remain ignored by Git under the existing dataset terms.

## First human guided session

All fifteen prompts completed, with a prediction on every attempt. All captures
contained 61 frames. Hand tracking was present in 25–32 of the 32 sampled frames;
no trial was rejected for insufficient hands. Predictions were selected from
all twelve native checkpoint labels, not just the five prompted labels.

| Prompt | Top-one agreement | Top-three agreement |
| --- | --- | --- |
| YES | 3/3 | 3/3 |
| NO | 3/3 | 3/3 |
| HELLO | 2/3 | 2/3 |
| THANKYOU | 1/3 | 2/3 |
| ME | 0/3 | 1/3 |
| Total | 9/15 | 11/15 |

YES and NO are now correct on these attempts, unlike the previous I3D guided
session's 0/3 each. These were different captures and native candidate lists,
so the comparison is not a controlled model accuracy estimate. HELLO and THANKYOU
are less consistent than the earlier restricted WLASL video-model attempts,
and ME remains a live failure despite 11/13 on source test clips. The model is
not declared a complete webcam solution, and no failing class is dropped.
The checkpoint is preserved rather than retrained using these trial outcomes.

End-to-end feature extraction and classification took median 2163.0 ms, range
1984.8–5222.2 ms, after each two-second capture. This is batch capture latency,
not a continuous per-frame recognition rate. The SVM scoring itself is only
part of this time; CPU MediaPipe extraction is included.

Report (ignored): `runs/webcam_world_model_20261104_01.json`, SHA-256
`221811b9c4c11503f2f8ba5a2d30239b89f12cb4c070b822e23f29da61106a24`.
Only predictions, timing, frame counts and aggregate tracking counts were saved;
no webcam pixels, landmark coordinates or user training examples were retained.

## Follow-up video/hand fusion probe

Frozen WLASL I3D mean-pooled penultimate descriptors (1,024 values) were extracted
on CUDA for the same 173 training and 46 validation clips only. Training-only
standardization preceded SVM fitting. No test descriptors or scores were used
in this probe, and no user webcam examples were used for training.

| Candidate | Validation | Validation macro accuracy |
| --- | --- | --- |
| Video descriptor, linear SVM | 28/46 | 0.601 |
| Video + 670-value hand summary, linear SVM | 35/46 | 0.737 |
| Video + hand summary, RBF SVM | 36/46 | 0.768 |

All were below the frozen hand model's 40/46 (macro 0.854). They were not
promoted to webcam use or evaluated on test clips. Repeated use of this small
validation partition limits confidence in further candidate comparisons.
Report: `runs/fusion_validation_probe_20261104.json` (ignored).

## Consented visual review — 2026-11-05

The user authorized a few live-preview frames being visible in chat, without
adding them to training data. Desktop screenshot capture timed out. The
preview-only module instead delivered three ME frames and detected-hand crops
through a temporary loopback receiver, with images held in memory only.
The hand was visible, but the front view alone did not resolve fingertip depth.
The user confirmed that the index fingertip pointed into the chest. This does
not establish a signing error or resolve the model's ME failure.

The THANKYOU board subsequently arrived. Fingertips began near the chin and the
hand moved outward, consistent with the basic motion in
[ASL University's THANK YOU description](https://www.lifeprint.com/asl101/pages-signs/t/thankyou.htm).
No obvious signing mismatch was established as the cause of either error.
Three sampled frames cannot establish expert sign correctness or the landmark
tracker's accuracy. This preview did not run a classifier and is not a scored
webcam test. The receiver was closed and image buffers cleared after review;
the consented preview images remain in chat, with none added to project files
or training data. The preview process exited normally and released the camera.

After adding the descriptor extraction and preview module, `scripts/check.ps1`
passed: 98 tests, lint/format, lockfile, vocabulary, base diagnostics and Git
whitespace checks. These checks do not establish recognition quality.
