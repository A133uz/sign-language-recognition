# Local activity landmark collection

## Scope

The user authorized preparation of local labeled MediaPipe landmark collection.
This is a separate, opt-in diagnostic workflow, not a
change to the existing webcam recognizer and not authorization to train models.
The collection records hand and MediaPipe pose coordinates, timestamps and
presence/quality masks. No raw camera frames, video, face mesh, face blendshape
data or embeddings are written. Pose points may include coarse face and body
positions. These numerical landmarks can still reveal movements and appearance;
they are private local data, not anonymous ground truth.

Each 7-second trial is stored under a uniquely named
`.cache/sign-activity-recordings/<session>/` directory as an NPZ and JSON
metadata. The directory is Git-ignored. The session has prompted examples of
rest, unrelated motion, known signs, unknown HOPE and an already-formed MY pose.
The second HELLO, YES and THANKYOU are requested at a slower pace. No exact
sign boundaries are collected. Sign trials may contain preparation and rest;
an entire trial must never be treated as fully labeled signing activity.

Each prompt is an expected category, not a verified label. Session and trial
confirmation fields start as `pending`; record the user's confirmation and any
affected trial numbers separately after the session. If a trial is interrupted,
its incomplete frames are discarded, completed trials remain local, and the
session manifest is marked interrupted. NPZ contents are numeric-only and each
completed trial is hash-verified. Do not treat recordings from a single signer
as evidence of signer-independent recognition. Group related windows by
recording and signer to avoid leakage in any later evaluation.

## Run only with live participation

```powershell
.\.venv\Scripts\python.exe scripts/collect_activity_landmarks.py
```

Press Space for each trial. Wait for SIGN NOW, follow the instruction and let
the capture stop automatically. For HOLD_MY, form the pose during GET READY.
The script closes after 14 trials; Q interrupts and keeps completed trials.
`--camera 1` and `--device cpu` are optional. The webcam processes the same
face, hand and pose detectors as the ASL prototype, but only hand/pose features
are retained. The session prints its unique local path on startup. There is no
automatic upload, classifier training or model update.

Deleting the individual session folder later removes its locally retained
landmark recordings and metadata; no cleanup has been performed automatically.

## First confirmed session — 25 November 2026

The user completed and confirmed all 14 prompts with normal capture. The local
session is `.cache/sign-activity-recordings/20261125-000055-ac80000c/`. The
manifest hash is `7bb034bc264163ddf74bdd19d673d61b86b508981e6846f158d7e4cedfd1f516`;
the separate confirmation refers to that hash. All 14 NPZ files and their JSON
metadata passed numeric-schema and SHA-256 checks. NPZ data totals 1,350,068
bytes. No video or face mesh was saved. Detection presence is not verified
tracking accuracy.

The recorded categories are three rest prompts, two unrelated-movement prompts,
six known-sign prompts (HELLO, YES and THANKYOU twice each), two HOPE prompts
outside the 15-label vocabulary, and one held MY-pose prompt. These are **one
signer** and whole-trial prompt labels. They are not frame-level activity
annotations, and slower attempts were not independently timed as slow signs.

`scripts/analyze_activity_collection.py <session-path>` checks the confirmation
hash and summarizes the existing wrist and hand-shape motion proxies across
overlapping two-second windows without training or writing derived coordinates.
All 14 trials yielded ten windows each; the hands-below-frame rest trial had
no valid wrist/hand motion measurements. The remaining trials had between five
and ten valid windows. The median wrist proxy across windows was approximately
1.06 and 0.56 shoulder-widths/s for the two visible-rest trials, 0.79 and 0.53
for the YES trials, and 0.26 for the held MY pose. These whole-trial windows
include rest and possible transitions; the values may reflect tracking noise
and body movement as well as hand movement. They support continued caution about
a simple motion gate, but are not a calibrated comparison or an accuracy test.

The data is insufficient for signer-independent evaluation, selecting a
threshold, or training a reliable activity detector. Obtain multiple signers,
independent sessions and accurate temporal labels before making such claims.
No training was performed.
