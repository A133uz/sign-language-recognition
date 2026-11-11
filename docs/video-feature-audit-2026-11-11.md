# ASL Citizen video and feature audit — 11 November 2026

**Stopped at the user's request on 11 November 2026.** The last status and all
214 completed reports agree: **214 / 836 shards, 21,400 training clips** checked.
The corpus audit is incomplete; no final full-corpus integrity verdict was issued.
No audit, cleaning or old verification process was running at the stop check.
No videos or extracted features were removed, filtered, relabeled or rewritten.
The completed focused findings below are retained. Do not automatically resume
the scan or run the final audit publisher. The next authorized experiment uses
five classes and earlier epoch selection, preserving the current preprocessing.

## Scope and investigation record

Requested after the corrected ST-GCN webcam session confused HELLO, THANKYOU,
NO, ME and YES. This audit investigates data integrity and representation quality;
model errors alone do not establish incorrect ASL annotations.

1. Read the extraction, preprocessing, cache preparation and existing diagnostic
   implementations. The extraction summary reports 83,399 successful clips and
   6,901,733 decoded frames, but this is not independent content verification.
2. Located the retained original ASL Citizen ZIP and its member sizes/CRC32 values.
   Original extraction did not hash each video, so new video SHA256 values establish
   an audit baseline rather than proving historical identity by themselves.
3. Implemented `scripts/audit_video_features.py` for a read-only, two-worker scan:
   official metadata and signer separation; every raw shard checksum and NPZ array;
   every prepared row reconstructed through the unchanged normalizer; every video
   SHA256/CRC32 compared with the original archive directory; and a reproducible
   subset of complete video decodes. Only the new audit output directory is written.
4. Preserve and compare the selected GRU/ST-GCN checkpoint hashes. Do not retrain,
   modify, discard, relabel or silently repair any source video or feature.

Command:

```powershell
.\.venv\Scripts\python.exe scripts/audit_video_features.py --workers 2 --output data/audits/asl-citizen-video-feature-audit-2026-11-11-v1
```

Full per-clip evidence and resumable shard reports are local under that ignored
directory. Missing detections, small hands, out-of-image coordinates and feature
clipping are review signals, not proof of corrupt videos or incorrect glosses.
Semantic review must use training/validation examples; the official test split
must remain unchanged.

Full-scan completion was superseded by the user's stop request.

## Completed focused checks before the stop

The five requested signs have **94 training/validation clips**. No test examples
were used to choose the visual review cases. Selected the lowest-coverage and
median-coverage clip per sign, and inspected five exact source frames per clip
with the saved raw landmarks overlaid: 10 clips / 50 frames. These are purposive
examples, not a random estimate of tracking accuracy.

| Gloss | Train + validation clips | Median frames with any detected hand | Median duration |
|---|---:|---:|---:|
| HELLO | 18 | 35.7% | 2.20 s |
| THANKYOU | 19 | 35.5% | 2.87 s |
| NO | 19 | 41.3% | 2.40 s |
| ME | 18 | 31.5% | 2.35 s |
| YES | 20 | 43.1% | 2.17 s |

### Confirmed hand-identity instability

`9009854854743657-YES.mp4` (training, 66 frames): source frames 16–17,
23–25 and 30–34 show the same visible signing hand. Its canonical slot changes
at frames **17, 24, 25, 31 and 34**. The raw handedness label changes at these
transitions with scores as high as 0.999, and the prepared features follow it.
Both pose wrist masks are invalid at frames 16–17; later one wrist is valid,
but this does not consistently prevent reassignment.

The current `canonical_hands` assignment uses a framewise handedness cost,
pose-wrist distances and a weaker recent-position cost. This example demonstrates
that its continuity rule does not keep one physical hand in one slot. Slot 0 and
slot 1 feed different feature groups / graph branches, creating artificial
temporal changes even though the signer did not switch hands.

An initial screen of the 94 prepared sequences flags 25 clips with a nearby
single-hand slot change. This is a **candidate rate**, not 25 confirmed errors:
alternate detections of two real hands can also satisfy the heuristic. A second,
corpus-wide screen requires valid body anchors on both frames and excludes padding;
its results will be reported separately. The proximity threshold is 0.5 normalized
shoulder-width units and is an audit heuristic, not a classification threshold.

Evidence: local `visual-review/yes-slot-switch-review.png`,
`yes-slot-switch-facts.json`, `canonical-slot-review.json`.

### Confirmed missing detection during visible motion

`0035172227152260316-ME.mp4` (training, 56 frames): only six frames have a detected
hand (10.7%). Source frames 14 and 28 show the hand near the chest while the saved
hand mask is false. Consequently, hand landmarks are zero-filled there. The pose
still has shoulder landmarks. This is a tracking failure in these frames; it is
not evidence that the source gloss is wrong.

### Rest periods and repeated gestures

The HELLO and THANKYOU boards show substantial pre/post-sign rest. In several
frames the hands are genuinely below the image or at rest; low whole-clip hand
coverage must not be interpreted as an equal fraction of missed signing frames.

`9953610281521326-YES.mp4` is **664 frames / 22.13 seconds**, much longer than
the focus-sign median. A 20-frame contact sheet spanning the clip shows repeated
gestures. Uniformly compressing this entire video to 128 samples combines multiple
repetitions at a different temporal resolution from a short webcam attempt.
This is timing/segmentation variability, not established mislabeling.

Evidence: local `visual-review/{hello,thankyou,no,me,yes}-review.png`,
`yes-longest-contact.png`, `focused-train-val.json`, `focused-summary.json`.

## Additional audit actions and limitations

- The first inventory attempt was stopped before it wrote a manifest or shard
  report: per-file `Path.is_file()` calls were unnecessarily slow. Replaced that
  inventory with `os.scandir`, then restarted the same read-only scan. No extraction
  worker or training process was stopped.
- Added six fixture checks for malformed/NaN arrays, timestamps, masks, missing
  fills and slot-screen padding/anchor exclusions. All six pass.
- Re-extracted the two specific ME/YES diagnostic videos with the current CPU
  hand/pose models, in memory only. Every saved raw array, every extraction
  metadata field and the normalized float16 features reproduced **exactly**.
  The ME clip again had only six detected-hand frames. This rules out a damaged
  cache as the explanation for these two examples; detector / assignment behavior
  itself reproduces the problematic inputs. Saved only comparison measurements in
  `fresh-extraction-check.json`.
- Verified that the prepared-cache label dictionary equals the raw label map and
  that its raw-run ID matches the extraction manifest. The full scan independently
  verifies official row IDs, labels and targets for each sequence.
- Confirmed official CSV class support: training **9–24 clips per class**,
  validation **2–7**, test **7–20**. These are examples per class, not unique signers
  per class. All splits contain 2,731 glosses. Signer-disjoint counts are 35/6/11.
- Reviewed the [official ASL Citizen datasheet](https://www.microsoft.com/en-us/research/project/asl-citizen/datasheet/).
  It acknowledges residual technical issues and possible mistaken submissions,
  as well as legitimate regional variants and multiple meanings of English glosses.
  None of those general statements establishes a wrong label in the examples above.
- New checks do not measure semantic annotation accuracy, and sampled video
  decoding cannot establish that every source file is decodable. Full source-byte
  CRC comparison and full landmark-array validation have separate scopes.

## Proposed next experiment (not performed)

Prioritize stable temporal hand identity and evaluate the change on training /
validation examples and a controlled webcam test. A corrected normalizer can build
a **separate, versioned prepared cache from existing raw landmarks**; this part does
not require repeating MediaPipe extraction. Models would need compatible retraining
and evaluation because their input representation changes. Missing detections need
a separate targeted detector review; re-normalization cannot invent absent joints.

Do not silently apply a changed normalizer to the existing checkpoints or rewrite
the existing cache. Keep the current test set and published baseline metrics intact.
