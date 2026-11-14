# Face features for the 15-sign ASL Citizen subset

Extracted on 2026-11-15 at the user's request. This is a separate face-feature
cache; it does not change the existing hand/pose cache, model or webcam defaults.
No sign-recognition model was trained.

## Scope and results

- All 15 labels from the existing pilot: 219 train, 60 validation and 182 test
  videos, for **461 clips**. Official signer-disjoint splits are unchanged.
- **36,079 decoded frames**, all preserved at their original dimensions and
  source-FPS-derived timestamps. A face was detected on 36,076 frames; three
  frames across three training clips have a false face-presence mask. Face
  detection presence is not proof that the landmarks are anatomically accurate.
- The two-worker CPU extraction session took **211.6 seconds**, excluding the
  initial raw-archive integrity scan. The earlier 10-clip pilot also completed
  successfully. The speed of this small subset need not generalize to the full
  corpus.
- Compressed feature arrays total **197,246,818 bytes** (~188.1 MiB). Dataset
  videos and extracted features are Git-ignored and remain local.

## Representation and alignment

One `.npz` per video is stored under
`data/landmarks/asl-citizen-face-fifteen-v1/<split>/`. Its arrays are:

| Array | Shape | Meaning |
|---|---|---|
| `face_xyz` | `[T,478,3]` float32 | MediaPipe face landmarks in normalized image coordinates |
| `face_blendshapes` | `[T,52]` float32 | MediaPipe blendshape scores in the order stored in `summary.json` |
| `face_present` | `[T]` bool | Whether the detector returned a face in that frame |
| `frame_index` | `[T]` int32 | Original decoded frame index |
| `timestamp_ms` | `[T]` int64 | Source-FPS-derived MediaPipe timestamp |

Missing faces are zero-filled and explicitly marked false. Never interpret zeros
without the mask as a real face. The MediaPipe face result is not a shared
metric-depth coordinate system with the existing hand and pose channels.

Each per-video JSON records the feature checksum, original video SHA256 computed
for this extraction, source size, and the corresponding original landmark
archive/member and their hashes. The old extraction recorded video byte size
but did not record a video SHA256; therefore the new video hash establishes its
identity now, not proof of historical byte identity by itself.

All **461** feature checksums, shapes, finite values, masks and zero-filled
missing frames were checked. Each face `frame_index` and `timestamp_ms` was
compared with its original hand/pose `.npz` and matched exactly. The original
raw archive hashes were checked before extraction. Aggregates and the three
missing-frame identifiers are in [verification](assets/face-fifteen-extraction-verification-2026-11-15.json).

The raw caches can now be joined by video identity and original frame index.
For training, normalize face and other features appropriately and apply the
**same** later temporal-sampling indices to all modalities. The old GRU accepts
254 input features and cannot consume the new channels without a compatible
model and training; extraction alone does not change recognition behavior.

## Reproduce or resume

```powershell
.\.venv\Scripts\python.exe -u scripts/extract_15_sign_faces.py --workers 2
```

The script validates the official subset, source/raw alignment and model hash.
Completed clips are resumed only after their saved data and input hashes match.
The pinned [Google Face Landmarker model](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
is in `models/face_landmarker.task` with SHA256
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
MediaPipe runs on CPU using two workers. The `.cache/face-fifteen-full-v1.log`
records the local run; routine MediaPipe feedback-manager warnings appeared,
not the GPU tensor-write errors seen in the earlier Colab hand/pose test.
