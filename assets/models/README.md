# Local model assets

Store downloaded MediaPipe Tasks model files here. Binaries are ignored by Git.
The current local experiments use the files below. They are not included in Git;
review the model provider's terms and model cards before using or redistributing
them. `latest` URLs can change, so checksums identify the files used here.

- [MediaPipe hand landmarker](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task):
  `hand_landmarker.task`, SHA-256 `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`.
- [MediaPipe pose landmarker lite](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task):
  `pose_landmarker_lite.task`, SHA-256 `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`.
- The experimental WLASL2000 video model and class list, including source and
  checksums, are documented in [docs/data.md](../../docs/data.md).
- The optional [ASL Citizen I3D baseline](https://github.com/microsoft/ASL-citizen-code/releases/tag/checkpoints_v1)
  uses `aslcitizen_i3d.pt`, SHA-256
  `3538319620331d5ba731e0f9ac79161deeb37a895b7680d215864fc00a14477d`, and
  `aslcitizen_classes.json`, SHA-256
  `cdc2b7f77b8d8797c19db24c1b7bf501618db5feb29def583c43b3da47c34910`.
  The JSON is generated from sorted unique native glosses in the official train
  CSV, rather than the WLASL class list. See the data guide for reproduction and
  the dataset's non-commercial research terms. Neither asset is redistributed.
- A stopped local [SignRep probe](../../reports/2026-11-05-signrep-probe.md) downloaded
  the official `v0.0.1` representation weights as `signrep_v0_0_1.pt`, SHA-256
  `f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b`.
  It passed a public-clip CUDA feature check but has no trained gloss classifier
  or webcam validation. It is not integrated into the package. Weights and the
  CC BY-NC-SA source remain local/ignored; see the report for exact provenance.
