# SignRep feasibility probe — stopped 2026-11-05

After the landmark experiments failed to establish reliable webcam recognition,
a newer RGB representation was investigated. This is an incomplete local research
probe, not an implemented project recognizer or a validated webcam improvement.
The user cleared the development goal while feature extraction was running;
that process was stopped and its partial cache preserved.

## Source and asset provenance

- Official [SignRep repository](https://github.com/ryanwongsa/SignRep), release
  `v0.0.1`, source commit `64e71946fe1efa6857703d15ebbb69c04023e407`.
- Downloaded source archive SHA-256:
  `25ec21b0a177a3eb86987efb5c0b19b16ae6c455915163998cabe536549a1fae`.
- Official [checkpoint release](https://github.com/ryanwongsa/SignRep/releases/tag/v0.0.1),
  asset `ckpt.pt`, 222,676,746 bytes. Download size and SHA-256 matched the GitHub
  release metadata: `f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b`.
- Local ignored weight path: `assets/models/signrep_v0_0_1.pt`.
- The inspected source license is CC BY-NC-SA 4.0. Source, adapters and weights
  remain in ignored local research storage; no SignRep implementation or weights
  are redistributed by this commit. See the official source for its terms and
  [paper](https://openaccess.thecvf.com/content/ICCV2025/html/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.html)
  for the authors' method and evaluation.

The official release is a representation model. Its public example converts RGB
frames to square 224x224, applies ImageNet normalization and extracts 768-value
features from consecutive 16-frame windows with stride two. A gloss classifier
would still need to be fitted and evaluated. This probe did not reproduce the
paper's recognition scores or imply pretrained webcam accuracy.

## Checks actually performed

The checkpoint loaded safely with `weights_only=True`; all 327 tensors matched
the official example model with strict loading. A local public ME clip produced
28 finite 768-value descriptors on the RTX 4060. The first two-window-batch probe
peaked at 738,797,568 allocated GPU bytes. Subsequent eight-window batching peaked
at 2,252,412,928 bytes during extraction. These are real CUDA inference checks,
not import checks or classification evidence.

The ignored prototype used evaluation-only equivalents for the two required timm
layers and an OpenCV/NumPy equivalent of the example resize/normalization. It
made no dependency or environment changes. These adapters were not integrated
into the package, and parity with an installed upstream dependency stack has
not been independently established.

The attempted extraction was limited to 219 native public training/validation
clips, preserving original memberships and checking video hashes. Source test
clips and personal webcam samples were excluded. Each cache key bound video,
weights, source commit, adapter and preprocessing hashes. The stopped run left
43 cache files under ignored `data/processed/signrep_segment_cache`; no final
`aslcitizen_signrep_20261105.npz` archive was produced. No label classifier,
cross-validation result, source-test score or webcam prediction exists for this
probe. Existing recognition models and defaults were not changed.
