# Webcam recognition diagnostic progress — 2026-11-04

The webcam goal remains incomplete: earlier user trials recognized `hello`,
`thank you`, `please` and `sorry` in restricted mode, but failed on `yes`, `no`
and the pronoun `me`. Dataset results below are not webcam validation.

## Corrected landmark rebuild

Five source intervals had been sliced using native 29.97-FPS frame indices
instead of WLASL's 25-FPS annotation timestamps. They were regenerated before
this run. All 132 local clips now pass the landmark preparation check, versus
131 before correction. The added clip changes the test count; this is not a
controlled accuracy comparison with older models.

- Features: `data/processed/wlasl_expanded_pose_fps25_20261104.npz`.
- Schema: `hands-pose-image-xyz-relative-v1`, 32 frames, 160 values per frame.
- All 132 recorded video hashes were checked against the current files.
- Split: 80 train, 21 validation, 31 test; all 12 labels represented in each.
- Fresh linear SVM: scale floor 0.001, seed 42, train-only scaler fitting.
- Results: 80/80 train, 11/21 validation, 17/31 test. Validation `me` 0/2,
  `i` 0/1, `yes` 1/2, `no` 1/3. Not selected for webcam deployment.
- Generated details: `runs/fps25_landmark_rebuild_20261104.json` (ignored).

## I3D preprocessing comparison

The same 21 validation IDs were used across all variants. Their membership is
unchanged between the historical and corrected feature archives. Features were
not reused: the script decoded current video pixels and recorded their hashes.
Predictions were restricted to the twelve configured labels. Test outcomes were
not used to select the preprocessing. The external checkpoint may have seen these
videos or signers, so these results are exploratory and not independent accuracy.

| Spatial preparation | Frames | Pool temporal logits | Top-one matches |
| --- | --- | --- | --- |
| Current aspect-preserving short side 256 | Up to 64 sampled | Mean | 20/21 |
| Current aspect-preserving short side 256 | Up to 64 sampled | Max | 20/21 |
| Current aspect-preserving short side 256 | All | Mean | 20/21 |
| Current aspect-preserving short side 256 | All | Max | 20/21 |
| Reference test loader | Up to 64 sampled | Mean | 8/21 |
| Reference test loader | Up to 64 sampled | Max | 8/21 |
| Reference test loader | All | Mean | 9/21 |
| Reference test loader | All | Max | 7/21 |
| Square 256 then center crop | Up to 64 sampled | Mean | 17/21 |
| Square 256 then center crop | Up to 64 sampled | Max | 18/21 |
| Square 256 then center crop | All | Mean | 18/21 |
| Square 256 then center crop | All | Max | 17/21 |

The reference spatial loader enlarges only frames whose short side is below
226 pixels, then applies a centered 224 crop with NumPy rounding. The square
variant is motivated by the training loader's resizing of larger images;
it does not reproduce the complete training procedure. All variants retain BGR
channels and normalize to [-1, 1]. No inference ran out of GPU memory.

Sources: [reference test entry point](https://github.com/dxli94/WLASL/blob/master/code/I3D/test_i3d.py),
[test dataset loader](https://github.com/dxli94/WLASL/blob/master/code/I3D/datasets/nslt_dataset_all.py),
[training dataset loader](https://github.com/dxli94/WLASL/blob/master/code/I3D/datasets/nslt_dataset.py),
[crop implementation](https://github.com/dxli94/WLASL/blob/master/code/I3D/videotransforms.py).

The existing default remains unchanged. Closer reproduction of the reference
test path did not improve this sample and does not explain the user's live errors.
Generated details: `runs/i3d_reference_comparison_fps25_20261104.json` (ignored).

## Guided webcam command

`test-webcam-i3d` randomizes repeated prompts, gives a two-second countdown,
captures one sign and compares aspect/square resize with mean/max temporal
pooling on that same capture. Frames stay in memory; only predictions, prompts,
timing and provenance hashes are saved. Early exit retains completed trial counts.
The report calls its metric prompt agreement: intended labels have not been
independently checked against the performed signs. Selecting a variant requires
fresh trials, including the other signs in the intended vocabulary.

Simulated lifecycle tests exercised countdown, capture, prediction, completion
and camera cleanup. Real CUDA inference on saved clips `64287` (`yes`) and
`38527` (`no`) verified that the comparison's `aspect_mean` results equal the
existing default exactly. Both default top-one guesses matched those two saved
clips. This was not a webcam signing test. A human run of the guided command is
recorded in [the follow-up results](2026-11-04-guided-webcam-results.md).
Generated details: `runs/i3d_guided_comparison_cuda_smoke_20261104.json`.

## Provenance

Code is an uncommitted worktree based on
`0ecee354c5739d3e75c01c9985cf00eb70860b78`; that commit alone does not contain
these experiments. Windows, Python 3.12.6, RTX 4060 8 GB; I3D ran on CUDA,
landmark extraction and SVM training on CPU.

| Input | SHA-256 |
| --- | --- |
| `uv.lock` | `de7a8b970f37a02790469fa5817fab4d8b92f5efc34b3236fab8e7dff5bfb2e9` |
| Twelve-label config | `5c0765dfde21cac0d5cce028cf0b424395b1e1956a2e9448089bbfd61c59cf36` |
| Corrected landmark archive | `cd0d2dec59961cd62d7bd0f4a92ae4069e5f326c1de6f3b8a89395d75e9b8909` |
| Historical archive used only for comparison split IDs | `85b9bc86d495ca6762bbbc37b461aa43114c11898d53c6ef6990fba76143a4b0` |
| I3D checkpoint | `243a19e6deef3becffbfc5b7dd8adb32916c8bee482565ca243c082584732620` |
| I3D class mapping | `2c131fb7f052340aa7ae8579c8cae9746028740b8e62efeb957a7790dbefcf4a` |

Raw clips, features, checkpoints and generated reports remain ignored by Git.
