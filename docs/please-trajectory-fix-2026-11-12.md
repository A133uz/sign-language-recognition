# Targeted PLEASE/MY recognition correction

The user requested all necessary checks to recognize the four known validation
PLEASE examples. This is explicitly **development-driven** work: these four
clips and their labels have already informed feature design and selection.
Success on them cannot be described as independent generalization evidence.
No official test data, live media or new personal training data is used.

## Initial experiment and acceptance criteria (before fitting)

- Preserve the 219/60 official train/validation subset and all 15 labels, prior
  checkpoints, five-class models, raw landmarks and videos. No cleaning,
  class removal, class-specific routing, filename-based inference or added
  validation examples in training.
- Append six label-independent wrist descriptors to the existing 17: loop
  strength, turn coherence, absolute turning, axis ratio, endpoint/path ratio
  and path length. Choose the strongest geometric window among 12/24/36 cached
  steps, entirely within a contiguous tracked/anchored dominant-hand span.
  These are estimated projected motion, not semantic rules or perfect circles.
- Keep two 256-unit GRU layers, mean pooling, dropout, augmentation, batch size,
  optimizer, seed and the existing 28-epoch cap/stopping policy. Geometry is
  recalculated from actual temporally/spatially augmented frames. Train-only
  imputation/standardization extends to all 23 summary values.
- New architecture: 256 pooled + 23 summaries = **279 inputs** to a 15-output
  linear head, 90 extra trainable weights (792,168 total). New model source and
  architecture tag preserve compatibility with previous checkpoints.
- Train only on the official training subset. Retain every epoch and its
  detailed 60-example validation predictions. Keep normal `best.pt` selected
  by the existing competitive-validation/small-gap policy.
- Additionally, a development-qualified `target-best.pt` may be selected among
  epochs that achieve **PLEASE 4/4, MY 4/4, and overall at least 50/60**. Apply
  the same competitive-validation/small-gap selection within that qualified
  set. State this target-specific selection clearly; no success if the set is
  empty. Do not replace the normal selection after the fact.
- Verify actual application loading and full per-class regressions before
  declaring the requested development target achieved. Preserve failed
  attempts. No default replacement or webcam launch is implied by this step.

Output: `models/asl-gru-fifteen-signs-trajectory-v1/`.

```powershell
.\.venv\Scripts\python.exe -u scripts/run_please_trajectory_fix.py
```

## Completed experiments

The development target is met. The selected model predicts **PLEASE first on
all four known validation clips**, including fresh extraction from the original
videos through the application. MY remains **4/4**. This is targeted development
validation; the official test split remains untouched and webcam accuracy has
not been established.

| Attempt | Validation | PLEASE | MY | Accepted? |
| --- | ---: | ---: | ---: | --- |
| Original temporal GRU, epoch 13 | 50/60 | 0/4 | 4/4 | No |
| Joint GRU + 23 features, selected epoch 6 | 51/60 | 3/4 | 3/4 | No |
| Frozen epoch-13 GRU, refitted readout, C=1 | 53/60 | 4/4 | 3/4 | No |
| Frozen epoch-13 GRU, refitted readout, **C=0.1** | **54/60** | **4/4** | **4/4** | **Yes** |

Joint training stopped after nine epochs. Some epochs reached PLEASE 4/4, but
none met all three criteria, so no `target-best.pt` was created. The scalar-only
check also failed: 17 descriptors scored 43/60 (PLEASE 4/4, MY 2/4); 23 scored
43/60 (PLEASE 4/4, MY 3/4). Every attempt and prior checkpoint is preserved.

After observing the C=1 failure, this bounded follow-up grid was declared. Each
head used the same **219 original official training examples**, frozen recurrent
weights and train-fitted normalization. Smaller C means stronger L2 regularization.

| C | Clean training | Validation | Validation CE | PLEASE | MY |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 211/219 | 53/60 | 0.7443 | 4/4 | 3/4 |
| **0.1** | **216/219** | **54/60** | **0.3861** | **4/4** | **4/4** |
| 1 | 219/219 | 53/60 | 0.4065 | 4/4 | 3/4 |
| 10 | 219/219 | 53/60 | 0.6201 | 4/4 | 3/4 |
| 100 | 219/219 | 54/60 | 0.9671 | 4/4 | 4/4 |

Selection used lowest validation CE among qualifying candidates. **This is
validation tuning**, informed by these known clips; it is not independent
evaluation. Validation rows never entered fitting, statistics or augmentation.
No class-specific routing or label removal was used.

## Selected model and workflow

Checkpoint: `models/asl-gru-fifteen-signs-trajectory-readout-c01/best.pt`.

SHA-256: `34b766c393cf20a77b001004376195e60ffb58943d7b1f9661bd23edbab81c91`.

```
Frames -> frozen MediaPipe Hands + Pose -> unchanged 254-channel normalization
  -> frozen two-layer 256-unit GRU -> valid-time mean pool [256]
  -> fixed 17 hand/body + six wrist-path descriptors -> train-fitted z-score [23]
Concatenate [279] -> refitted float32 Linear(279,15) -> logits -> displayed gloss
```

The branches consume the same sequence. GRU weights are bitwise identical to
the original temporal model's epoch-13 backbone. **No additional GRU epochs**
were run for this candidate. Only the 4,200 classifier coefficients and biases
were fitted using multinomial logistic regression (cross-entropy with L2,
C=0.1, lbfgs, max_iter=2000, seed 42). The Torch linear layer stores the result;
sklearn is not required at inference. Total parameters: 792,168. Epoch 13 in
the checkpoint identifies the inherited backbone, not a new training run.
Geometry statistics are fitted on training clips only; MediaPipe stays frozen.

All 15 labels compete without a filename, expected label or prompt as input.
Softmax scores are uncalibrated. Unknown/no-sign rejection and continuous-sign
segmentation remain unsolved. The default remains unchanged. Explicit command
for a later live test (no webcam was opened during these checks):

```powershell
.\.venv\Scripts\python.exe scripts/recognize_webcam.py --checkpoint models/asl-gru-fifteen-signs-trajectory-readout-c01/best.pt
```

## Checks performed

1. The [preceding video/feature audit](please-my-trajectory-2026-11-12.md)
   verified full decoding, raw archive hashes/member CRCs and exact cache
   reproduction for 35 PLEASE/MY train/validation clips. The circular motion
   already exists in all four failed PLEASE examples; none exceeds 128 frames.
2. New vectorized descriptors match the audited reference formulas. Absent
   tracked spans are not bridged; padding is excluded.
3. The actual application restored the selected class, 15 labels, strict source
   fingerprints and frozen statistics. All 60 cached validation score vectors
   exactly match the saved evaluation (maximum difference 0). Torch/sklearn
   export decisions also match.
4. All eight validation PLEASE/MY videos were decoded fully and freshly tracked
   in memory at original FPS through the webcam application's `predict_frames`.
   Expected labels entered only the subsequent report. Both scored 4/4; no
   video, stills or landmark arrays were saved.
5. Matched fixed-C readout ablation: keeping only the original 17 descriptors
   gave 51/60, PLEASE 4/4, MY 1/4. Adding six path descriptors gave 54/60,
   PLEASE 4/4, MY 4/4. Refitting alone improves PLEASE here, while path features
   help retain MY. This does not establish one cause of every historical error.
6. Prior `best.pt`/probe weights and prepared-subset hashes remain unchanged.
   No cleaning, official-test reading, default replacement, push or archive ran.
7. Repository checks passed: lockfile, compilation, **145 unit tests**, and
   dependency imports. Software checks do not establish live recognition quality.

Fresh application results (top-1 softmax score, not calibrated confidence):

| Participant | Video | First guess | Score |
| --- | --- | --- | ---: |
| P26 | `05859480723249688-PLEASE.mp4` | PLEASE | 77.28% |
| P39 | `6330094542104612-PLEASE.mp4` | PLEASE | 92.05% |
| P12 | `4096147054343935-PLEASE.mp4` | PLEASE | 93.14% |
| P21 | `42961246858695024-PLEASE.mp4` | PLEASE | 79.05% |

## Broader validation and limitations

Selected model: **90.00% top-1 (54/60), 98.33% top-5 (59/60), macro F1
0.885185, CE 0.386118**. Clean training: 216/219 (98.63%).

| Class | Original temporal model | Selected readout |
| --- | ---: | ---: |
| HELLO | 4/4 | 4/4 |
| NO | 3/3 | 3/3 |
| PLEASE | 0/4 | 4/4 |
| THANKYOU | 1/3 | 1/3 |
| YES | 4/5 | 4/5 |
| ME | 3/5 | 4/5 |
| YOU | 4/4 | 4/4 |
| WE | 3/3 | 3/3 |
| MY | 4/4 | 4/4 |
| YOUR | 4/4 | 4/4 |
| OUR | 4/4 | 4/4 |
| HELP | 5/5 | 5/5 |
| GOOD | 4/4 | 3/4 |
| NEED | 2/3 | 2/3 |
| LOVE | 5/5 | 5/5 |

ME improves by one; GOOD regresses by one. THANKYOU, YES, ME, GOOD and NEED
still contain errors. There are only 3–5 validation examples per class and
the settings were tuned on them. Fresh tracking of the same recordings checks
pipeline consistency, not new signers or webcam conditions. A controlled live
check and later frozen untouched-test evaluation remain necessary before
claiming a broader recognition improvement.

## Reproduction and evidence

Runners preserve previous outputs. New fitting needs a new output directory;
the grid runner reuses completed candidates.

```powershell
.\.venv\Scripts\python.exe scripts/run_please_trajectory_fix.py
.\.venv\Scripts\python.exe scripts/check_trajectory_feature_readout.py
.\.venv\Scripts\python.exe scripts/refit_trajectory_readout.py
.\.venv\Scripts\python.exe scripts/compare_trajectory_readouts.py
.\.venv\Scripts\python.exe scripts/check_trajectory_readout_ablation.py
.\.venv\Scripts\python.exe scripts/verify_please_trajectory_runtime.py
.\scripts\check.ps1
```

- [Joint training](assets/please-trajectory-fix-2026-11-12.json)
- [Scalar-only check](assets/trajectory-readout-check-2026-11-12.json)
- [Initial C=1 readout](assets/please-trajectory-readout-2026-11-12.json)
- [Complete grid/selection](assets/please-trajectory-readout-study-2026-11-12.json)
- [Matched 17/23-feature ablation](assets/please-trajectory-readout-ablation-2026-11-12.json)
- [Application/fresh-video evidence](assets/please-trajectory-runtime-2026-11-12.json)

The notebooks record this architecture and development comparison, with
earlier 17-feature and full-vocabulary results retained as historical records.

## Subsequent live feedback

The [first live check](trajectory-readout-live-feedback-2026-11-12.md) reports
correct GOOD, HELP and PLEASE first guesses, but five explicitly matched ME
attempts all predict WE first and ME second (ME 0/5). This confirms a remaining
live failure despite the offline PLEASE/MY target being achieved. See that
record for capture IDs, scores, tracking measurements and diagnostic limits.
