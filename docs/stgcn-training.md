# Independent ST-GCN27 training experiment

Authorized on **2026-11-10**: train ST-GCN and compare its recognition results.
The GRU models/archive, raw landmarks, prepared cache and investigation records
are preserved. No landmark extraction is repeated.

## Design and declared differences

This project implementation follows the 27-node XY topology and ten-block
spatial/temporal-convolution backbone described in
[ASL Citizen's configuration](https://github.com/microsoft/ASL-citizen-code/blob/main/ST-GCN/stgcn_training.py)
and [ST-GCN](https://arxiv.org/abs/1801.07455).
It is **trained from scratch**, not initialized from the authors' checkpoint.

The existing `asl-gru-128-v1` cache already stores body-relative XY for 13 pose
joints and all 42 hand joints. A batch adapter selects seven pose joints and
ten joints on each anatomical hand, producing `[batch,2,128,27]`. There is no
new full-corpus feature extraction or duplicated cache. The graph uses three
independently column-normalized self/inward/outward adjacency partitions and
learnable edge importance. This differs from the reference's graph partition
implementation, detector, exact normalization/padding, optimizer and precision;
it is not a reproduction of the authors' published scores.

The classifier has **3,788,165 trainable parameters**: ten graph/temporal
blocks (64/128/256 channels, temporal kernel 9, two stride-2 transitions),
masked temporal/node average pooling and a linear 256→2,731 head. Padding is
zeroed before/after blocks and excluded from pooling using propagated lengths.
The input adapter uses XY geometry; it does not consume the GRU's local hand
XYZ or explicit missingness feature channels. Missing coordinates remain zero.

Therefore this compares a different model **and** graph representation,
not solely a GRU/ST-GCN layer swap. Better live performance is not assumed.
The diagnosed hand-visibility, tracking and camera-detail issues still matter.

## Preset protocol

| Setting | Value |
| --- | --- |
| Vocabulary | All 2,731 ASL Citizen labels |
| Train/validation/test | Official signer-disjoint 40,154 / 10,304 / 32,941 |
| Frames | At most 128; unchanged whole-sign sampling in prepared cache |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.001 |
| Training augmentation | Same small XY roll ±5° / body-scale 0.9–1.1 profile as the regularized GRU |
| Dropout | 0.5 before the final linear head |
| Batch / loading workers | 32 / 2 |
| Budget / early stopping | Up to 50 epochs / patience 10 |
| LR scheduler | Halve after three stale validation-loss epochs |
| Seed | 42 |
| Selection | Minimum official validation cross-entropy |
| GPU/backend | RTX 4060, CUDA AMP; cuDNN enabled for convolutions after a successful separate smoke test |

Train and validation cache checksums, labels, official row order and signer
metadata are verified before training. The test split is excluded from
training/selection; the selected frozen checkpoint is evaluated once after
training. The previous GRU test score is already known; this is a project
comparison, not a claim of a previously unseen research benchmark.

The full check covered all 40,154 training and 10,304 validation clips.
A verification receipt permits reuse only while array/index size and modification
times, preparation-marker bytes, metadata and relevant source identities agree.
It avoids re-reading the same multi-gigabyte payload on the immediate job launch.
An incompatible receipt triggers the full checksum check again.

## Verification and throughput probe

1. Added graph/model tests for joint ordering, connectivity/normalization,
   padding-value invariance and gradients reaching early frames and the head.
   All **54 project tests passed** initially.
2. A separate GPU smoke run completed eight training and four validation
   batches, with finite loss, zero skipped AMP updates and clean process exit.
   Peak allocated CUDA tensor memory: **1,446,283,776 bytes**. Its tiny sample
   is not an accuracy result. Files: `models/asl-stgcn27-smoke/`.
3. Real-feature training-step probe (3 warm-up + 5 measured steps):

| Batch | Mean steady step | Samples/second | Peak allocated CUDA bytes |
| --- | ---: | ---: | ---: |
| 32 | 0.0876 s | 365.3 | 1,446,719,488 |
| 64 | 0.1799 s | 355.8 | 2,798,563,840 |

Batch 32 was retained for slightly higher sample throughput and lower memory.
This excludes loading and full validation; estimate the full run from complete
epochs, not the smoke run's initialization time. Probe output:
`.cache/stgcn-batch-benchmark.json`.

## Background job and artifacts

```powershell
.\.venv\Scripts\python.exe scripts/local_stgcn_job.py run
.\.venv\Scripts\python.exe scripts/local_stgcn_job.py status
.\.venv\Scripts\python.exe scripts/local_stgcn_job.py stop
```

Run is launched with a hidden console. Stop requests terminate only this job's
process tree; resume restarts from the last complete epoch checkpoint. Atomic
checkpoint writes preserve committed epochs. A supervisor lock prevents two
instances of this job. The temporary Windows idle-sleep request is released
when the job exits; user-initiated sleep/shutdown still interrupts computation.

- Model/config/labels/epoch records: `models/asl-stgcn27-v1/`.
- Selected and resume weights: `best.pt`, `last.pt`; no GRU outputs reused.
- Immutable engine copy: `models/asl-stgcn27-v1/engine_snapshot/`, validated
  against its manifest on train/resume/evaluation.
- Supervisor, pipeline stage and logs: `.cache/stgcn-training/`.
- Frozen test artifacts: `data/evaluations/asl-stgcn27-test-v1/`.
- A results table is generated as `docs/stgcn27-results.md` only after successful
  training and frozen test evaluation. It compares validation/test metrics with
  the regularized GRU and makes no webcam-accuracy claim.
- The results reporter then regenerates two executed reporting notebooks under
  `notebooks/`, with embedded figures and a portable snapshot. This completion
  hook is in the future results subprocess; the active trainer/frozen engine
  was not restarted or modified to add it. See [notebook guide](../notebooks/README.md).

An identical completed test report is reused on a pipeline resume; a different
existing evaluation is refused. No test artifacts or trained model are pushed
to Git. Do not edit the frozen engine or current run configuration.

After successful training, a separate user-attended webcam check can use:

```powershell
.\.venv\Scripts\python.exe scripts/recognize_webcam.py --checkpoint models/asl-stgcn27-v1/best.pt
```

This is the same isolated-sign capture path; no webcam check starts automatically.

## Current state

**Completed successfully on 2026-11-11 at approximately 01:26 Asia/Tashkent**,
supervisor exit code 0. Training stopped early after 49 epochs; epoch 39 was
selected by minimum validation loss. Test top-1 **57.25%**, top-5 **82.32%**,
macro F1 **56.74%**, loss **2.08828**, over all 32,941 test clips/11 signers.
Full job wall time was about 104.9 minutes; summed epoch time was 103.1 minutes.
See [the generated results table](stgcn27-results.md).

The last epoch reached higher validation top-1 (68.85%) than selected epoch 39
(67.23%), but worse validation loss (1.44400 vs 1.39338). The predeclared
minimum-loss selection rule was retained; no additional checkpoint was tested
in response to its test score. Both reporting notebooks contain final outputs.
The ST-GCN webcam check has now been performed. In the corrected session 02,
the user reported wrong first guesses for hello, thank you, no, me and yes;
the prediction names match the saved ST-GCN log. No formal live accuracy or
improvement is established. See [the live-test record](stgcn-webcam-test-2026-11-11.md)
for feedback, the fixed-grid inference correction and the resolved misleading
GRU UI label. All model weights and offline metrics are preserved.

## Action record

1. Reviewed existing training, cache, preprocessing, job controls, evaluation
   and preserved model state; checked the Microsoft reference graph/backbone.
2. Added the graph adapter, model and a separate trainer, reusing the existing
   loss/metric/AMP/checkpoint helpers. Original GRU model and training modules
   were not edited.
3. Extended checkpoint dispatch in evaluation/webcam loading to support
   ST-GCN while retaining legacy GRU behavior. The frozen-evaluation test now
   exercises both model dispatch paths.
4. Passed the CPU/unit suite, full train/validation integrity verification,
   saved-checkpoint CUDA reload and GPU training smoke. Raw/prepared data were
   not modified. Benchmark script: `scripts/benchmark_stgcn.py`.
5. Added an independent hidden supervisor with stop/resume, frozen source,
   post-training evaluation and a generated results report. Verified no other
   project trainer was running before launch. The supervisor's pipeline moves
   through cache verification → training → frozen test → results report.
6. Initial running observation: epoch 1 reached at least 200/1,255 training
   batches, GPU utilization approximately 95%, total device memory reported
   around 2,247 MiB. One initial AMP overflow lowered the scaler and skipped
   that optimizer update; subsequent batches continued with finite losses.
   This is recorded as a recovered numeric event, not a training failure.
7. No Git push, webcam launch, RSL training or change to existing GRU/archive
   weights was performed. The earlier diagnosis's file hashes describe its
   source versions at that time; model-dispatch source has since changed for
   this explicitly authorized experiment.
8. First full epoch committed successfully: 40,154 train / 10,304 validation
   examples, train loss 7.88986, validation loss 7.70266, validation top-1
   0.1359%. Training took 115.48 s and validation 13.91 s; total epoch time
   129.44 s. Epoch 2 then started. These are early learning diagnostics, not
   the trained model's selected/final accuracy. A first-epoch extrapolation is
   approximately 108 minutes for 50 epochs plus startup/test/report overhead;
   later throughput and early stopping can change that duration.
9. Final setup checks: all 54 tests passed, the smoke checkpoint reloaded and
   returned finite `[2,2731]` CUDA outputs, all train/validation checksums
   passed, frozen engine hashes and local document links were verified, and
   `git diff --check` passed. The GRU best checkpoint and its ZIP archive still
   match the SHA-256 values recorded before this experiment.
10. The user subsequently requested two notebooks covering completed work and
    metric graphs after training. Added a read-only builder, draft outputs with
    14 embedded figures, a portable metric snapshot and PNG exports. The existing
    results subprocess now calls the builder with `--require-complete` after
    frozen evaluation. The final notebook refresh rejects pending/incomplete
    results. No notebook/kernel packages or training dependencies were changed
    while GPU training was active.
11. Completion verification on 2026-11-11 independently streamed all 32,941
    test predictions against the official index, reconstructed the confusion
    matrix and recomputed top-1/top-5 and macro/weighted metrics. All per-class
    and per-signer exports matched. Correct counts: 18,858 top-1; 27,117 top-5.
    The selected checkpoint SHA-256 is
    `6617c495405197dd4c232e4a27e2a438fd698c8f9ba6b0d1edc1b795b5e86fad`.
    It and the GRU/archive hashes remain unchanged. Loss was taken from the
    full-logit evaluation report, not inferred from truncated top-five scores.
12. Added final-notebook summaries, selection explanation and paired class
    analysis. Overall test top-1 improves by 7.33 percentage points, but HELLO
    falls from 8/14 correct (GRU) to 5/14, THANKYOU from 9/12 to 4/12, and NO
    stays at 7/11. These small class samples do not establish webcam accuracy.
    Reporting refreshes do not rerun training or test inference.
