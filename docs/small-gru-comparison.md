# Controlled 64-unit GRU comparison

On 2026-11-10 the user selected **64 hidden units per layer**, and explicitly
chose to queue the full run after the existing cross-validation study.

**Completed successfully on 2026-11-10:** 50 epochs, selected epoch 49,
validation top-1 **45.89%**, training top-1 47.82%. The smaller gap came with
lower validation accuracy than either 256-unit model. See
[the completed comparison](model-comparison-results-2026-11-10.md), including
the remaining possibility of slower convergence within the 50-epoch budget.

## Experimental controls

Compare against the original official-split baseline, not against models from
different CV folds. Change only the hidden size and the resulting classifier
input width:

| Setting | Original baseline | Small GRU |
| --- | --- | --- |
| GRU layers | 2 | 2 |
| Hidden units per layer | 256 | 64 |
| Trainable parameters | 1,489,835 | 263,915 |
| Linear head | Linear(256, 2731) | Linear(64, 2731) |
| Dropout | 0.3 | 0.3 |
| AdamW weight decay | 0.0001 | 0.0001 |
| Augmentation | None | None |
| Sequence limit | 128 | 128 |
| Batch / DataLoader workers | 64 / 2 | 64 / 2 |
| Learning rate / seed | 0.001 / 42 | 0.001 / 42 |
| Epoch limit / early-stopping patience | 50 / 10 | 50 / 10 |
| Checkpoint selection | Minimum validation cross-entropy | Same |

Both use the unchanged prepared feature cache, official training/validation
splits, cross-entropy, CUDA AMP, gradient clipping and the verified ATen CUDA
RNN backend. The smaller network starts fresh; smoke-test weights and CV
weights are not used to initialize it. The test split remains unused.

The small model has about **82.3% fewer parameters**. This is a capacity
experiment, not a guarantee of improved generalization. Compare validation
loss, top-1/top-5 and macro F1 at the declared selected checkpoint. A smaller
training/validation gap alone could reflect underfitting. The reference
baseline's selected validation top-1 is 55.73%, macro F1 54.67%.

## Queue, status and recovery

The 64-unit CUDA smoke test passed and exited cleanly. Baseline hyperparameters
were checked against the original saved configuration. All **41 checks** passed,
including parameter count, output shape, queue ordering, failed-CV handling and
frozen CV source routing.

The hidden background supervisor waits for **successful completion of the
entire current study**, including its official-split refit, before launching
the full 64-unit run. It does not compete with CV as a second full GPU trainer.
If CV fails or is stopped, the queue fails without starting the comparison.
Training code is checked against the smoke-test hashes before launch; source
changes while queued are rejected.

```powershell
.venv\Scripts\python.exe scripts/local_small_gru_job.py status
.venv\Scripts\python.exe scripts/local_small_gru_job.py stop
```

Status includes `queue.status` (`waiting_for_cv`, `training`, `completed`, or
`failed`) and model epoch metrics once training starts. Stopping this supervisor
cancels its queue/current small-model run; the separate CV study is unaffected.
The supervisor prevents idle sleep while active without changing the power plan.
No desktop notification is implemented.

Restart the same job after an interruption:

```powershell
Start-Process -WindowStyle Hidden `
  -FilePath "$PWD\.venv\Scripts\python.exe" `
  -ArgumentList @('-u', "$PWD\scripts\local_small_gru_job.py", 'run') `
  -WorkingDirectory "$PWD"
```

Matching `last.pt` resumes the last complete epoch; an interrupted epoch repeats.
Fresh output is `models/asl-gru-64-baseline/`, separate smoke output is
`models/asl-gru-64-smoke/`, and logs/control state are in `.cache/small-gru/`.
Completed checkpoints and the raw/normalized caches are preserved.

Manual full-run equivalent (do not run concurrently with the queued job):

```powershell
.venv\Scripts\python.exe scripts/train_local.py train --hidden-size 64 --profile baseline --workers 2 --epochs 50 --batch-size 64 --patience 10 --output models/asl-gru-64-baseline
```

## Preserving the ongoing CV engine

Before adding the hidden-size argument, the current source was copied to
`models/asl-gru-128-cv-regularized-v1/engine_snapshot/`. Its five recorded
training-source hashes match the running CV smoke configuration. Future CV
stages and resumes route through those frozen sources; the snapshot path is
inherited by spawned DataLoader workers. CV remains at 256 hidden units and
keeps its original regularization, configuration identity and checkpoint guards.
The source snapshot is retained with ignored model artifacts. The original
selected baseline weights and all prepared features are unchanged.
