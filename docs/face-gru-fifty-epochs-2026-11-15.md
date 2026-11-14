# Fifteen-sign face-aware GRU: 50 epochs without early stopping

At the user's request, the 2×256 hand/pose/face GRU was trained from scratch
for **all 50 epochs**, with no early-stopping condition. The previous 17-epoch
model and data remain unchanged. The input representation, splits, seed,
augmentation, optimizer and final C=0.1 frozen-readout fitting procedure are
the same as the prior face-aware experiment.

The actual model and training code, executed cells, learning curves and
per-sign scores are in [notebook 08](../notebooks/08_15_sign_face_gru_50epochs.ipynb).
An in-progress recovery checkpoint was maintained and removed after successful
completion. Only the selected final model remains in the new run directory.

## Checkpoint selection

Training reached epoch 50; it was **not** stopped at epoch 36. After all 50
epochs, the pre-existing validation-competitive/small-gap rule selected
backbone **epoch 36** from eligible epochs 32, 36, 44, 48, 49 and 50. The
selected backbone was frozen and its linear readout fitted on training data.
The test set did not choose the epoch.

At epoch 50, the training-stage classifier measured 92.69% on clean training
and 85.00% on validation; it was not the final refitted readout. At epoch 36,
that classifier measured about 84.93% on clean training and 85.00% on validation.
The post-fit results below are the separately refitted readout on epoch 36.

| Face-feature run | Epochs run | Selected backbone | Validation top-1 | Test top-1 | Test macro F1 |
|---|---:|---:|---:|---:|---:|
| Earlier, early stopping | 17 | 17 | 53/60 (88.33%) | 141/182 (77.47%) | 78.00% |
| 50 epochs, no early stopping | 50 | 36 | 50/60 (83.33%) | 151/182 (82.97%) | 83.07% |

The same 60 validation and 182 test videos were used. The longer run's test
score is higher and its validation score lower; samples are small and both
splits have been inspected in previous development. This is one seed, not proof
that 50 epochs generally improve generalization. No webcam test had been
performed when these offline scores were recorded.

The saved model reloaded successfully and reproduced **151/182** with matching
test loss. The original face-aware model's hash remained unchanged. The new
checkpoint, configuration, labels, history and report are stored under
`models/asl-gru-fifteen-signs-face-50epochs-v1/`; they are local and Git-ignored.
The older curated archive does not contain this new model.

## Webcam field test

Run `python scripts/webcam_15_face_subtitles.py`. The window identifies the
50-epoch run and selected epoch 36. Space starts a three-second isolated-sign
capture; Q closes the window. Exactly one top guess is drawn as a subtitle. Face,
shoulders and hands should be visible. No video or landmarks are saved; the
local session report contains predicted labels and timing. The existing default
hand/pose model remains separate.

Before opening the field-test window, the new inference adapter reproduced all
60 notebook validation inputs exactly, and the same 50/60 validation predictions.
An original validation video passed through all three detectors and reproduced
its cached-input top prediction and score. These are preprocessing checks, not
evidence of webcam accuracy. Training/validation were not rerun.

After the field test, the user reported that YES and NO were recognized a bit
better and described this model as their most stable snapshot so far. Treat
this as qualitative feedback: intended labels were not recorded per capture,
so the session cannot establish a live accuracy or a controlled improvement.
The user chose to keep this snapshot as the current field-test choice; its
weights remain unchanged. Use the face-aware subtitle command above to select it.
