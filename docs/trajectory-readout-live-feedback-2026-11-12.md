# Trajectory readout: first live feedback

Selected model: `asl-gru-fifteen-signs-trajectory-readout-c01/best.pt`,
`SignTrajectoryReadoutGRU`, inherited backbone epoch 13, CUDA, 15 labels.
SHA-256: `34b766c393cf20a77b001004376195e60ffb58943d7b1f9661bd23edbab81c91`.
Session: `webcam-trajectory-c01-20261112-234320`.

The user reports GOOD, HELP and PLEASE as correct first guesses. They report
ME ranked second behind WE, and explicitly confirm that captures **4, 7, 8,
10 and 11** were ME attempts. These five matched attempts score **ME 0/5**,
all predicted WE first. Other captures have no confirmed per-ID labels;
do not infer their ground truth or calculate overall session accuracy.

| Capture | Intended | WE score | ME score/rank | Duration | Frames with hands |
| ---: | --- | ---: | --- | ---: | ---: |
| 4 | ME | 96.01% | 1.06%, second | 2.94 s | 41/86 (47.7%) |
| 7 | ME | 64.75% | 30.99%, second | 7.91 s | 92/226 (40.7%) |
| 8 | ME | 76.16% | 10.74%, second | 5.80 s | 137/157 (87.3%) |
| 10 | ME | 88.64% | 9.79%, second | 8.00 s | 51/240 (21.2%) |
| 11 | ME | 87.45% | 6.31%, second | 8.01 s | 171/240 (71.2%) |

Scores are uncalibrated softmax values, not probabilities of correctness.
Actual retained FPS is 26.88–29.87 for these five attempts. Shoulder/body
anchors are present in every captured frame. Capture 4 is already short;
capture 8 has high hand-presence coverage. Thus neither a long capture ending
nor absent hands alone explains all failures. Presence counts cannot establish
joint accuracy, physical hand identity, or whether hands were intentionally
outside the image. No personal video or joint coordinates were saved, so the
exact failed sequences cannot be replayed or geometrically inspected now.

The [selected readout](please-trajectory-fix-2026-11-12.md) scored ME 4/5 and
WE 3/3 on the known development validation videos. Those small, tuned offline
results do not establish live generalization. This session provides evidence
of a repeatable ME-to-WE live confusion, not a proven source-data defect,
MediaPipe defect, overfitting diagnosis or one specific classifier-layer fault.
No new model, rule, threshold, default or training change was applied.

Next diagnostic: compare all ME/WE training/validation source videos with their
cached wrist/finger trajectories and fixed-head predictions, keeping class
labels and sources intact. Then use a controlled ME/WE live comparison to
separate capture timing from the available hand/body motion representation.
Any new feature/model change should follow those checks, retain all 15 classes
and be evaluated for regressions on PLEASE/MY and the other classes.

[Portable session snapshot, explicit user-confirmed labels and measurements](assets/trajectory-readout-live-feedback-2026-11-12.json).
The snapshot records predictions and aggregate counts only. The webcam session
may continue after that snapshot; later captures need their own confirmed labels.

Regeneration (refuses to overwrite existing evidence):

```powershell
.\.venv\Scripts\python.exe scripts/report_trajectory_live_feedback.py
```
