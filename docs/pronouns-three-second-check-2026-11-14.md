# Three-second pronoun check — 2026-11-14

The frozen 15-class trajectory-readout GRU completed six guided captures in the order ME, WE, YOU, ME, WE, YOU. The user identified **#2 as LOVE performed instead of WE**. It is excluded from prompted WE evaluation. Their clarification identifies the other five as the unaffected trials; those labels are recorded as confirmed. The original report and intermediate protocol-feedback record remain preserved under `.cache/`.

All 15 labels competed. Each input was capped at three seconds, excluding late camera frames. The retained frame cadence was approximately 29–30 FPS. Recognition tracked each capture once; all windows and controls reuse those observations. No live images, coordinates or embeddings were persisted. No model training occurred.

| Capture | Confirmed sign | Full-window first guess | Model score |
| --- | --- | --- | --- |
| 1 | ME | WE | 91.6% |
| 2 | LOVE instead of prompted WE | LOVE | 93.7%; excluded |
| 3 | YOU | YES | 80.3% |
| 4 | ME | WE | 84.9% |
| 5 | WE | LOVE | 51.5% |
| 6 | YOU | YES | 33.6% |

These are model softmax scores, not calibrated correctness probabilities. Five valid captures are insufficient for a general accuracy estimate. The previous twelve free-form captures remain unlabelled because the user could not recall their intended signs; see [qualitative feedback](three-second-live-feedback-2026-11-14.md). Earlier incorrectly performed WE trials remain excluded under their separate correction ledger.

## Demonstrated mechanisms and matched controls

- **ME:** Both valid captures remain WE in the full sequence, first-two-second proxy and detected-hand envelope, with either frozen model. Disabling the original 17 geometry features, the six path features, or all geometry in the current head does not change this. Repeating the same active frame makes ME the first guess on both attempts across all three windows. Sequence/context sensitivity is reproducible; static repetition is a destructive diagnostic, not a recognition fix.
- **ME margin:** On full capture #1, WE-minus-ME is +2.816: bias +0.677, pooled GRU +1.035, original geometry +0.209, path geometry +0.895. On #4 it is +1.878: +0.677, +0.376, +0.553 and +0.272 respectively. Every group favors WE on those full inputs; removing geometry alone cannot reverse these two decisions.
- **WE:** The sole valid WE (#5) changes from LOVE at 51.5% on the full sequence to WE at 36.8% on the first-two-second proxy. The hand envelope still gives LOVE, narrowly ahead of WE. This demonstrates a window-dependent decision on one valid WE example, not a reliable two-second solution.
- **YOU:** Both full sequences give YES. On #3, the hand envelope narrows YES-minus-YOU to +0.443; its pooled GRU contribution favors YOU (-0.729), while bias (+0.496), original geometry (+0.670) and path geometry (+0.006) favor YES. Disabling original or all geometry on that matched envelope produces YOU; the original temporal model also gives YOU. This particular readout decision is demonstrably affected by geometry and bias. The same intervention does not correct full #3 or either geometry ablation on #6.
- **Head replacement is not a universal explanation:** Original and current heads both give WE for ME. The original head gives YOU on full #6 where the current head gives YES, but fails on other matched inputs. Both heads use identical frozen GRU weights; switching heads is not justified by these few trials.
- Reversal, deterministic shuffling, anatomical reflection, hand-slot swapping and active-frame repetition were scored with recalculated geometry in every available window. Hand-slot swapping produces YOU on both full YOU trials; reversal does so on #6 only. These transformations establish sensitivity to sequence and slot representation; they do not establish that the original slot assignment was anatomically incorrect.

The complete scores, interventions, margin decompositions, timestamps-derived aggregates and user-feedback record are in [the portable results](assets/pronouns-three-second-check-2026-11-14.json). Full versus prefix and envelope comparisons refit normalization on their own raw observations before 128-frame sampling. They can therefore change preprocessing context as well as content. These proxies are not verified sign boundaries. Fixed-context normalization controls belong to the separate earlier investigation.

## Supported hypotheses and missing measurements

The ME decisions are not caused solely by the additional readout geometry or a capture longer than three seconds. Both heads and all geometry ablations reproduce them, while static repetition reverses them. A mismatch in movement, onset/ending content, temporal encoding, or tracking over time remains plausible. Presence counts cannot distinguish these alternatives or prove accurate joint positions.

YOU shows both sequence/slot sensitivity and a specific geometry-driven decision on one cropped input. It is not established that YOUR/YOU depth cues, hand orientation, signing style or detector quality caused these errors. The YOU source clips have not yet received the detailed frame-by-frame audit already performed for ME/WE.

**Smallest justified next step:** keep the frozen model and three-second interface; audit public YOU source videos against their cached landmarks, then use an explicitly labelled in-memory comparison of a held pronoun pose and the same pose with its natural entry/exit. Compare successive window decisions and aggregate motion without persisting live coordinates. This measures whether the correctly recognized static posture is lost during motion/context. No default geometry suppression, slot swap, static repetition, retraining or new checkpoint is justified yet.

## Verification

The current and original checkpoints match their declared SHA-256 values, and all trained source-file hashes match the checkpoint configuration. Both checkpoints and the normalizer were hashed before and after reporting with identical results. Source videos and cached features were not changed or processed by this live runner. The earlier complete video/cache integrity audit remains a separate historical check.

Seventeen scoped tests passed for guided-runner lifecycle, errors, privacy, timing and strict automatic capture, including the new six-prompt three-second path and delayed-frame exclusion. The read-only reporting notebook includes this session with #2 excluded, plotted scores and explicit limitations. No corpus cleaning, official test evaluation, Git push, model replacement or archive changes occurred.
