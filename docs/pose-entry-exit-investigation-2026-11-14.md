# Pronoun pose versus entry/exit investigation — 2026-11-14

## Scope and recorded steps

1. Preserve the frozen 15-class trajectory-readout checkpoint and original temporal-head comparator, existing archive and prior reports. Do not train, clean, relabel source data or evaluate the official test split.
2. Extend the existing read-only audit runner with explicit label/output arguments. Audit all 19 YOU training/validation clips with the same checks as ME/WE, adding a source-video CRC/size comparison to the original download ZIP.
3. The first launch stopped because a relative contact-sheet directory could not be expressed relative to the absolute workspace root. Preserve its partial JSON as `docs/assets/you-source-audit-2026-11-14.partial-path-error.json`, the initial log and `.cache/you-source-audit-20261114-v1/`. Resolve supplied output paths and run into a new v2 board directory. This was a reporting-path error; no input data changed.
4. Completed [YOU audit](assets/you-source-audit-2026-11-14.json): 15 training and four validation videos decode with matching FPS/frame count/shape; original ZIP CRC/size matches; raw archive checksums/record identities and prepared label indices match; raw normalization exactly reproduces cached float16 features. All 60 saved validation predictions reproduce with maximum score difference zero, including the model's original 54/60 total. All 58 protected hashes remain unchanged. YOU itself is 15/15 training and 4/4 validation.
5. Visually reviewed contact sheets for all four validation examples (P39, P12, P21, P26) and three training examples (P52, P33, P30). Forward-pointing fingers are foreshortened in the source frames. P39 frames 61 and 64 show a visible forward hand with no skeleton overlay, while other sampled frames have finger landmarks. These are sampled observations, not a quantified keypoint error rate. Low whole-clip hand coverage also includes rest intervals; it does not by itself measure tracking failure. No semantic mislabel or corrupted source/cache example was established.
6. Add a separate diagnostic helper and runner mode for ME/YOU held-posture versus natural entry/exit. Keep the default recognizer, preprocessing and model weights unchanged. The mode tracks each capture once in memory, then compares the same observations under raw-first cropping, refitted versus full-context-fixed normalization, both heads, destructive temporal/slot/orientation controls and standardized geometry ablations. Save scores, logit contributions, timing and aggregate motion only.
7. Before webcam capture, 22 scoped tests passed. They cover raw-timestamp phase boundaries, fixed-context normalization, missing detections, no persisted arrays, single-pass recognition tracking and guided eight-trial lifecycle with a strict three-second input cap. A separate real-source preflight is recorded in the run log.

## Live protocol

Run `scripts/diagnose_pronouns_live.py --pose-entry-exit --report NEW_PATH`.
All 15 labels compete, and expected labels/conditions are attached only after inference.

Eight trials, with the sequence repeated twice:

| Trial in repetition | Label | Condition |
| --- | --- | --- |
| 1 | ME | Form the posture during GET READY, then hold throughout capture |
| 2 | ME | Start relaxed; sign once in the first two seconds, then lower hand |
| 3 | YOU | Start relaxed; sign once in the first two seconds, then lower hand |
| 4 | YOU | Form the posture during GET READY, then hold throughout capture |

Press Space once per prompt. Preparation lasts 2.5 seconds; capture ends automatically at three seconds. Use the same signing hand, posture and framing throughout. ME means index pointing into the chest; YOU means index toward the camera/addressee. Held posture is a deliberate diagnostic manipulation, not a natural isolated-sign accuracy test. Conditions are separate performances, so repeated condition effects are suggestive rather than a perfectly matched movement intervention.

Each capture is scored as full, first two seconds, detected-hand envelope, first/middle/last one-second windows and last two seconds. Crops precede normalization and 128-frame sampling. Every crop also receives a full-context-fixed normalization control, which uses future context and is retrospective, not a streaming implementation. Track visibility and signed motion using actual timestamps without joining detection gaps. Aggregate phase motion helps check whether a held trial was steadier; it does not establish accurate tracking or exact sign boundaries.

User confirmation is required before prompted labels/conditions enter any evaluation. Wrong gestures, missed timing and interrupted captures are excluded separately. Do not treat the earlier WE-to-LOVE mistake as a WE failure. No live video, stills, coordinates or embeddings are saved.

## Decision rules and current status

The user confirmed all eight gestures, conditions and normal capture. The session completed and released the camera. Both frozen checkpoint hashes remain unchanged. [Full aggregate results](assets/pose-entry-exit-results-2026-11-14.json) include scores, controls, margin components, phase motion and the complete confirmed trial record. [The phase plot](assets/pose-entry-exit-results-2026-11-14.png) shows unavailable windows as gaps.

| Capture | Label / condition | Full result |
| --- | --- | --- |
| 1 | ME hold | Unscorable: zero usable hand **and body-anchor** frames |
| 2 | ME natural | PLEASE 51.2%; ME third 11.7% |
| 3 | YOU natural | YES 46.0% |
| 4 | YOU hold | YES 78.0% |
| 5 | ME hold | ME 76.0% |
| 6 | ME natural | ME 87.8% |
| 7 | YOU natural | YES 92.0% |
| 8 | YOU hold | YES 61.1% |

These are uncalibrated softmax scores. Capture #1 retained 46 frames at approximately 15.2 FPS; the remaining captures retained 86–90 frames at 28.6–29.7 FPS. All retained time spans were below three seconds. The user-confirmed condition is preserved for #1, but its unavailable observation stream cannot establish a model decision. The saved aggregates do not identify whether its failure arose from image quality, framing, detector initialization or another capture/tracking issue.

### Demonstrated decision mechanisms

- **No repeated hold benefit.** The one usable ME hold (#5) and its repeated natural counterpart (#6) both work. Their tracked projected wrist path lengths are almost identical (0.409 versus 0.403 shoulder-width units), so this repetition does not establish a meaningful contrast in measured movement. The first hold is unavailable and cannot support the comparison. Both YOU holds and both natural trials fail.
- **YOU persists under static and head controls.** All four full YOU sequences remain YES under temporal reversal, deterministic shuffle, repeated active frame, anatomical reflection and each geometry-group ablation. The original frozen temporal head also gives YES on all four. In the current head, pooled-GRU contributions to YES-minus-YOU range from +3.73 to +4.84 logits, while the fixed bias contributes +0.496. Geometry favors YOU overall on two of these four full sequences and still cannot overturn the recurrent contribution. Therefore neither motion order alone nor the added geometry readout alone explains these YOU failures.
- **No fixed-normalization repair.** All 42 scored, matched refit/fixed window pairs have the same first guess. For #2 ME, the first and last one-second proxies give ME, the middle second gives WE, and the full sequence gives PLEASE. The same phase pattern survives fixed normalization. Window/content-dependent recognition is demonstrated; it does not provide verified sign boundaries or a universally correct crop.
- **Slot swap is inconsistent.** Swapping hand slots gives YOU on full #4 but GOOD on the other three full YOU captures; it damages the correct ME hold. This is sensitivity, not proof that the original slots were wrong. The dataset audit likewise shows that swapping slots reduces YOU results from 15/15 to 5/15 training and 4/4 to 3/4 validation.

### Supported hypotheses and remaining measurement

YOU fails even on real held postures and repeated-frame controls. This shifts attention toward the recorded hand/body representation and its generalization to these webcam postures. It does not establish whether the detector estimates the pointing finger correctly or the trained GRU misinterprets an accurate estimate. One hold (#4) has a 27-frame missing-hand span, while #8 has continuous hand presence and still fails. Presence counts cannot establish accurate joint geometry.

The public YOU training examples predominantly occupy canonical slot 1 (13/15); only two occupy slot 0, whereas these four live YOU trials occupy slot 0. This is a descriptive distribution difference, not a proven cause: two slot-0 validation examples are correctly recognized, anatomical reflection does not repair these live inputs, and a simple slot swap is inconsistent.

**No model correction is justified yet.** Keep the current weights, three-second cap and preprocessing. The smallest missing measurement is an annotated, in-memory inspection of the actual live YOU finger estimates alongside a public YOU reference, checking index-tip placement, foreshortening, palm scale and body-relative location. The current aggregate-only logs cannot answer this because no live coordinates/images were retained. Image review would require a separately agreed privacy scope; it has not happened in this experiment. ME needs a repeatable movement/pose contrast before claiming entry/exit is its cause.

A repeated hold-versus-moving difference would support sensitivity to motion/context; fixed versus refitted comparisons can locate preprocessing-context effects. Neither proves which joint estimates are correct. A readout intervention must reproduce its effect on identical observations; changed predictions under swapped slots or static repetition are diagnostic sensitivity, not an approved fix.

If correct held postures still fail with stable tracking aggregates, review the live landmark geometry before assigning the cause to the sequence model. If real held postures work and natural motion fails repeatedly, the next missing measurement is the exact transition/phase at which the competing logit margin changes. No retraining, replacement model or default preprocessing change is authorized in this investigation.

## Final verification and deliverables

After confirmed capture, 31 scoped tests passed, including the report's separation of unscorable, wrong, pending and excluded trials. The real-source preflight's largest margin-reconstruction residual was approximately 3.6e-7. All 58 YOU-audit protected hashes were rechecked after the complete investigation and still match. The webcam is closed.

The read-only [reporting notebook](../notebooks/02_model_training_and_evaluation.ipynb) now includes this audit and the confirmed pose/phase plot, with no notebook execution errors. All 30 generated artifact-manifest hashes were verified. The workflow notebook and previous investigation sections remain preserved. Detailed commands and execution logs are under `.cache/pose-entry-exit-*`; public YOU contact sheets are in `.cache/you-source-audit-20261114-v2/`. No Git push or archive change was performed.
