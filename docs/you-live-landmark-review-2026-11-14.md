# Geometry-enabled live YOU landmark review — 2026-11-14

The user authorized the proposed annotated, in-memory live YOU review and explicitly required geometry to remain enabled. This supersedes the previous experiment's aggregate-only image scope for this review. Four annotated boards from two captures may be visible in the chat; no live media or landmark-coordinate files are written to the workspace or training data. Chat-visible images form part of the conversation.

## Protocol and safeguards

- Use only the frozen `SignTrajectoryReadoutGRU` checkpoint, SHA-256 `34b766c393cf20a77b001004376195e60ffb58943d7b1f9661bd23edbab81c91`. All 15 labels compete. The unchanged production `predict_observations` receives its normal arguments; no geometry ablation, alternate head, temporal variant or slot transformation is evaluated.
- Two automatic captures, each at most three seconds: #1 natural YOU, signing once then relaxing; #2 YOU posture formed during preparation and held through capture. User confirmation is required before treating these conditions as performed labels.
- Recognition tracks each retained frame once. Annotation uses those same observations, with the frozen canonical hand assignments, rather than the separate asynchronous preview tracker. Select the nearest retained frames to 1.0 and 1.6 seconds independently of predictions. No sign-boundary claim is made.
- Display the original unmirrored camera frame, its tracked skeleton, enlarged hand crops and a public P12 validation YOU reference at source frame 18 from `6229294436435182-YOU.mp4`. Index joints 5–8 are numbered; index chain is green and tip 8 red. Slot numbers are assignments, not verified anatomical identity.
- Save only predictions, quality/timing and aggregate hand-shape ratios. Ratios describe estimated projected extension/foreshortening; they do not establish physical finger depth or tracking accuracy. PNG boards stay in a bounded four-image RAM buffer served only on loopback with a random URL token, no request logging and `Cache-Control: no-store`. Clear and shut down after review; expire after at most five minutes after the camera closes.

## Preparation and verification

1. Reuse the existing camera runner with an optional wrapper instruction line. Its default model, preprocessing and behavior remain unchanged. Add separate `pronoun_landmark_review.py` and `review_pronoun_landmarks.py`; do not reuse the earlier file-writing still exporter.
2. Sixteen scoped tests passed, covering normal production prediction arguments, no geometry intervention, one recognition tracking pass, bounded loopback images, no filesystem image writes, explicit clearing, source-array preservation, missing detections and three-second capture timing.
3. A real public-source preflight passed: the normal model predicted YOU, with a 279-value classifier input containing 256 GRU values plus all 23 non-disabled standardized geometry features. Exactly one normal model call was made with no forward keyword overrides. Two RAM boards were created, no images written, and checkpoint bytes remained unchanged. See [preflight record](assets/you-landmark-review-preflight-2026-11-14.json).
4. Launch session `you-landmark-review-20261114-132328`; startup verified the expected checkpoint/class and three-second cap. The user confirmed both conditions and normal capture, then clarified that the index in #1 was straight toward the camera.

## Confirmed observations

Both captures used geometry-enabled production inference throughout. Natural YOU (#1, 2.969 seconds) produced YES 67.4%; held YOU (#2, 2.980 seconds) produced YES 82.4%. These are uncalibrated softmax scores. No geometry-off prediction was evaluated in this review.

All four boards were visually inspected. In #1, index joints 5–8 are tightly compressed in projection; at 1.611 seconds the estimated tip-to-MCP distance is only 0.017 palm units, and the projected index endpoint chord is 0.063 of its joint-chain length. The user states the physical finger was straight toward the camera. Foreshortening and uncertain projected landmarks remain plausible; the stills do not provide calibrated 3D joint ground truth. In #2, the extended index is more visibly separated, yet recognition is still YES. Therefore projected collapse alone cannot explain both failures. The laterally extended live thumb also differs from the selected tucked-thumb reference, but this does not establish a semantic gesture error or a causal model mechanism.

| Sample | Projected tip-to-MCP / palm | Projected tip-to-wrist / palm | Projected chord / joint path | Estimated index z span / palm |
| --- | --- | --- | --- | --- |
| #1 at 0.969s | 0.229 | 0.956 | 0.975 | 0.348 |
| #1 at 1.611s | 0.017 | 1.126 | 0.063 | 0.477 |
| #2 at 1.013s | 0.501 | 1.573 | 0.943 | 0.302 |
| #2 at 1.615s | 0.516 | 1.576 | 0.963 | 0.275 |
| Public P12 frame 18 | 0.734 | 2.032 | 0.999 | 0.531 |

These are summaries of estimates, not coordinates or measured physical depth. Body framing, hand orientation and finger depth can change the projections. One public comparison frame is insufficient for a class-wide conclusion.

## Public YOU/YES shape check

To avoid treating one reference as representative, run `scripts/audit_pronoun_hand_shapes.py` on all 19 YOU and 20 YES training/validation clips. Use the fixed middle tracked frame and whole-clip dominant slot without consulting shape values; read existing raw landmarks, perform no new MediaPipe extraction and no model inference. Record raw-archive identity/checksums and protected cache hashes. These frame proxies may be rest/transition, not verified sign apexes.

The public-probe projected tip-to-wrist median is 2.094 for YOU and 0.777 for YES, with substantial overlapping ranges. The first capture's 1.611-second projected tip-to-MCP distance lies below all 19 selected YOU probes, while the second capture's values overlap public YOU/YES descriptors. Estimated index z also differs, but is detector output rather than physical depth. No descriptor threshold, gesture relabeling or classifier is fitted. See the [public probe report](assets/you-yes-hand-shape-probes-2026-11-14.json) and [aggregate-only comparison plot](assets/you-live-landmark-review-2026-11-14.png).

## Interpretation and smallest next step

**Demonstrated:** live hand-shape estimates differ across the two views; the first includes strongly compressed projected finger joints; the unchanged model predicts YES on both, with all geometry enabled. The second visibly extended index still fails. This establishes a representation concern, not a universal detector error or an isolated geometry-head defect.

**Supported hypothesis:** orientation/foreshortening and the resulting hand-shape distribution contribute to poor transfer from public video to these webcam inputs. **Unresolved:** whether live 3D estimates are inaccurate, whether accurate estimates are outside the learned distribution, and how much thumb pose, body-relative placement or temporal context contributes. The four selected stills and model scores cannot separate those causes.

The smallest further experiment is a controlled front-versus-oblique view of the same straight-index posture, measuring tracking and model response with geometry enabled. Treat changed-view sequences as diagnostic manipulations, not automatically valid YOU accuracy examples. This would require fresh captures; no additional captures or model changes were made here. Retraining or broad representation changes are not yet justified by this review alone.

## Closure and integrity

The camera is closed. The four preview images were cleared from the local RAM receiver, the receiver shut down, and its token URL was removed from the metadata. No personal image or landmark file was added to project/training data. The four chat-visible images remain part of this conversation. [The portable result](assets/you-live-landmark-review-2026-11-14.json) stores predictions, aggregate measurements, review notes and user confirmation only.

All 76 protected source, cache, archive and checkpoint hashes match after review and the public probes. Geometry remained enabled in all live/preflight model calls. The reporting notebook contains the aggregate comparison without personal stills.

Final verification: 24 scoped tests passed (16 review/recognition tests and eight notebook tests). Notebook execution completed without errors, and all 31 generated artifact-manifest hashes match. Session-file inventory contains no PNG/JPEG/video/NumPy coordinate files. The model weights and trained preprocessing sources remain unchanged.

The earlier 19-clip YOU audit and eight-trial pose comparison remain separate preserved evidence. No training, cleaning, relabeling, checkpoint replacement, Git push or archive change is part of this review.
