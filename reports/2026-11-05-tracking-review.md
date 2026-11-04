# Tracking review and temporal coverage — 2026-11-05

The timing and geometry candidates did not resolve live ME/THANKYOU errors.
This diagnostic inspects the features visually before further model changes.
It does not train on personal webcam samples or alter recognition defaults.

## Public-clip and consented webcam inspection

The new in-memory preview overlays the 21 image hand points with colored finger
chains and numbered tips (thumb 4, index 8, middle 12, ring 16, little 20). It
also draws normalized estimated world-hand XY/XZ views. Mirroring is applied
before labels are drawn, keeping numbers readable. Body/crop panels are mirrored;
world views show the unmirrored model coordinates. These are model estimates,
not measured depth. [MediaPipe's result documentation](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
describes the separate image/world coordinate outputs.

Training and validation ME/THANKYOU examples were inspected using the archived
features paired with their exact sampled public frames. A saved public ME frame
then exercised both prompts in the real MediaPipe preview flow, rendered two
880x960 boards, and released the replay camera. Public diagnostic images are
under ignored `.cache`; they contain no user webcam images.

The user explicitly authorized two annotated live-preview boards being visible
in chat without being added to project files or training data. Three frames each
of ME and THANKYOU were delivered through a bounded temporary loopback receiver.
The index markers followed the visible pointing finger in ME. THANKYOU's finger
chains broadly followed the visible fingers through the sampled movement.
No obvious gross 2D finger swap was established in these six frames. This is
qualitative review, without expert landmark ground truth; it does not establish
3D accuracy or rule out tracking errors elsewhere in the capture.

The loopback server was closed, image buffers cleared, and the preview process
exited normally, releasing the camera. The authorized preview boards remain in
chat. No user pixels, landmark coordinates or embeddings were written to project
files or used for training.

## Measured temporal-coverage mismatch

The prepared training archive was compared with aggregate tracking counts from
the three landmark webcam sessions (eight captures per prompted label). Source
counts are the number of 32 sampled frames with at least one detected image
hand. Webcam counts use the same definition. Source signers/labels remain
unreviewed; this comparison cannot establish a causal explanation.

| Class | Training clip counts | Webcam counts |
| --- | --- | --- |
| ME | 5–17 / 32; median 9 | 31–32 / 32; median 32 |
| THANKYOU | 8–14 / 32; median 10.5 | 25–32 / 32; median 31.5 |
| HELLO | 6–17 / 32; median 11 | 20–31 / 32; median 26 |
| YES | 7–17 / 32; median 13 | 30–32 / 32; median 32 |
| NO | 9–18 / 32; median 11.5 | 22–32 / 32; median 28.5 |

The existing summary mixes zero-filled rest/missing-hand frames with detected
hand features. Different temporal coverage can therefore change its inputs even
for similar visible hand shapes. Detection presence does not certify sign
activity, and the difference occurs in both successful and failed classes.

## Bounded active-window probe: not selected

A candidate selected the interval from first to last frame containing a paired
valid image/world hand, retained one frame of context at each end, and resampled
that interval to 32 points. Three predefined candidates used the same training-
signer three-fold splits and train-fold-only scalers as prior probes:

| Candidate | Training-signer CV | Macro accuracy | THANKYOU | ME |
| --- | --- | --- | --- | --- |
| Existing summary + timing | 133/173 | 0.766 | 10/16 | 9/11 |
| Active-window summary | 119/173 | 0.692 | 4/16 | 7/11 |
| Active-window summary + timing | 121/173 | 0.706 | 6/16 | 8/11 |

The baseline remained selected. Neither trimming candidate was integrated,
trained for deployment, scored on validation/test or tested on the user. The
measured temporal mismatch remains a diagnostic finding, not a validated fix.
Report: `runs/active_window_probe_20261105.json` (ignored).

The full local checks passed all 104 tests after adding the preview; a subsequent
mirrored-label correction passed lint/format and the public-frame runtime replay.
No model weights or dependency setup were changed by the preview work. Webcam
recognition remains unreliable and the goal remains incomplete.
