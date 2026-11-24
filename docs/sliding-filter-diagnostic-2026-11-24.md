# Guided sliding-window filter diagnostic — 24 November 2026

## Objective and conditions

Compare provisional filters on identical windows from the retained epoch-36
15-label face-aware GRU. This is an exploratory diagnostic, not a calibrated
rejection system or a model update. All 15 labels competed. Each trial lasted
about seven seconds; scoring used the most recent two seconds, starting after
the initial buffer filled and then approximately every 0.5 seconds. The 0.60
threshold and requirement for two consecutive matching eligible windows were
specified before the session, but not chosen by calibration. Scores are softmax
outputs and should not be interpreted as calibrated probabilities.

Three decisions were calculated on every same scored window:

- Raw: the highest-scored label.
- Threshold: emit the top label only at score >= 0.60.
- Threshold + stability: require two consecutive threshold-passing windows
  with the same label; reset after a gap or an unscored window.

The user confirmed all nine prompts were followed with normal capture. The
prompts were REST, unrelated MOVE, HELLO, THANKYOU, PLEASE, YES, NO, HOLD
(hold the HELLO end pose), REST. The HOLD prompt is not a complete sign and is
reported separately, without assigning it a ground-truth gloss. For sign prompts,
the instruction was to sign once in the first three seconds and then relax.
No sign boundary timestamps were recorded, so a matching label somewhere during
the seven-second trial is not proof of correct sign localization.

The immutable local report `.cache/sliding-filter-guided-20261124-v1.json`
contains predictions, scores, timing and frame counts, not images, raw landmarks
or embeddings. Its SHA-256 is
`e8454dd87addc0afd6385a5bc4890c6998f18583e88f2914ceb4cdf7e07b7884`.
The separate local confirmation record attaches the user's confirmation to
this hash. Neither local artifact is tracked by Git.

## Observations

All nine trials produced ten scored windows each, with no unscored windows.
These heavily overlapping windows are not 90 independent sign examples.

| Condition | Raw windows emitted | Threshold windows emitted | Threshold + stability windows emitted |
|---|---:|---:|---:|
| REST #1 and #9 combined | 20/20 | 7/20 | 5/20 |
| Unrelated MOVE #2 | 10/10 | 0/10 | 0/10 |
| Held HELLO end pose #8 | 10/10 | 0/10 | 0/10 |

Each REST trial still had a run of threshold-and-stability false outputs: MY
in #1 and YOUR in #9. In REST #1, MY reached 0.89; thus confident, repeated
predictions did not reliably reject rest. The held-pose observation alone does
not establish whether an actual static sign would be recognized.

| Prompt | Raw matched prompt in at least one window | Threshold matched | Threshold + stability matched |
|---|---|---|---|
| HELLO | Yes | Yes | No |
| THANKYOU | Yes | No | No |
| PLEASE | Yes | Yes | Yes |
| YES | Yes | Yes | Yes |
| NO | Yes | Yes | Yes |

The only raw THANKYOU match was in a window ending around 5.7 seconds, while
the instruction was to perform the sign in the first three seconds and then
relax. Without verified boundaries, that match cannot establish detection of
the intended sign. Threshold + stability missed HELLO and THANKYOU while still
accepting false labels during both REST trials. In the THANKYOU trial it also
accepted OUR. Thus this candidate should **not** be enabled by default.

Counts measure decisions on sampled, overlapping windows, not independent
accuracy, false events per minute, full-time display durations or sign-onset
latency. One signer and one trial per sign are insufficient to generalize.
The diagnostic scores synchronously in its capture loop; any processing gaps
and timing differences limit comparisons with the asynchronous prototype.

## Follow-up

Preserve the unchanged raw prototype and checkpoint. Collect a separate,
independent validation set with repeated sign attempts, idle poses, unrelated
movements and unknown signs, with confirmed timing or a controlled protocol.
Use it to compare sign activity detection or rejection approaches while
measuring missed signs, false outputs and delay. Test held complete signs as
well as static poses; the held-pose control here is not a substitute. Do not
tune a threshold on this one session and call the result validated. No model
training or model replacement occurred.
