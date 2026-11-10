# First ST-GCN webcam test — 2026-11-11

The user requested a live test of the trained ST-GCN. The selected epoch-39
checkpoint was verified against its frozen test report:
`6617c495405197dd4c232e4a27e2a438fd698c8f9ba6b0d1edc1b795b5e86fad`.
This session uses the complete 2,731-label vocabulary on CUDA, with the saved
graph, preprocessing, label order and model source identities checked on load.

## Preflight finding and correction

1. Inspected the existing generic webcam entry point and checkpoint dispatch.
   Confirmed that the requested checkpoint constructs `SignSTGCN`, not GRU.
2. Compared inference for a 57-frame cached training sequence supplied as a
   physically short tensor versus the 128-step padded tensor used in training
   and official evaluation. First guesses matched, but the maximum absolute
   logit difference was **7.3125**. Matching only the first label was therefore
   insufficient to establish input equivalence.
3. Corrected `SignRecognizer.predict_observations` to reproduce the full
   **128×254 prepared input**, with zero right padding and the original true
   length. The graph adapter still produces a 2×128×27 tensor. Temporal
   convolutions' boundary behavior can differ when their physical tensor size
   changes; masking at block boundaries does not guarantee equivalence.
4. Kept model/graph/preprocessing files, trained parameters, frozen engine and
   dataset artifacts unchanged. This is a webcam input-contract correction,
   not retraining or a revision to the reported offline scores. GRU packed
   sequences already ignore temporal padding; this finding does not explain
   the previous GRU webcam errors.
5. Added a regression check using ST-GCN and the same fixed-grid input as cache
   evaluation. All **60 tests passed**. For the actual trained model, normalized
   raw observations and the matching prepared-cache input produced identical
   top-five labels and scores (**maximum score delta 0.0**). The checkpoint
   hash remained unchanged.

## Launch and user handoff

```powershell
.\.venv\Scripts\python.exe scripts/recognize_webcam.py --checkpoint models/asl-stgcn27-v1/best.pt --report .cache/webcam-stgcn-20261111-01.json
```

Launched with a hidden console and visible OpenCV window. Console confirmation:
`MODEL_READY | model=SignSTGCN | epoch=39 | classes=2731 | device=cuda`, followed
by `WINDOW_READY`. The session report confirms `architecture: stgcn27-v1`,
`camera_open: true` and `status: running` at launch.

Space starts a preparation countdown; perform one complete sign during SIGN
NOW, then Space ends capture. Q closes the window; R cancels/clears a capture.
Hand/pose processing happens after capture, so a result has a processing delay.
Only predictions, timing and tracking counts are saved. No video, landmark
coordinates, training or parameter updates are saved/performed.

## Reported feedback and unresolved log association

The session completed with **29 saved captures**, no recorded capture exceptions,
and unchanged checkpoint hash. The user reported:

- HELLO recognized on the second attempt.
- NO appeared in fifth place.
- THANKYOU and GOOD did not receive correct first guesses.

These reports are preserved verbatim in
[the feedback artifact](assets/stgcn-webcam-feedback-2026-11-11.json).
The captures do not store intended labels; no formal webcam accuracy is computed.

An independent check found a discrepancy between that feedback and both the
saved JSON and console output: **HELLO never appears in any saved top-five
list**, while **NO appears once, in first place (capture 6)**. GOOD appears in
fifth place in capture 25. THANKYOU never appears in a saved top-five list.
These are log occurrences, not evidence that those labels were the signs
performed in those captures.

Clarification was requested about the window title/session. Until resolved,
HELLO recognition and the reported NO rank remain **user-reported, not
log-corroborated**. Do not turn these observations into a confirmed improvement
over GRU, relabel the 29 captures, or infer an accuracy denominator. Median
hand-detection coverage across captures was about 91.4%; detection presence
does not establish correct landmarks or signs.

The initial commentary treated the feedback as a successful HELLO first guess;
the log check did not corroborate that interpretation. This record preserves
both the original report and the conflicting evidence rather than choosing
one without clarification. No retraining, checkpoint change or new webcam
session was started in response to this feedback.

The source/metric snapshots in the reporting notebooks describe the completed
offline experiments; this later live-input correction and feedback are recorded here.

## Confirmed misleading UI label and corrected rerun

The user clarified that the webcam interface displayed GRU and that they had
trusted the assistant's statement that the ST-GCN window was running. Source
inspection found the hardcoded processing message:
`PROCESSING hands + body, then GRU...`. This text was incorrect for ST-GCN.
The assistant acknowledged the mistake and apologized; the user should not
have needed to resolve inconsistent labels.

Session 01's launch output says `model=SignSTGCN`, with epoch 39 and the
verified ST-GCN checkpoint hash. The UI-label bug is confirmed; loading GRU
weights in that logged session is not supported by the available evidence.
The reported HELLO/rank observations still cannot be matched to its unlabeled
capture records. Correcting the model label does not resolve that separate
association question retroactively.

Corrections:

- Persistent header and window title derive the model name from the actual
  loaded class. The processing message uses that same model name.
- The standard `recognize_webcam.py` command now defaults to the trained
  ST-GCN epoch-39 checkpoint; GRU can be selected explicitly with `--checkpoint`.
- Session ID and numbered captures appear in the UI and saved records. They
  permit feedback to be matched to individual results without guessing labels.
- Report metadata includes actual model class/display name; camera-open state
  is cleared when the session closes. No weights or extraction settings changed.
- All 61 tests passed, including an ST-GCN status-message regression across
  ready/countdown/recording/processing states.

Reopened as `.cache/webcam-stgcn-20261111-02.json`. Launch output confirms
`MODEL_READY | model=SignSTGCN | epoch=39` and
`WINDOW_READY | ST-GCN webcam | epoch 39 | webcam-stgcn-20261111-02`.
The user was asked to confirm the persistent header and report capture numbers
with intended signs and first guesses.

## Session 02 user feedback

The user subsequently reported these incorrect first guesses:

| Performed sign, user-reported | First guess, user-reported | Matching logged capture IDs |
| --- | --- | --- |
| HELLO | GRANDFATHER | 2 |
| THANKYOU | GOATEE | 3, 10 |
| NO | DEVIL | 4 |
| ME | COMPLAIN | 5 |
| YES | KNOCK | 6, 13 |

All five prediction names are corroborated by the corrected ST-GCN session's
log. IDs 2–6 form a sequence matching the feedback order, but capture IDs were
not supplied by the user; duplicate first guesses prevent uniquely assigning
every reported sign to a saved capture. The raw records were not relabeled.
In particular, a correct label appearing elsewhere in an unlabelled record
is not automatically a successful recognition of a known intended sign.

Several matching failed-prediction captures had high detection coverage:
GRANDFATHER at capture 2 had hands/shoulder anchors in 53/53 frames; KNOCK at
capture 6 had both in 63/63. This rules out complete detector dropout for those
captures, not inaccurate geometry, wrong hand assignment, timing, viewing
angle or signer/style differences. This feedback does not resolve the exact
cause of the errors or demonstrate that the user performed a sign incorrectly.

Conclusion: the corrected ST-GCN recognizer remains unreliable for the five
reported signs despite better aggregate dataset metrics. No formal webcam
accuracy or comparative improvement over GRU is claimed from this session.
Further useful work is a targeted reference-video/live-landmark comparison
before choosing another architecture or training experiment. No new training,
weight changes or additional webcam session was started in response.

[Machine-readable feedback](assets/stgcn-webcam-session02-feedback-2026-11-11.json)
preserves the reported pairs, matching candidate predictions and limitations.
An immutable metadata-only snapshot was saved separately while the session was
still open, so later captures cannot alter the evidence attached to this report.
