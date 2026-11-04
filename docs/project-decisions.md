# Project decisions

Agreed on 2026-10-25:

- ASL first; RSL remains a secondary comparison accepted by the supervisor.
- WLASL is the initial isolated-sign dataset.
- Originally planned to start with 10 everyday signs and expand to 25.
- Desired expressions include hello, thank you, goodbye, you're welcome, and
  see you. Dataset labels and ASL expressions must be checked individually.
- The provisional config uses hello, thank you, goodbye, please, sorry, yes,
  no, help, fine, and good. Review usable clips before freezing this list.
- Avoid separate bye/goodbye classes until overlap has been reviewed. Do not
  equate WLASL's welcome label with you're welcome without checking the sign.
- Local hardware is an RTX 4060 with 8 GB VRAM.
- Ordinary Python files form the main pipeline. Jupyter is optional.
- Prioritize implementation and the university report; publication is not required.
- Use a small landmark-sequence model first; continuous translation is later work.

Repository setup uses uv with a committed lockfile rather than parallel manual
requirements files. Development, full CPU, and full CUDA profiles share one
package and configuration. Uv is installed once as a standalone user tool; the
project uses a single `.venv`, including pip for familiar package inspection.
The local WLASL metadata can be inspected by gloss. Videos and trained models
are local experiment artifacts and are not committed to Git.

As of 2026-11-01, the intended vocabulary is all WLASL glosses, with WLASL2000
as the full-dataset target. The current 10-sign draft remains a pipeline pilot;
it is not a final vocabulary limit. Build label handling and manifests to scale
from the pilot to all available classes. Audit clip availability and signer
coverage, then measure per-class performance and report missing or weak classes.
Full coverage and reliable recognition are goals to validate, not guarantees.

The full local metadata has been audited for counts and recorded signer-ID overlap.
A bounded direct-MP4 probe decoded the first frame of two `hello` examples; it
does not establish full-video or dataset-wide availability. The WLASL paper says
signer IDs were derived from face matching with manual checks, so group by ID but
keep the possibility of identification errors in mind. Next, investigate signer
coverage for the pilot. A seeded, metadata-only pilot manifest now keeps signer
IDs and exact shared media separate and reports class coverage. It is provisional
until actual usable clips and signer-ID reliability are checked. Next, build a
verified clip inventory and regenerate the split from usable data before training.

As of 2026-11-01, hosted CI is not needed for now. Run the documented checks locally.

The local video inventory records missing files, decoding results and checksums;
it does not certify labels, frame intervals, or complete decoding. Inventory-based
splitting remains planned after the handling of clip intervals is established.

On 2026-11-04, an experimental isolated-sign pipeline was added: it extracts hand
landmarks from local, already prepared clips, groups examples by signer and exact
media identity, trains a GRU, and offers a space-triggered webcam capture. The
baseline uses hands only; pose and facial features are not yet included. This
is an experimental milestone, not proof of reliable live recognition. The earlier
inventory is not consumed by the preparation command, which performs its own
decode and hand-presence checks. Clip labels and intervals still require review.

User webcam feedback exposed incorrect predictions for `hello`, `thank you` and
an untrained sign. The project now also supports pose-based context, an experimental
linear SVM comparison and an uncertainty threshold. These are not validated fixes:
the threshold cannot reliably reject unknown signs and held-out `hello` results
were poor. The user prefers trying available dataset clips before collecting their
own labeled webcam examples. The full WLASL vocabulary remains the long-term goal.

A pretrained WLASL2000 I3D video model is available as an additional experiment.
Its unrestricted mode failed an initial live test; its optional ten-label pilot
mode returned `hello`, `thank you`, `please`, and `sorry` correctly on one user
attempt each; `yes` and `no` were not first guesses. The
pilot mode is closed-set and must not be presented as handling all signs or
rejecting other signs. Further live and signer-disjoint evaluation is needed.

The user clarified that their earlier `I` test meant the pronoun gesture toward
the signer, rather than the fingerspelled letter. Sampled WLASL examples label
these separately as `me` and `i`; an optional twelve-label vocabulary retains
both exact labels without merging them. Recognition of the pronoun remains to
be improved: on one user webcam attempt, the twelve-label model returned `hello`
first and `me` third.

A preprocessing audit found that five locally extracted source videos were
29.97 FPS while WLASL's annotations refer to 25 FPS. The interval extractor now
resamples by annotation timestamps. The five clips have been regenerated with
their previous versions archived; feature archives and local training results
created before that correction are provisional and require regeneration.

The corrected-clip landmark archive has now been regenerated from all 132 local
clips and its video hashes verified. A fresh twelve-label SVM still scores only
11/21 on validation and is not selected for webcam use. The original archives
are retained as historical artifacts.

A comparison of I3D resizing, temporal sampling and pooling on 21 validation
clips favored the existing default (20/21 restricted-label top-one). The reference
test path scored 7/21; changing preprocessing is not a demonstrated live fix.
The project now includes a guided, randomized webcam diagnostic that saves
predictions and timing, without saving pixels or features. It compares variants
on the same capture and reports prompt agreement, with uncompleted trials visible.
Its simulated lifecycle and saved-clip CUDA checks passed before the first human
session. The webcam recognition goal remains incomplete,
and the full WLASL vocabulary remains the longer-term scope.

The first guided human session completed all 15 prompted trials. The user
confirmed following the prompts without capture problems. Each of the four
resize/pooling variants returned `hello` and `thank you` first on 3/3 trials each,
but `yes`, `no` and `me` on 0/3 each. Captures contained 61–62 frames in roughly
two seconds. This does not establish reliable webcam recognition; no comparison
variant is selected as a fix. A subsequent body-relative landmark probe also
failed to improve validation sufficiently and was not deployed. See
[the live-test record](../reports/2026-11-04-guided-webcam-results.md).

The user also confirmed that the displayed WLASL reference clips match their
performed `yes`, `no` and `me` gestures. A separate ASL Citizen pretrained I3D
baseline is now available for comparison because its source data includes
everyday recording settings. This is an additional experiment, not a replacement
for WLASL or the full WLASL vocabulary goal. It uses native class labels and
source-derived preprocessing with explicit deterministic padding. No datasets
were merged and no WLASL label aliases were assumed. Ten source-test clip CUDA
checks passed tensor/prediction parity, but both source `THANKYOU` clips were
misclassified as `GOOD`. Its subsequent guided human session scored 5/15 top-one:
HELLO 2/3, THANKYOU 1/3, YES 1/3, NO 0/3 and ME 1/3. It did not improve this
session and remains an optional experiment.

A four-prompt WLASL crop diagnostic found detected hands inside the model crop,
with hand sizes comparable to the training reference. The user's nose appeared
around model y=91–94 versus a training median 56.6, and shoulder span around
121–124 versus 103 pixels. This is a measured framing difference, not proof of
the recognition-error cause. An optional body-alignment comparison now uses one
fixed per-capture scale/translation from median nose/shoulders to a reference fit
on 80 training clips only. On 21 exploratory validation clips it scored 21/21,
versus 20/21 unaligned. Source exposure and repeated validation reuse limit these
numbers. The full guided path passed a saved-clip replay using real CUDA and
MediaPipe. The same-capture human comparison completed all four prompts and
matched only hello, both before and after alignment. Yes, no and me remained
incorrect. Alignment is not selected as a fix; no default is changed.
See [the framing record](../reports/2026-11-04-framing-diagnostic.md).

For the next hand-shape modeling experiment, a bounded native ASL Citizen pilot
has been obtained: 177 training, 46 validation and 146 test clips for twelve
distinct native labels. Its selected signer IDs are disjoint across the three
source splits (28/5/11 signers). All archive members passed CRC verification;
full decoding also passed, with one 664-frame training YES clip exceeding the
current 600-frame preparation limit. It needs explicit handling or exclusion,
not silent truncation. This data acquisition is not a trained model or proof of
webcam improvement. Native feature preparation and the new hand-shape training
experiment were the next step. No WLASL label mapping is assumed.

Native preparation is now implemented with estimated world-hand shape centered
at each wrist and normalized by wrist-to-middle-MCP length. It preserves image-x
hand pairing and appends valid-world-hand masks to the existing hand/pose data
(288 values per frame). Original source splits are retained; signer-ID overlap
and identical file bytes across splits are rejected. The prepared set contains
173/46/146 train/validation/test clips and 27/5/11 signer IDs. Three training
clips failed hand detection and one exceeded 600 frames; all twelve labels
remain represented in every split.

Four candidates were compared with test scoring disabled. A compact four-bin
hand-shape summary with a linear SVM was selected by validation macro accuracy:
40/46 validation, versus 34/46 for summary RBF, 32/46 for raw linear and 27/46
for the GRU. The selected checkpoint was then evaluated once on the 146 held-out
clips and scored 104/146 (YES 9/10, NO 10/11, ME 11/13). Weak classes remain;
FINE2 scored 2/8 and SORRY 6/12. No test outcome was used to select among those
candidates, and the selected checkpoint remains frozen.

The new landmark guided backend captures frames first, then extracts the same
32 full-frame samples as preparation. It reproduced features exactly on three
saved yes/no/me clips and passed a full saved-clip guided replay. It rejects
captures with insufficient tracked hands but cannot reject arbitrary untrained
signs. The first human session completed 15 prompts: YES and NO matched 3/3
each, HELLO 2/3, THANKYOU 1/3 and ME 0/3 (9/15 overall). This is real progress
on yes/no but not reliable recognition across the vocabulary. No failed class
is removed from the task. This does not replace WLASL's
longer-term vocabulary target or establish reliable webcam recognition.

A subsequent training/validation-only probe combined frozen WLASL I3D video
descriptors with the compact native ASL Citizen hand summary. Video-only linear
classification scored 28/46 validation, linear fusion 35/46 and RBF fusion
36/46, below the existing hand model's 40/46. None was promoted to webcam use;
no additional test scoring was performed. Reuse of this small validation set
limits further model-selection conclusions. See the world-hand experiment report.

On 2026-11-05, the user authorized reviewing a few ME and THANKYOU preview
frames in chat, without adding them to training data. Direct desktop capture
failed, so an explicit preview-only module sends at most two three-frame boards
to a temporary loopback receiver in memory. It does not save webcam images or
landmarks. The ME preview showed the hand clearly, but its depth was ambiguous;
the user confirmed that the index fingertip pointed into the chest. ME remains
an unresolved recognition failure, rather than a confirmed signing error.
The THANKYOU preview showed fingertips near the chin followed by outward motion,
consistent with the basic ASL University description. Three sampled frames do
not establish expert sign correctness, classifier behavior or tracking accuracy;
this preview did not run a recognizer. No obvious signing mismatch was established
as the cause of either error. The local receiver was closed and its image buffers
cleared after review; authorized images remain visible in chat. The preview
process also exited normally and released the camera.

A timing augmentation experiment selected an optional candidate using three-fold
training-signer cross-validation (133/173 versus baseline 117/173; ME 9/11 versus
5/11). The existing validation score decreased to 39/46 from 40/46, so this is not
a validated general improvement. The candidate preserves the same representation
and native labels, uses no personal training data, and has no new test score.
An optional same-capture comparison now shares one landmark extraction between
compatible checkpoints; the original remains frozen. All 15 human trials
completed with user-confirmed normal signing/capture. The timing candidate
matched 9/15 versus 7/15 for the original: HELLO 3/3 versus 2/3 and ME 1/3 versus
0/3, but THANKYOU 0/3 for both. YES remained 3/3 and NO 2/3. This is insufficient
for reliable recognition; the candidate stays optional and no default changes.
Details are in [the timing record](../reports/2026-11-05-timing-augmentation.md).

A subsequent optional `body_geometry` representation retains every hand point
relative to nose/shoulders and rotation-invariant finger-shape measurements.
Training-signer CV selected geometry plus timing (140/173, ME 11/11), but the
existing validation result remained 39/46, including THANKYOU 1/3 and BYE 1/3.
No new test scoring was done and no default changed. Runtime parity/replay and
104 local tests passed. The shorter human comparison completed ten trials with
user-confirmed normal capture: geometry matched 3/10 versus original 6/10,
including YES/NO/THANKYOU regressions. It is rejected as a replacement; source
cross-validation did not establish live improvement. See
[the geometry record](../reports/2026-11-05-body-geometry.md).

A consented annotated preview found no obvious gross 2D finger swap in three
ME and three THANKYOU frames; it did not establish accurate depth or rule out
other tracking failures. No personal frames/features were retained in project
files or training data. A comparison of saved aggregate counts found much more
continuous hand presence in webcam captures (ME median 32/32) than source
training clips (ME median 9/32). Active-window trimming reduced training-signer
CV performance and was not selected or tested live. See
[the tracking record](../reports/2026-11-05-tracking-review.md).

A newer SignRep RGB representation was checked on a public clip with real CUDA
inference and strict checkpoint loading. Its weights/source remain local and
ignored. The user cleared the development goal during training/validation feature
extraction; the process was stopped with a partial cache and no final feature
archive. No SignRep label classifier, recognition score or webcam functionality
has been implemented. No dependency or default-model change was made. See
[the stopped probe record](../reports/2026-11-05-signrep-probe.md).
