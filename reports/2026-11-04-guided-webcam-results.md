# Guided webcam results — 2026-11-04

The fifteen prompted trials completed on CUDA. The user confirmed they followed
the displayed prompts and capture appeared normal. No video, landmarks or
embeddings were saved. Prompt agreement is not independently reviewed accuracy.

Predictions were selected from the full twelve-label pilot configuration even
though only five signs were prompted. Three trials per sign were randomized with
seed 42; each capture followed a two-second countdown and lasted two seconds.

| Prompt | Current aspect/mean top-one agreement | Aspect/max | Square/mean | Square/max |
| --- | --- | --- | --- | --- |
| hello | 3/3 | 3/3 | 3/3 | 3/3 |
| thank you | 3/3 | 3/3 | 3/3 | 3/3 |
| yes | 0/3 | 0/3 | 0/3 | 0/3 |
| no | 0/3 | 0/3 | 0/3 | 0/3 |
| me | 0/3 | 0/3 | 0/3 | 0/3 |
| Total | 6/15 | 6/15 | 6/15 | 6/15 |

Top-three agreement was 8/15, 9/15, 8/15 and 8/15 respectively. A correct lower
guess is not counted as correct top-one agreement. No resize or pooling variant
fixed any of the failing signs in this session; no variant was selected as a fix.
`please`, `sorry`, `goodbye`, `help`, `fine`, `good` and the fingerspelled `i`
were not prompted in this session. The goal remains incomplete.

Captures contained 61–62 camera frames with spans of 2.000–2.032 seconds.
The combined comparison of all four variants took a median 434.3 ms per trial,
range 294.6–919.2 ms, including the first inference. This is comparison latency,
not single-model latency or continuous-recognition frame rate. Consistent capture
counts provide no evidence of dropped-frame timing as the explanation here;
they do not prove that framing, lighting or sign performance was ideal.

Generated report (ignored): `runs/webcam_guided_20261104_01.json`.
SHA-256: `6475818715dbe12d63aaf9752b351647dc0aec9336e963a2beace45c3b4b546a`.
Asset, vocabulary and lockfile hashes are embedded in that report and match
the [preceding experiment record](2026-11-04-webcam-diagnostics.md).

A subsequent dataset-only probe compared raw and body-relative hand/pose features,
using sequences or four temporal means plus standard deviation, with linear and
RBF SVMs. All scalers were fit on the existing 80 training clips, and only the
21 validation clips were scored. Test results were not evaluated in that probe.
The six variants scored 7–12/21; the best variant still scored 0/2 on `me` and
1/2 on `yes`. The normalization did not justify replacing the webcam model.
Generated details: `runs/relative_landmark_probe_20261104.json` (ignored).

Reference clips `64287` (`yes`), `38527` (`no`) and `35546` (`me`), which the
current I3D model labels correctly, were shown to the user to compare the dataset
versions with their signing. The user confirmed these reference hand shapes,
positions and motions match what they performed. This rules out an obvious
self-reported mismatch; it is not an independent sign-language assessment.
