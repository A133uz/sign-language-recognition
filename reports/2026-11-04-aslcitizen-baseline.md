# Optional ASL Citizen baseline — 2026-11-04

The first guided WLASL webcam session scored 0/3 each on `yes`, `no`, and `me`
across all four preprocessing variants. The user confirmed that capture was
normal and the WLASL reference gestures matched their signing. A body-relative
landmark probe also failed to justify a replacement. The webcam goal remains
incomplete.

This prompted a separate comparison with Microsoft's ASL Citizen I3D baseline.
[Microsoft's project page](https://www.microsoft.com/en-us/research/project/asl-citizen/)
describes recordings by everyday signers in everyday environments. This is a
reason to investigate a different training distribution, not proof of improved
webcam recognition. WLASL remains the main dataset and vocabulary target.

## Assets and label identity

Weights were downloaded from the [official GitHub release](https://github.com/microsoft/ASL-citizen-code/releases/tag/checkpoints_v1).
The archive contained one state dictionary; it loaded with `weights_only=True`
and matched the existing vendored I3D architecture strictly with 2,731 outputs.
The source class mapping is sorted unique stripped glosses from the official
training CSV. It is kept local under the dataset terms, rather than committed.

| Input | SHA-256 |
| --- | --- |
| Official model ZIP | `42c5349f95595fac79af77a902584510da92290e802903f3c96c4689a4b19a48` |
| Model state dictionary | `3538319620331d5ba731e0f9ac79161deeb37a895b7680d215864fc00a14477d` |
| Ordered class JSON | `cdc2b7f77b8d8797c19db24c1b7bf501618db5feb29def583c43b3da47c34910` |
| Source train CSV | `87a105649b98ea577182ebfce49d59fe9ca834a2489384f6fe7bc03f14966329` |
| Source `use.txt` | `5f712014b70142ef06d0bac3d47cb8ff210da0671ce555a6bdc3a7b3f3538d64` |
| Native pilot configuration | `68e3e020093b257fe755c724ea3cfc58a84e5078fe17f818534b7aa316d5e490` |

The native pilot contains HELLO, THANKYOU, BYE, PLEASE, SORRY, YES, NO, HELP,
FINE1, FINE2, GOOD and ME. Distinct source variants remain distinct. WLASL's
fingerspelled `i` is not silently reassigned or included. The backend requires
an `asl-citizen` vocabulary and rejects WLASL vocabulary files.

The official dataset archive is 45,924,134,223 bytes. Only split CSVs, its use
terms and ten selected clips were retrieved using verified HTTP byte ranges;
the full archive was not downloaded. A filename-filter mistake also retrieved
31 videos of the sign LICENSE; these unnecessary cached clips were removed,
and the filter was corrected to explicit document filenames. All data/model
assets remain ignored. The dataset's Microsoft Research License Terms restrict
use to non-commercial research and restrict redistribution; no source videos,
class metadata or weights are published as part of these changes.

## Preprocessing and runtime evidence

The implementation follows the [source dataset loader](https://github.com/microsoft/ASL-citizen-code/blob/main/I3D/aslcitizen_dataset.py)
and [testing procedure](https://github.com/microsoft/ASL-citizen-code/blob/main/I3D/aslcitizen_testing.py):
BGR input, source-dependent resize and center crop, centered temporal stride,
64-frame input, and maximum over linearly upsampled temporal logits. Short
sequences deterministically repeat the last frame. The source loader randomly
chooses first or last for padding, so this difference is explicit.

On all ten selected source test clips, the package's tensors matched the
reference-derived probe exactly and the full top-five label sequences matched
exactly on CUDA. This proves those tested preprocessing and inference paths,
not human webcam performance. Two source signers were selected per gloss in
CSV order; the sample is deliberately small and unrepresentative.

| Source test gloss | Full 2,731-class top-one | Native twelve-label top-one |
| --- | --- | --- |
| HELLO | 2/2 | 2/2 |
| THANKYOU | 0/2 | 0/2 |
| YES | 1/2 | 2/2 |
| NO | 1/2 | 2/2 |
| ME | 2/2 | 2/2 |
| Total | 6/10 | 8/10 |

Both THANKYOU examples produced GOOD first. On the ten existing WLASL
validation clips for these five glosses, full-class top-one was 3/10 and native
pilot top-one was 7/10. Those five gloss correspondences are used only for an
exploratory cross-dataset probe; no training examples or labels were merged.
These probes do not justify replacing the current default or claiming a better
recognizer. Generated details (ignored):

- `runs/aslcitizen_native_probe_inventory_20261104.json`
- `runs/aslcitizen_native_probe_predictions_20261104.json`
- `runs/aslcitizen_wlasl_probe_20261104.json`
- `runs/aslcitizen_production_parity_20261104.json`

The optional backend is exposed through `test-webcam-i3d --backend citizen`.
It uses the same guided prompts and predictions-only reporting. The camera
requests 640x480, and actual dimensions are recorded; native frames remain
only in memory with a 256-MiB bound. Timestamp resampling precedes the source
temporal stride. No webcam video, landmarks or embeddings are saved.

At this point, 76 tests passed with the locked environment, including both
backend lifecycle paths and validation of all three repository vocabularies.
Real CUDA inference was exercised on the ten source clips. A human webcam
session subsequently scored 5/15; see [the follow-up record](2026-11-04-framing-diagnostic.md).
Code is an uncommitted worktree based on
`0ecee354c5739d3e75c01c9985cf00eb70860b78`; dependency lock and hardware are
unchanged from the preceding experiment record.
