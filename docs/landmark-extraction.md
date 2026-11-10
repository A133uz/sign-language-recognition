# ASL Citizen landmark extraction on Colab

The user authorized the extraction step on 2026-11-07 and confirmed that videos
are at `/content/asl_citizen/ASL_Citizen` and Google Drive is available. GRU
training remains on hold. The source code and a self-contained setup cell are
implemented; actual execution in the user's Colab runtime still needs checking.

## Continue the existing notebook

Keep the notebook/runtime containing the downloaded and unzipped dataset. Do not
delete that runtime or switch hardware before preserving needed files: the raw
videos live on its temporary `/content` disk.

1. Generate the delivery files locally with
   `.venv\Scripts\python.exe scripts/build_colab_bundle.py`.
2. Copy the entire generated `.cache/colab-delivery/asl_citizen_extraction_cell.py`
   into one new cell in the existing Colab notebook and run it. The generated
   `notebooks/asl_citizen_landmark_extraction.ipynb` contains the same cell for
   inspection or fresh runtimes; it must not replace the runtime holding the data.
3. Authorize Google Drive mounting when Colab asks. The cell installs extraction
   dependencies into one isolated runtime venv, so it does not replace packages
   already imported by the notebook. It embeds a checksummed snapshot of project
   source; there is no dependency on an unpushed GitHub branch or local Windows
   paths. No dataset, weights or private features are embedded in the notebook.
4. The cell automatically processes a ten-clip training-only pilot. It downloads
   Google's small pretrained hand and pose-lite task bundles if needed. This is
   landmark inference, not classifier training.
5. Inspect `pilot_result`, run `inspect_pilot()`, and review failures and tracking
   counts. The helper overlays saved landmarks on the source video frame when
   notebook OpenCV is available; `overlay_video=False` shows skeletons alone.
   The plot is an observation from a dataset clip, not a predicted sign label.
   Keep notebooks containing participant images private. Check more than one pose/frame before committing to a
   long extraction run; a finite array is not proof of accurate tracking.
6. When the pilot is acceptable, run the following in another cell:

```python
result = run_full_extraction()
print(json.dumps(result, indent=2))
```

The full call saves all official train/validation/test examples in resumable
batches. There is no training command in the delivered workflow. The full-run
cell in the generated notebook is commented out to avoid starting it with Run All
before reviewing the pilot.

## Environment and CPU behavior

### Diagnose a setup import failure

The first user Colab attempt stopped at the dependency import check, before
extraction. The follow-up diagnostic identified an inherited
`MPLBACKEND=module://matplotlib_inline.backend_inline`: MediaPipe imports
Matplotlib, but the isolated venv cannot resolve the notebook's inline backend.
The updated setup passes `MPLBACKEND=Agg` specifically to the dependency probe
and extraction subprocesses (including their spawned workers). It does not
change the notebook kernel's plotting backend or require another package.
It also captures combined stdout/stderr, prints each import in order, saves
`PROJECT / 'environment-check.log'`, and includes diagnostics in its exception.
These template-only changes retain the embedded source ID and existing runtime
environment/output paths.

Replace the old setup cell with the regenerated delivery cell and rerun it in
the same runtime. No runtime reset or dataset download is needed for this backend
error. The correction was checked locally with an unavailable inherited inline
backend; the user subsequently reported a completed Colab pilot and its rate.

If using an older copied setup cell, this compatibility patch in a preceding
cell makes its child processes inherit Agg too:

```python
import os
os.environ['MPLBACKEND'] = 'Agg'
```

This older-cell workaround changes the kernel's environment variable for future
imports/children; it does not switch an already-loaded pyplot backend. Prefer
the regenerated setup cell, which limits the override to extraction children.

For an already-failed cell, run this small diagnostic in a new cell in the same
runtime; `ENV_PYTHON` was defined before the failure:

```python
import subprocess
check = subprocess.run(
    [str(ENV_PYTHON), '-u', '-c',
     "import sys; print(sys.version, flush=True); import mediapipe, cv2, numpy"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
print(check.stdout or '(No diagnostic output.)')
print('Exit code:', check.returncode)
```

Share this diagnostic output to identify any further failure. This cell does
not install packages, run extraction or change dataset
files. Keep the current runtime; an import-check failure alone does not justify
resetting it or downloading the dataset again.

The setup targets Python 3.12/3.13 on Linux x86-64, using `requirements-colab.txt`:
MediaPipe 1.2.2, NumPy 2.2.6 and OpenCV-contrib 4.11.0.86. Compatible Linux wheel
availability was checked; this is not a claim that the environment has been
executed or GPU-tested on Colab. The Windows project venv stays separate and uses
its existing lockfile/NumPy 1.26.4. Core versions are recorded in each feature run.

The extractor explicitly uses CPU delegates. Assigning a T4 does not accelerate
these calls. One worker is the default; a second worker can be benchmarked using
`run_pilot(workers=2, clips=10)` before changing the full run to `workers=2`.
More processes can be slower on a CPU-limited runtime. Avoid extrapolating a
ten-clip compatibility check into a reliable full-corpus completion promise.
For a better timing sample use `run_pilot(clips=100)`.

The pilot prints a rough time and storage extrapolation from newly processed
clips, including per-clip detector initialization, decoding and shard commits.
Use it to check available Drive quota before a full run. A cached pilot does not
re-run inference and is not a new speed benchmark. The user reported a two-worker
Colab pilot rate of 4.95 seconds per newly processed clip (about 114.6 active
hours extrapolated across the corpus). This is a pilot observation, not a
measured full-run duration.

### Optional GPU compatibility diagnostic

Copy the entire `notebooks/colab_gpu_check_cell.py` (also delivered as
`.cache/colab-delivery/asl_citizen_gpu_check_cell.py`) into a new cell in the same
notebook after setup/pilot. It reuses `ENV_PYTHON`, `PROJECT`, `DATASET_ROOT` and
both existing task bundles. It installs nothing and changes no extraction
settings or feature archives. No training or full-corpus pass is performed.

The cell reports NVIDIA inventory, then requests `Delegate.GPU` for hand and
pose landmarkers in separate isolated processes. Each decodes up to eight frames
near the middle of one official training clip and invokes video inference.
It preserves native GL/EGL/delegate diagnostics, exit codes and timeouts in
`gpu_check_results`. It never explicitly retries on CPU. Each probe times out
after 120 seconds; a native crash does not prevent the other probe from running.

`GPU_REQUEST_PASS` means the requested task initialized and inference returned;
it does not independently prove that every operation ran on the NVIDIA GPU.
Inspect renderer/delegate logs for NVIDIA versus software-rendering/fallback
evidence. NVIDIA inventory alone, nonzero detections or a successful import is
not proof of T4 acceleration. Follow compatibility with a representative
CPU-versus-GPU throughput comparison before changing the full extractor. The
user ran the diagnostic on Colab with a Tesla T4 (driver 580.82.07,
MediaPipe 1.2.2). Both tasks failed during initialization: default EGL
initialization failed, the alternate display selected Mesa `llvmpipe`, and
MediaPipe rejected GPU emulation. Neither task reached video inference. This
establishes that GPU delegation is unusable in that tested configuration; it
does not establish that all Colab configurations or the physical T4 are
unsupported. NVIDIA EGL library/driver exposure and selection need diagnosis
before any targeted environment adjustment. Do not replace system GPU drivers
or restart the runtime solely on the basis of this probe.

The user's follow-up loader check found generic EGL/GLES libraries loadable,
`find_library('EGL_nvidia')` unresolved, and only `50_mesa.json` registered.
This supports missing/unregistered NVIDIA graphics-side libraries; a loader
lookup alone is not an exhaustive filesystem search.

An experimental runtime-local retry is provided in
`notebooks/colab_egl_retry_cell.py` (delivery copy:
`.cache/colab-delivery/asl_citizen_egl_retry_cell.py`). Run after the GPU check in
the same Colab runtime. It requires driver **580.82.07**, downloads NVIDIA's
matching official no-compat32 archive (about 310 MB), verifies its published
SHA-256, and uses `--extract-only` so the system installer never runs. Temporary
extraction requires up to a few GiB of free `/content` storage. Selected graphics
libraries and their SONAME aliases stay under `/content/asl-nvidia-egl-580.82.07`.
No CUDA/NVML replacement, system linker-cache edit, kernel-module installation,
runtime reset or extraction-cache modification is performed.

A private GLVND manifest and child-only `LD_LIBRARY_PATH` select these libraries
for a repeat of both model probes. The script does not change `os.environ` or
the main extractor's CPU delegate. The user subsequently ran this retry on
Colab successfully: both probes initialized EGL 1.5 and reported OpenGL ES 3.2
NVIDIA 580.82.07 with renderer `Tesla T4/PCIe/SSE2`. Each processed eight frames
with eight landmark detections. Hands took 2.787 seconds and pose 5.761 seconds,
including initialization; these separate tiny probes are not full-pipeline
throughput measurements and cannot establish a speedup over the CPU pilot.

The hand probe also logged a tensor multiple-write error about unsupported
writes and possible synchronization problems. Both probes logged the normalized
ROI projection warning. Inference returned successfully, but this is not proof
that landmark outputs are reliable: compare finite coordinates, masks and
tracking behavior across full clips on CPU/GPU before bulk extraction. The
tensor error is unresolved; do not suppress it or describe it as harmless.
The private NVIDIA EGL setup therefore passed the small compatibility test.
The subsequent ten-clip comparison below supplies preliminary full-clip timing
and agreement evidence; larger-scale stability and tracking accuracy remain
unverified. The main extractor still explicitly selects CPU.

### Compare full clips on CPU and GPU

After the successful EGL retry, copy the entire
`notebooks/colab_compare_delegates_cell.py` (delivery copy:
`.cache/colab-delivery/asl_citizen_compare_delegates_cell.py`) into a new cell in
the same notebook. It defaults to ten seeded, randomly selected official
training clips, processed in full with both hands and pose. Both delegates use
one worker and the same models, frames, timestamps and thresholds. CPU/GPU order
alternates between clips to reduce systematic ordering effects.

The GPU candidate is a private copy of the existing embedded source with only
the two explicit CPU delegates replaced by GPU; unexpected source layouts are
rejected. The main project source, full-run settings and existing feature cache
are not modified. Fresh trackers are created per clip in both modes, matching
the current extractor. Recorded extraction time includes initialization,
decoding, tracking and NPZ compression; it excludes interpreter startup and
Drive/shard writes. It is not directly comparable to the two-worker CPU pilot's
wall-clock rate. A later pipeline benchmark must include storage overhead.

Results, full native logs, temporary feature NPZs and a preview board are kept
only in a unique `/content/asl-delegate-compare-...` folder. No training or
full-dataset extraction occurs. Each clip/delegate has a five-minute timeout.
Failures remain explicit and are excluded from paired timing; review their
logs before interpreting speed. The summary reports CPU-time/GPU-time, so a
value greater than one means GPU was faster on the completed pairs.

The comparison rejects mismatched frame timing and nonfinite output. Hand slots
are matched per frame by minimum XY distance, avoiding false discrepancies from
detection-order swaps. Missing/extra hands are reported separately. Pose
coordinates are compared only at points valid in both runs. XY distances are
in normalized image coordinates and measure disagreement, not accuracy; masks
and detector counts are also not ground truth. Inspect the preview, especially
if GPU loses detections or diverges substantially. Preview samples are middle
frames from up to three clips with largest hand-coordinate disagreements;
they are a spot check, not an exhaustive tracking review. Keep participant
images private. The summary also counts clips whose native logs contain the
tensor write error, and records NVIDIA/software renderer evidence. A clean
exit and finite arrays do not establish that the tensor issue is harmless.

Share the printed summary and qualitative preview observations before choosing
a delegate for full extraction. A ten-clip comparison is preliminary; repeat
with more clips if the first run is usable.

User-reported Colab result on 2026-11-07: all ten pairs completed, covering
844 frames. An NVIDIA renderer appeared in all ten GPU clips with no software
renderer. CPU took 52.5284 seconds (5.25284 seconds/clip, 16.07 FPS); GPU took
22.2775 seconds (2.22775 seconds/clip, 37.89 FPS). The paired extraction speedup
was **2.3579x** (57.59% less time), excluding interpreter startup and storage
overheads described above. All ten individual GPU timings were faster.

Agreement: hand counts matched in 97.5118% of frames, pose presence in 100%,
and pose validity masks in 99.2676%. Matched-point mean XY differences were
0.0021765 for hands and 0.0087420 for pose, in normalized image coordinates.
The largest individual clip's p95 differences were 0.0232815 and 0.0460545;
these are not pooled p95 values and not recognition/tracking accuracy scores.
CPU reported hands in 417/844 frames and GPU in 420/844, with pose in all frames
for both. Missing-hand frames require context/visual review, not an automatic
claim of tracking failure or acceptable coverage.

All ten GPU clips logged the tensor write error; none of the CPU clips did.
No clip failed, but this recurring native error is unresolved. A preliminary
linear extrapolation of GPU extraction time is 51.61 active hours for 83,399
clips, excluding Drive/shard overheads and interruptions. This is not a full
run estimate validated on representative corpus-wide lengths/load, nor a
comparison against the earlier two-worker CPU pilot. Keep the main extractor
on CPU pending preview review, larger/repeated validation and investigation of
the GPU error; do not silently promote the candidate or suppress its logs.

Local validation: four metric tests cover swapped hand slots, absent detections,
normalized XY distances, invalid values and mismatched timestamps. The full
13-test project check passed. The worker decoded/extracted two real local clips
(133 frames); copied CPU arrays supplied the second side solely to validate
summary/preview plumbing, not as GPU evidence. The preview was rendered and
inspected, delivery-cell equality and syntax were checked, and the embedded
extractor source ID remained unchanged. Actual Colab CPU/GPU timing is the
user-reported result above; no full-corpus GPU pass has been performed.

Subsequent 100-clip user-run comparison (2026-11-07): all 100 pairs completed,
covering 9,000 frames. CPU took 533.3548 seconds (5.33355 seconds/clip,
16.87 FPS); GPU took 245.1075 seconds (2.45108 seconds/clip, 36.72 FPS),
giving a **2.1760x** paired speedup. GPU was faster on 95/100 individual clips
using the rounded printed timings. All GPU clips reported NVIDIA rendering;
none reported software rendering. The compute-only extrapolation is now 56.78
active hours for the corpus, excluding Drive/shard overhead and interruptions.

Hand-count agreement was 96.6111% (305 frames differed), pose presence agreement
100%, and pose-mask agreement 99.0125%. Mean matched XY differences were
0.0049273 for hands and 0.0095384 for pose. The largest individual clip's hand
p95 was **0.3961409**, substantially larger than the ten-clip sample's
0.0232815. Pose's largest clip p95 was 0.0591593. Means do not establish that
this hand outlier is acceptable. CPU found hands in 3,910/9,000 frames and GPU
in 3,922/9,000 (43.44% versus 43.58%); both reported pose in 8,998 frames.
All 100 GPU clips logged the tensor write error; no CPU clip did. The error
and extreme hand disagreement remain unresolved, with no automatic GPU
promotion or full-corpus run.

### Audit saved hand-disagreement outliers

Copy `notebooks/colab_audit_comparison_cell.py` (delivery copy:
`.cache/colab-delivery/asl_citizen_audit_comparison_cell.py`) into a new cell
after the completed comparison in the same runtime. It reads `comparison_dir`
and saved NPZs only; no MediaPipe inference, new extraction or download occurs.
It ranks five clips by their hand p95 difference, scans their actual frame-level
hand matches, and prints counts for missing/extra hands, model-reported
handedness disagreements, and high-distance frames. Triage thresholds of
0.05/0.10 are not quality acceptance limits. Three side-by-side previews show
the frames with highest per-frame matched-point p95, not generic middle frames.
The JSON/preview stay inside the existing runtime-local comparison folder.

The current distance metric matches available hands spatially without enforcing
reported handedness. If CPU detects only one physical hand and GPU only the
other, those observations can be compared despite representing different hands.
The resulting large metric could reflect detector/identity disagreement rather
than malformed coordinates; neither delegate is ground truth. The audit shows
the raw frame, skeletons and model-reported Left/Right labels so the user can
distinguish those cases. A handedness label mismatch alone is not proof of
opposite physical hands, since those labels can also be wrong. Preserve the
underlying data and logs instead of correcting or discarding outliers silently.

Audit validation: three new unit checks cover opposite reported hands, actual
worst-frame selection/missing groups, and detection-slot swaps. All 16 project
tests passed. The audit ran on the two saved CPU-only fixture pairs without
inference, wrote JSON and a rendered preview, and that preview was inspected.
These local checks validate audit plumbing; actual Colab outlier diagnosis
still depends on the user's saved 100-clip results.

Archive SHA-256:
`c8a3c1306a6abb1d9856d966b42dc8ebd0d5f76275a34e59650d9c14e57cc0c4`.
The official checksum and extraction-only/target options were checked on
2026-11-07. Sources:
[official archive index](https://download.nvidia.com/XFree86/Linux-x86_64/580.82.07/),
[extraction-only behavior](https://download.nvidia.com/XFree86/Linux-x86_64/580.82.07/README/installdriver.html),
[GLVND vendor selection](https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md).

Local validation exercised the same probe's decoding/options/inference with CPU
for hands and pose (eight frames and eight detections each). The unmodified GPU
cell was also run on Windows: both tasks reported GPU-disabled build errors,
and the cell preserved both tracebacks and completed its summary. Those Windows
results do not establish Linux/Colab GPU availability. Generated cell syntax
and delivery-copy equality were checked.

References: [BaseOptions](https://developers.google.com/edge/api/mediapipe/python/mp/tasks/BaseOptions),
[MediaPipe GPU requirements](https://developers.google.com/edge/mediapipe/framework/getting_started/gpu_support).

## Local CPU benchmark

The user chose to evaluate the local machine after Colab's session limits and
the unresolved GPU tensor error. The i5-9400F was benchmarked on 100 identical
training clips (9,257 frames), using the normal hands-plus-pose CPU extractor:
one worker averaged 3.143 seconds/clip, two 1.613, four 1.039. Four workers were
the fastest tested setting, with a rough full-corpus estimate of 24.1 active
hours including local shard-write overhead. All runs had zero failed clips and
no tensor-write errors; all 100 NPZs matched exactly across worker settings.

Use `scripts/benchmark_landmarks.py --clips 100 --workers 1 2 4` through the
project venv to repeat a fresh bounded comparison. Full measurements, timing
scope and limitations are in
[the local benchmark report](local-cpu-benchmark-2026-11-07.md). The benchmark
did not launch full extraction or training. Local and Colab runs have different
environment identities; do not merge or resume them in the same feature folder.

## What is saved

Full local extraction was subsequently authorized with **two workers** and
started on 2026-11-07 at 19:04 Asia/Tashkent. The manifest covers all official
train/val/test clips (83,399 total, 836 shards, no limit). The first 100-clip
batch committed successfully with zero failures. Output is separate from
benchmark and Colab folders. A hidden supervisor writes progress/native logs,
provides status/stop commands, and holds an execution-scoped Windows idle-sleep
prevention request. See [local extraction controls](local-extraction.md).
Full completion is pending; training remains on hold.

Default extraction keeps both hands and 33 body-pose landmarks, without a face
mesh. Body/head reference points retain information about where a hand is placed.
Every decoded frame is retained, with native-FPS timestamps and frame indices.
There is no augmentation, normalization, temporal resampling or padding yet.

| Array | Shape | Meaning |
| --- | --- | --- |
| `hands_xyz` | T × 2 × 21 × 3 | Raw normalized hand coordinates |
| `hand_present` | T × 2 | Whether a hand was reported in that detection slot |
| `handedness` | T × 2 | Model-reported Left=0, Right=1, missing=-1 |
| `handedness_score` | T × 2 | Handedness category score, not joint confidence |
| `pose_xyz` | T × 33 × 3 | Raw normalized pose coordinates |
| `pose_visibility`, `pose_presence` | T × 33 | Model confidence fields |
| `pose_detected` | T | Pose reported in that frame |
| `pose_valid_mask` | T × 33 | Visibility and presence both at least 0.5 |
| `timestamp_ms`, `frame_index` | T | Source timing and frame index |

Missing groups use zero coordinates and explicit masks. Low-confidence pose
coordinates remain in the raw archive but their validity mask is false. Hands
are saved in **detection order, not permanent anatomical left/right slots**.
The future training/inference preprocessor must canonicalize hand slots using
the saved handedness/body context. It must not flatten detection-order arrays
directly into a GRU and assume stable identity. Hand z is wrist-relative; pose z
uses a different origin. These are not a shared metric 3D coordinate system.

Per-clip JSON preserves filename, gloss, class ID, participant, official split,
decoded/declared frame counts, FPS, frame dimensions, source byte count and
aggregate tracking counts. No raw frames/video are copied into feature shards.
The input dataset's existing video bytes must remain unchanged within a run;
individual raw videos are not content-hashed by this extractor. CSV/model/code
fingerprints do not independently verify the raw corpus's content.

## Drive output and resume

The default output is a private folder under:

```text
MyDrive/sign-language-recognition/landmarks/raw-v1-hands-pose-<source-id>/
  pilot-w1-n10/
  full/
```

Each run contains `run-manifest.json`, `feature-schema.json`, `labels.json`,
`summary.json` and split-specific ZIP shards plus `.complete.json` markers.
The full run defaults to 100 clips per shard (836 shards for the official
split counts). Each ZIP contains per-clip compressed numeric NPZ arrays and
`records.json`. NPZ files contain no object/pickled arrays; read with
`numpy.load(..., allow_pickle=False)`.

Only a fully written, verified shard copy gets a completion marker. On resume,
archive hashes and record membership are checked; incomplete or corrupted shards
are recomputed. A stopped in-progress shard may therefore repeat up to 100 clips,
while previous completed batches are reused. Feature caches must not be silently
mixed across changed source code, models, CSVs, schema, core library versions,
split selection, limits or shard sizes. The manifest rejects such changes.

Run only one writer VM against an output folder. A local nonblocking lock also
prevents duplicate writers using the same local work directory, but it does not
coordinate separate VMs. Use the same setup cell and settings after reconnecting.
If a VM was deleted, raw videos must be available again before pending clips can
be processed. Stored features survive only if Drive writes completed; keeping the
notebook file alone does not preserve raw runtime data.

Use `run_full_extraction(max_shards=1)` to process at most one new full-sized
shard in a session, then the ordinary full call to continue. Changing worker count
does not change the schema and can resume the same cache. Changing pose settings
requires a separate cache. The source-ID folder separates generated code snapshots.

## Failures and quality review

A missing/unreadable clip, invalid FPS, incomplete decode, nonfinite coordinates
or an excessive frame count produces an explicit failure record, not fake zero
features. A zero-hand observation in an otherwise decoded clip remains a valid
raw observation with absence masks; it is not automatically a correct tracking
result. Review hand/pose coverage and per-class successful coverage before
training. Idle frames can legitimately have no hands; counts alone are not recall.

Exit code 2 / `complete_with_failures` means failed clips need attention. The
summary lists them; good feature batches remain saved. After investigating or
repairing inputs, `run_full_extraction(retry_failed=True)` recomputes whole shards
containing failures. Do not silently remove failures or change official splits.
Native MediaPipe diagnostics stay on the runtime at `PROJECT/native-extraction.log`.
Stopping a notebook extraction cell terminates its process group; completed Drive
batches remain reusable.

## Local validation and current limits

Nine project tests cover extraction observations/masks, incomplete decoding,
duplicate-writer locking, resume, checksum damage repair, failure reporting,
configuration separation and the existing dataset/camera behaviors. Two actual
training clips were processed with hands and body
pose: 220 frames, zero extraction failures. These are functional smoke checks,
not tracking-accuracy or recognition measurements. Full local corpus extraction
has now started with two workers; completion is pending. Colab pilot/comparison
results and local benchmarks are recorded above. The current step produces
raw features only. GRU implementation, canonicalization, training and inference
export still need their own work and training authorization.
