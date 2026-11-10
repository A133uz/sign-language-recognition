# Colab data preparation and future GRU training

Updated 2026-11-07. The user plans to run the data-preparation cells in their own
chosen notebook. The user reports completing the first three preparation steps
and has shared output from the delivered extraction pilot and GPU comparison.
The data-only cells below do not start MediaPipe extraction or training. The
user has now authorized landmark extraction after completing the first three
steps and confirmed `/content/asl_citizen/ASL_Citizen` plus Google Drive access.
Continue with [the implemented extraction workflow](landmark-extraction.md).
Local GPU GRU training was subsequently authorized and implemented on
2026-11-09; see [local-training.md](local-training.md) for the current runnable
trainer and settings. The Colab training-notebook adaptation below remains
planning material; it is not a claim that a runnable training notebook exists.

## What runs where

| Component | Responsibility | Status |
| --- | --- | --- |
| OpenCV | Read video/webcam frames | Local webcam tested |
| MediaPipe Tasks | Detect hand joints and skeletons | Local CPU preview tested |
| Dataset preparation | Download/unzip and check official metadata | Completed locally |
| Landmark extraction batches | Save raw hand/body sequences and masks | Full local CPU run started with two workers; Colab GPU remains experimental |
| GRU | Classify a sequence into an ASL sign label | Not implemented or trained |
| Small Transformer | Possible later classifier comparison | Planned option |
| Local inference | Use trained classifier with the webcam | Planned |

Cloud training does not require cloud webcam inference. A Colab notebook can be
a thin launcher for shared Python modules, with the same model and preprocessing
code later used locally. Do not create a separate, incompatible model in notebook
cells.

The current project uv lock/environment constraints and optional torch indexes
target Windows AMD64. Do not assume `uv sync --locked` reproduces that environment
on Colab's Linux VM. A pinned, compatible Colab environment for extraction and
training is not reproduced by that Windows lockfile. The extractor's isolated
environment uses separate pins in `requirements-colab.txt`; the user has run
it on Colab. The separate GPU candidate showed a 2.18x paired extraction speedup
over one-worker CPU on 100 clips, while logging a tensor write error in all
GPU clips and an extreme hand-coordinate disagreement in one clip. See the
[comparison record](landmark-extraction.md#compare-full-clips-on-cpu-and-gpu)
before choosing a delegate. Full extraction still selects CPU. The data-only cells below use Python's
standard library and the runtime's `wget` command.

## 1. Open the intended notebook and check disk space

Run the cells in the notebook where the dataset will be used. Files are placed
in that notebook's connected VM, not inside the `.ipynb` file and not in the local
Windows project. Saving or sharing a notebook does not preserve/share its VM's
custom files. [Colab FAQ](https://research.google.com/colaboratory/faq.html)

The locally verified version-1 archive is 45,924,134,223 bytes (42.77 GiB).
Extracted entries total 49,604,368,459 bytes. Together that is about 95.5 GB
decimal, before working space. The cell below reserves an additional 5 GiB and
accounts for an existing partial download in the same runtime.

```python
import shutil
from pathlib import Path

ARCHIVE = Path('/content/ASL_Citizen.zip')
ARCHIVE_BYTES = 45_924_134_223
EXTRACTED_BYTES = 49_604_368_459

already_downloaded = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0
required = max(0, ARCHIVE_BYTES - already_downloaded) + EXTRACTED_BYTES + 5 * 1024**3
free = shutil.disk_usage('/content').free
print(f'Available: {free / 1024**3:.1f} GiB')
print(f'Needed with headroom: {required / 1024**3:.1f} GiB')
if free < required:
    raise RuntimeError('Insufficient disk space for the whole archive and extracted dataset.')
```

If the runtime does not have enough space, stop before downloading the archive.
A chunked preparation workflow or a runtime with more disk would be needed.
The repository does not yet implement a Colab pipeline that extracts/deletes
video batches to fit a smaller disk. If a complete extracted dataset is already
present, verify it rather than rerun this fresh-setup sequence.

## 2. Download directly from Microsoft

This is the URL published in the [official project instructions](https://www.microsoft.com/en-us/research/project/asl-citizen/).
It avoids transferring the archive from the laptop. This downloads raw data,
not a trained recognition model.

```python
import subprocess

url = (
    'https://download.microsoft.com/download/b/8/8/'
    'b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip'
)
subprocess.run(['wget', '-c', '-O', str(ARCHIVE), url], check=True)
```

`-c` resumes an interrupted download only while the partial file is still
available. It cannot recover files deleted with an old VM. The known local
archive fingerprint is
`349c0e33f13b876f19ad4dee72205f4d2eae02937613e275200aa92283268eab`;
it is a project-calculated SHA-256, not a separately published Microsoft hash.
If Microsoft changes the archive, review the new version and metadata before
mixing it with existing partial downloads or features.

## 3. Extract the archive

This rechecks actual required extraction space and validates each member's CRC
while reading it. It is ZIP extraction, not landmark extraction.

```python
from zipfile import ZipFile

with ZipFile(ARCHIVE) as archive:
    needed = sum(item.file_size for item in archive.infolist())
    if shutil.disk_usage('/content').free < needed + 5 * 1024**3:
        raise RuntimeError('Insufficient space to extract the dataset.')
    archive.extractall('/content/asl_citizen')

DATASET_ROOT = Path('/content/asl_citizen/ASL_Citizen')
print('Videos:', DATASET_ROOT / 'videos')
print('Splits:', list((DATASET_ROOT / 'splits').glob('*.csv')))
```

CRC checks detect ZIP corruption; they do not prove that every video decodes or
contains usable hand tracking. Keep dataset materials private and follow the
[dataset license](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/).

## 4. Check official split counts

```python
import csv

expected = {'train': 40154, 'val': 10304, 'test': 32941}
participants = {}
vocabularies = {}
filenames = set()

for split, expected_count in expected.items():
    with (DATASET_ROOT / 'splits' / f'{split}.csv').open(
        newline='', encoding='utf-8-sig'
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == expected_count, f'Unexpected {split} row count'
    participants[split] = {row['Participant ID'].strip() for row in rows}
    vocabularies[split] = {row['Gloss'].strip() for row in rows}
    for row in rows:
        name = row['Video file'].strip()
        assert name not in filenames, f'Duplicate video: {name}'
        filenames.add(name)
    print(split, len(rows), 'videos;', len(vocabularies[split]), 'labels')

assert len(filenames) == 83399
assert len(set.union(*participants.values())) == 52
assert not (participants['train'] & participants['val'])
assert not (participants['train'] & participants['test'])
assert not (participants['val'] & participants['test'])
assert all(len(vocabularies[split]) == 2731 for split in expected)
assert vocabularies['train'] == vocabularies['val'] == vocabularies['test']
print('Official metadata counts, vocabulary and signer separation passed.')
```

This cell checks CSV metadata, not all video-file presence or decoding. The
local project's completed CRC/file-presence audit is recorded separately in
`docs/project-decisions.md`; it is not evidence that your Colab copy has run or
passed the same checks. Preserve the official splits instead of randomly
reassigning videos from the same signers.

## 5. Future extraction and training stages

Landmark extraction is now authorized and implemented using the linked guide.
The raw schema retains all hand/body observations, detection-order hand slots,
reported handedness and explicit masks. Normalization and stable hand-slot
canonicalization belong to the future trainer/inference preprocessor. No GRU
training command or runnable training notebook has been implemented yet.

1. Use and review the versioned raw hand/body schema, including detection order,
   handedness and missing-point masks. Finalize normalization, stable hand slots
   and temporal preparation for the future trainer; raw coordinates preserve
   flexibility for those steps. Current live preview tracks hands only; the
   offline extractor also saves body pose, without a face mesh.
   The agreed sequence limit is **128 frames per sign**: uniformly downsample
   longer signs across their full duration; keep shorter signs' true lengths
   and ignore right-padding through packed GRU processing. Temporal padding
   and missing-landmark masks serve different purposes. No extraction rerun
   is needed; this preprocessing still needs implementation.
2. Benchmark a representative 100–300-clip sample in the allocated runtime.
   Record frames, elapsed time, tracking coverage and the actual CPU/GPU delegate.
   A T4 selection alone does not use the GPU: the current hand tracker does not
   request a GPU delegate. A compatible backend must be configured and tested.
   [MediaPipe base options](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/core/base_options.py)
3. Extract resumable batches. Record video IDs, split/signer membership, labels,
   feature schema/version, extractor model hash, settings and failures. Save
   completed feature batches to private persistent storage before VM loss.
4. Load cached landmark batches from the VM's local disk for training. Avoid
   repeated reads of tens of thousands of small files through a Drive mount.
   The [Colab FAQ](https://research.google.com/colaboratory/faq.html) recommends
   reducing Drive I/O and copying archive data onto the VM for active work.
5. Train the GRU on the 40,154 training samples, select settings/checkpoints on
   the 10,304 validation samples, then evaluate the final chosen model on the
   32,941 held-out test samples. Do not tune augmentation or model choices on test.
6. Save resumable training checkpoints periodically to persistent storage:
   model/optimizer/scheduler state, epoch/step, random states and configurations.
   Free Colab sessions can end early and run for at most 12 hours; resource and
   GPU availability vary. Do not depend on one uninterrupted multi-day session.
   [Colab resource limits](https://research.google.com/colaboratory/faq.html#resource-limits)

## Runtime budgets: estimates, not benchmarks

| Stage | Discussion estimate | Assumptions and limits |
| --- | --- | --- |
| Whole-corpus hands-only extraction | Roughly 1–3 days initially | Unmeasured on Colab; can take longer with a slower CPU or extra landmarks |
| Small GRU on a T4, 50–100 epochs | Roughly 1–6 hours | Cached features, two layers, 128–256 hidden units, about 64 frames, batch 64–128; unmeasured |

The 2026-11-09 sequence decision uses a **128-frame limit**. The historical
64-frame estimate above is not a measured budget for the selected configuration;
benchmark real training and validation epochs before estimating total time.

The local hand detector averaged 17.56 ms/frame, excluding decode/save overhead.
Assuming 90 frames per clip would produce about 7.5 million frames and roughly
37 hours of detector work alone at that local rate. The 90-frame average is an
illustration, not a measured corpus statistic. A T4 does not change the estimate
for an extractor that still runs on the CPU. Body/face tracking adds work.

For a measured training-plus-validation epoch of three minutes, 50 epochs would
take about 2.5 hours and 100 about five hours. Replace this illustration with a
forecast from several measured epochs after authorization; include I/O, saving,
setup and final evaluation time. There are currently no Colab timings or training
accuracy results for this project.

## Training-only augmentation plan

Augmentation is recommended for evaluation, not a prerequisite to run a model.
The training set averages about 15 examples per class. Compare a baseline against
conservative augmentation, using validation results to decide whether it helps.

- Apply small geometric transforms consistently across joints and frames, so
  they preserve hand shape, relative positions and the sign's motion.
- Vary temporal sampling/speed modestly while retaining the complete sign.
- Consider bounded landmark noise and limited masking with an explicit missing
  data representation. Do not turn missing zeros into apparent valid positions.
- Validate horizontal flipping and corresponding left/right slot changes before
  use. Avoid time reversal, aggressive finger distortion and crops that remove
  defining parts of the gesture.

Apply augmentation on the fly to cached training features. Do not rerun MediaPipe
on every epoch or store many duplicate copies. Keep validation and test
preprocessing deterministic, with no augmented examples leaking between splits.

## Return the model to the local webcam application

Export one compatible bundle containing:

- Trained model parameters, architecture configuration and implementation version.
- Exact class-ID/gloss mapping generated from the training vocabulary.
- Landmark schema, hand ordering, normalization, missing-value handling, sequence
  length and sampling settings.
- Extractor/library versions and landmark model hash.
- Evaluation results and dataset/split fingerprints for reproducibility.

Local inference must recreate the same architecture and input preparation, load
the trained weights and use the exact mapping. Optimizer state is needed to
resume training, not to classify webcam captures. The local application must not
depend on a live Colab session. Current model/code compatibility and a transfer
round-trip test still need implementation; none has been claimed as validated.
