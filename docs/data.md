# Dataset and experiment records

WLASL's official source is https://github.com/dxli94/WLASL. Read its Computational
Use of Data Agreement before obtaining or using the data. The repository states
academic/computational use and excludes commercial use. Missing URLs may require
the maintainers' video-request process. This project does not bypass that process
or redistribute dataset clips.

Keep raw clips under `data/raw/` and derived landmarks under `data/processed/`.
Both are ignored by Git. Do not assume the published sample count equals the
number of downloadable, successfully processed examples.

Track the following small records under `data/manifests/` once preparation exists:

- Dataset release, source URL, retrieval date, and metadata checksum.
- Video ID, exact source gloss, signer ID, source split, and assigned split.
- Relative local path, clip frame bounds/FPS, download status, and extraction status.
- Exclusions with reasons, per-class counts, and signer coverage.
- Split-generation seed/method and a signer-overlap audit across all splits.

Do not silently merge similar labels or replace dataset labels with English
phrases. Keep any reviewed mapping explicit and versioned. Label order in
`configs/vocabulary.yaml` will define class IDs; save it with every trained model.
PyYAML treats unquoted yes/no as booleans, so those labels are quoted in the config.

Before training, establish a feature schema shared by offline extraction and the
webcam app: landmark ordering, coordinate space, normalization, handedness/mirror
handling, missing-hand masks, sequence sampling/padding, and timestamps. Keep
location relative to the body available where needed rather than removing all
position information through wrist normalization.

Signer-disjoint splitting and coverage need to be audited for the chosen subset;
do not assume every upstream split meets the project's evaluation requirements.
With few samples, some classes may need additional consented recordings or a
smaller evaluation scope. Add idle/non-target footage and a rejection strategy
before treating a closed-set classifier as a live recognizer.

For each experiment, record commit, lockfile/config identifiers, data and feature
versions, vocabulary size, seed, split counts, metrics, hardware, and latency.
Curate small results under `reports/`; keep generated output under ignored `runs/`.
