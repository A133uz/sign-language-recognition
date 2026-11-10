# Local CPU landmark extraction benchmark — 2026-11-07

Measured on the user's Intel Core i5-9400F @ 2.90 GHz, six logical CPUs,
Windows 10 build 19043. Environment: Python 3.12.6, MediaPipe 1.2.2,
NumPy 1.26.4, OpenCV-contrib 4.11.0.86. The RTX 4060 was not used.

All configurations processed the **same 100 complete official training videos**,
containing **9,257 frames**. Selection uses the current extractor's sorted train
filenames, `Random(0)` shuffle and first 100 entries. Hands and pose-lite were
enabled, with the existing CPU delegates and thresholds. Each configuration
started a fresh CLI process and wrote a new shard; no feature cache was reused.

| Workers | Time for 100 clips | Seconds/clip | Frames/second | Full-corpus extrapolation |
| --- | ---: | ---: | ---: | ---: |
| 1 | 314.28 s | 3.143 | 29.45 | 72.8 active hours |
| 2 | 161.35 s | 1.613 | 57.37 | 37.4 active hours |
| 4 | 103.89 s | 1.039 | 89.10 | 24.1 active hours |

Timing includes interpreter startup, metadata/model checks, tracker creation
per clip, full video decoding, both landmarkers, numeric feature compression,
ZIP writing, local copy/hash checks and completion-marker writes. Subsequent
feature validation is outside the timer. These timings include local shard I/O
and should not be equated to the earlier Colab comparison's compute-only scope.

**Four workers were the fastest tested setting**, 3.03x faster than one and
1.55x faster than two. Three workers were not measured. A separate 30-second
whole-machine CPU sample averaged 37.4% during the two-worker run and 74.9%
during four workers; these include other applications and are not peak or
process-isolated utilization measurements.

## Validation and limits

- All three runs completed 100/100 clips with zero failures.
- No tensor-write error or other error-level native log line was observed.
  Native warning logs remain available in the benchmark folders.
- All 300 extracted NPZs passed finite-value, expected hand/pose array shape,
  frame-index and increasing-timestamp checks.
- All 100 NPZ feature files were byte-identical between one/two workers and
  between one/four workers. Numeric arrays were also explicitly compared.
- Every run reported hands in 3,952 frames and pose in 9,255 frames. These are
  model detection counts, not independently measured tracking accuracy.
- The identical run ID confirms matching source, models, environment,
  metadata and selected-sample configuration across worker counts:
  `5d1f4a1e5f8c030e465ffd568c97fb1cfbf2320f509c475b2ca67093c3171542`.

The full-corpus figures scale measured wall seconds/clip by 83,399 clips.
They are estimates from one 100-clip training sample, not a full-corpus run or
an estimate of sign-recognition accuracy. Other splits, clip lengths, OS file
cache, machine load, sleep and interruptions can change the result. Runs were
sequential, not simultaneous. One hundred clips produced 7,137,951 bytes of
feature archives, suggesting roughly 5.54 GiB across the corpus at this sample's
compression rate; allow additional disk headroom and sample variation.

Use four workers for an unattended local extraction candidate, or two if
preserving desktop responsiveness matters. Keep the computer awake. Full
extraction and background-job setup were **not started** by this benchmark;
no classifier training occurred.

## Reproduce

With the project venv available:

```powershell
.venv\Scripts\python.exe -u scripts/benchmark_landmarks.py --clips 100 --workers 1 2 4
```

The measured campaign used `--workers 1 2` first, then a separate `--workers 4`
invocation after a CPU utilization sample showed headroom. The script always
creates a new output folder, validates completed arrays, and records timings,
source/model/library provenance and native logs. It does not train a model.

Local artifacts (ignored by Git):

- `.cache/landmark-benchmarks/20261107T134235Z-1465c8/benchmark-report.json`
- `.cache/landmark-benchmarks/20261107T135036Z-345c6f/benchmark-report.json`
- `.cache/landmark-benchmarks/20261107T134235Z-1465c8/combined-report.json`
- Worker-specific stdout/native logs, feature shards and CPU-utilization samples
  in those same directories.
