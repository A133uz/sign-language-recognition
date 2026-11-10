# Local background landmark extraction

The user authorized full local extraction with **two CPU workers** on
2026-11-07 and increased it to **four workers** on 2026-11-08.
It was launched as a hidden Windows background process using
`scripts/local_landmark_job.py`. This is hands-plus-pose landmark inference,
not GRU/classifier training.

The live manifest confirms all **83,399** official train/val/test clips,
**836** shards of up to 100 clips, `limit=null`, CPU delegation and pose enabled.
Run ID:
`f55c6cdb4e7e711d7d850cd8c3cbd0776e91ac5a873f98dc2476257b32659cf2`.

Output:
`data/landmarks/asl-citizen-raw-v1-cpu-hands-pose/`.
Control state and attempt-specific logs:
`.cache/local-landmark-extraction/`.
Temporary shard work/locks: `.cache/extraction-work/`.
All dataset/features/runtime logs remain ignored by Git.

## Check progress

Run from the project root:

```powershell
.venv\Scripts\python.exe scripts/local_landmark_job.py status
```

This reports the supervisor's live lock state, worker count, current process
IDs, heartbeat, log locations, recent progress and committed-batch summary.
The log prints progress every ten processed clips, while `summary.json` updates
after a shard commits. An unfinished batch is not yet a durable completed shard.
Use the status command for live progress rather than a Markdown snapshot.

The initial attempt started at 2026-11-07 19:04:36 Asia/Tashkent
(14:04:36 UTC), with supervisor PID 2072 and extraction launcher PID 12824.
These are historical startup IDs, not permanent identifiers for later retries.
Initial logs are `20261107T140436Z.stdout.log` and
`20261107T140436Z.native.log` under the control directory. Startup was verified
with advancing video progress, the full run manifest and a live heartbeat.

## Stop or resume

To request a stop:

```powershell
.venv\Scripts\python.exe scripts/local_landmark_job.py stop
```

The supervisor detects the request and terminates only its extraction process
tree. It does not commit a partial batch. Completed verified shards remain
resumable; up to one unfinished batch may repeat. Check status afterward.

To resume in a hidden background process after it has stopped:

```powershell
Start-Process -WindowStyle Hidden `
  -FilePath "$PWD\.venv\Scripts\python.exe" `
  -ArgumentList @('-u', "$PWD\scripts\local_landmark_job.py", 'run', '--workers', '4') `
  -WorkingDirectory "$PWD"
```

The same output directory and configuration resume verified completed shards.
The supervisor prevents duplicate instances, and the extractor also locks its
output/work directory. Do not start a second raw extractor against that folder.
Keep data, model bundles and extraction code unchanged for this run; incompatible
resume settings are rejected. Use a separate folder for a changed schema or
environment. Benchmark outputs are separate from full-run batches.

## Worker crashes and recovery

Two four-worker attempts ended with `BrokenProcessPool`: a worker exited
abruptly, after 45,254 and 65,758 committed clips respectively. The logs do not
establish the underlying native-process failure. Zero failed clip records does
not mean there were no process crashes.

The supervisor now invokes the unchanged extractor with `--max-shards 25`.
After each group of at most 2,500 new clips it starts a fresh extractor and
worker pool, revalidating and resuming completed shards. This bounds worker
lifetime without changing landmark computation, dependencies, source hashes or
the existing cache identity. It adds some startup and cache-verification cost.
Refreshing workers is a mitigation; it does not establish the crash's cause.

If the native log ends with `concurrent.futures.process.BrokenProcessPool`,
the supervisor automatically retries after ten seconds. At most two retries
are allowed without new committed progress; repeated failure at the same
checkpoint becomes `failed`. Other errors are not automatically retried.
Stop requests remain active during retries and between worker refreshes.
Each attempt has separate logs; `attempt_history` in the status records its
exit code, action, logs and committed clip counts. No clips are silently
skipped, and an unfinished shard is repeated in full.

## Background behavior and completion

While running, the supervisor requests that Windows prevent **automatic idle
sleep**. This is scoped to the supervisor's lifetime: no power-plan setting is
changed, the monitor may switch off, and the request is released on completion,
stop or failure. A manual shutdown/sleep/restart can still interrupt extraction.
The local benchmark suggests roughly **37.4 active hours** with two workers;
actual data lengths, machine load and interruptions may change this.

On completion the state becomes `completed`, `completed_with_failures` or
`failed`, with an exit code and completion time. Inspect the extractor summary
and native log for failed clips; do not silently discard them. No training
automatically follows extraction. See [the extraction guide](landmark-extraction.md)
for raw schema, masks, handedness caveats and resume validation, and
[the CPU benchmark](local-cpu-benchmark-2026-11-07.md) for measured throughput.
