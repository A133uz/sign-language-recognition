# Development workflow

Use Python 3.12 and the committed uv 0.13.5 lockfile. Work in ordinary `.py`
files; notebooks are optional and not needed for the main pipeline.

## Checks

```powershell
.\scripts\check.ps1
.\.venv\Scripts\python.exe -m ruff format .
uv build
```

Run formatting before checks if you changed Python code. New functionality
should have focused tests for meaningful data or behavior failures. Runtime
claims need actual webcam/GPU checks, not just import or unit-test success.

## Dependencies

Change `pyproject.toml`, then resolve deliberately:

```powershell
uv lock
.\scripts\setup.ps1 -Profile cuda
.\scripts\check.ps1
```

Commit `pyproject.toml` and `uv.lock` together. Use `uv lock --upgrade-package NAME`
for an intentional package upgrade. Update the uv version in `pyproject.toml`,
script diagnostics, CI, and installation docs together. CPU and CUDA extras cannot be combined;
do not use `--all-extras`. Bare `uv sync` selects the development profile and
removes optional dependencies; use the setup script to retain the full profile.

## Git

Keep `main` usable. Use a branch such as `feature/landmark-extraction` for work
spanning multiple commits. Review `git status` and `git diff --staged` before
committing. Commit code, configs, manifests, documentation, and curated metrics;
keep videos, landmark arrays, model weights, secrets, and environments ignored.

Create an empty remote repository, then connect its actual URL:

```powershell
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Replace the placeholder before running. A remote is not created by local setup.
Once hosted, require the CI checks before merging to `main` if branch protection
is available. No repository-specific identity or remote credentials are embedded.

For published model artifacts, record the producing commit, vocabulary order,
feature schema, configuration, dependency lock, metrics, and SHA-256 checksum.
Store binaries externally or deliberately adopt Git LFS for selected releases.
