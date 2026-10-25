# Repository instructions

- Read README.md and docs/project-decisions.md before changing project scope.
- Python 3.12; use the project .venv. Dependencies come from pyproject.toml and uv.lock.
- Use ordinary Python modules under src/sign_language; scripts should be thin entry points.
- Keep CPU and CUDA extras separate. Bare uv sync removes optional dependencies.
- Run scripts/check.ps1 on Windows, or the equivalent documented Linux checks.
- Keep videos, features, weights, credentials, and environments out of Git.
- Do not claim recognition or GPU behavior is validated by import checks alone;
  state exactly which runtime checks were performed.
- Do not claim training or webcam functionality exists before it is implemented.
- Record decisions and implementation status in the repository docs.
