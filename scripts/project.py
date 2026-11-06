"""Thin project CLI; always run with this project's .venv."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sign_language.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
