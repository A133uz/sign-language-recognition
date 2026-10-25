"""A failed environment probe must fail clearly rather than claim readiness."""

from types import SimpleNamespace

import numpy as np

from sign_language import cli


def test_cuda_requirement_fails_on_cpu_runtime(monkeypatch):
    torch = SimpleNamespace(
        __version__="test",
        cuda=SimpleNamespace(is_available=lambda: False),
        ones=np.ones,
    )
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: torch if name == "torch" else SimpleNamespace(__version__="test"),
    )
    report, success = cli.doctor(vision=False, require_cuda=True)
    assert not success
    assert report["cuda_available"] is False
    assert report["errors"]


def test_missing_library_is_reported(monkeypatch):
    def missing(name):
        raise ImportError(f"Missing {name}")

    monkeypatch.setattr(cli.importlib, "import_module", missing)
    report, success = cli.doctor(vision=False, require_cuda=False)
    assert not success
    assert any("ImportError" in error for error in report["errors"])
