from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_setup_module():
    setup_path = Path("setup.py").resolve()
    spec = importlib.util.spec_from_file_location("arignan_repo_setup", setup_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_setup_py_detects_packaging_invocations() -> None:
    module = _load_setup_module()

    assert module.is_packaging_invocation(["setup.py", "egg_info"])
    assert module.is_packaging_invocation(["setup.py", "bdist_wheel"])
    assert module.is_packaging_invocation(["setup.py", "editable_wheel", "--dist-dir", "tmp"])
    assert not module.is_packaging_invocation(["setup.py"])
    assert not module.is_packaging_invocation(["setup.py", "--dev"])


def test_setup_py_parser_accepts_lightweight_flag() -> None:
    module = _load_setup_module()

    args = module.build_parser().parse_args(["--lightweight"])

    assert args.lightweight is True


def test_check_venv_raises_when_not_in_venv(monkeypatch) -> None:
    module = _load_setup_module()
    # Simulate system Python: prefix == base_prefix means no venv is active.
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)

    with pytest.raises(SystemExit) as exc_info:
        module._check_venv()

    assert exc_info.value.code == 1


def test_check_venv_passes_when_inside_venv(monkeypatch) -> None:
    module = _load_setup_module()
    # Simulate an active venv: prefix differs from base_prefix.
    monkeypatch.setattr(sys, "prefix", "/tmp/fake-venv")

    # Should return normally without raising.
    module._check_venv()
