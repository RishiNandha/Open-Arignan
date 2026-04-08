from __future__ import annotations

import importlib.util
from pathlib import Path


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
    assert not module.is_packaging_invocation(["setup.py"])
    assert not module.is_packaging_invocation(["setup.py", "--dev"])
