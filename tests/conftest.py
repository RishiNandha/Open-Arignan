from __future__ import annotations

from pathlib import Path

import pytest

@pytest.fixture()
def app_home(tmp_path: Path) -> Path:
    return tmp_path / ".arignan"
