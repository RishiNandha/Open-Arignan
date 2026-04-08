from __future__ import annotations

from pathlib import Path

import pytest

from arignan.storage import StorageLayout, validate_hat_name


def test_storage_layout_ensure_creates_readme_shape(app_home: Path) -> None:
    layout = StorageLayout.from_home(app_home).ensure()

    assert layout.root.exists()
    assert layout.ingestion_log_path.exists()
    assert layout.global_map_path.exists()
    assert layout.hat("default").vector_index_dir.exists()
    assert layout.hat("default").bm25_index_dir.exists()
    assert layout.hat("default").summaries_dir.exists()
    assert layout.hat("default").map_path.exists()


def test_storage_layout_creates_custom_hat(app_home: Path) -> None:
    layout = StorageLayout.from_home(app_home).ensure(include_default_hat=False)

    robotics = layout.hat("robotics").ensure()

    assert robotics.root.exists()
    assert robotics.map_path.exists()


@pytest.mark.parametrize("value", ["", " auto ", "bad/name", "with space"])
def test_validate_hat_name_rejects_invalid_names(value: str) -> None:
    with pytest.raises(ValueError):
        validate_hat_name(value)
