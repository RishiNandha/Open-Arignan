from __future__ import annotations

from pathlib import Path

from arignan.markdown import compose_topic_markdown
from arignan.markdown.rendering import compose_topic_markdown as compose_topic_markdown_rendering
from arignan.model_registry import resolve_model_storage_dir


def test_markdown_package_exports_rendering_helpers() -> None:
    assert compose_topic_markdown is compose_topic_markdown_rendering


def test_resolve_model_storage_dir_uses_resolved_repo_id(tmp_path: Path) -> None:
    assert resolve_model_storage_dir(tmp_path, "Qwen3-1.7B") == tmp_path / "models" / "Qwen__Qwen3-1.7B"
