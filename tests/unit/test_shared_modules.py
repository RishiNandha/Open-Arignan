from __future__ import annotations

from pathlib import Path

from arignan.markdown import compose_topic_markdown
from arignan.markdown.rendering import compose_topic_markdown as compose_topic_markdown_rendering
from arignan.model_registry import infer_local_llm_backend, resolve_model_storage_dir, resolve_ollama_model_id
from arignan.runtime_env import configure_text_runtime_environment


def test_markdown_package_exports_rendering_helpers() -> None:
    assert compose_topic_markdown is compose_topic_markdown_rendering


def test_resolve_model_storage_dir_uses_resolved_repo_id(tmp_path: Path) -> None:
    assert resolve_model_storage_dir(tmp_path, "Qwen3-1.7B") == tmp_path / "models" / "Qwen__Qwen3-1.7B"


def test_model_registry_resolves_ollama_aliases_and_infers_backends() -> None:
    assert resolve_ollama_model_id("Qwen/Qwen3-1.7B") == "qwen3:4b-q4_K_M"
    assert infer_local_llm_backend("qwen3:4b-q4_K_M") == "ollama"
    assert infer_local_llm_backend("Qwen/Qwen3-1.7B") == "transformers"


def test_configure_text_runtime_environment_sets_text_only_flags() -> None:
    env: dict[str, str] = {}

    applied = configure_text_runtime_environment(env)

    assert applied == {
        "TRANSFORMERS_NO_TF": "1",
        "USE_TF": "0",
        "TRANSFORMERS_NO_FLAX": "1",
        "USE_FLAX": "0",
    }
    assert env == applied
