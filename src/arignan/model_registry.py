from __future__ import annotations

from pathlib import Path

DEFAULT_LOCAL_LLM_DISPLAY_NAME = "Qwen3-1.7B"
DEFAULT_LOCAL_LLM_REPO_ID = "Qwen/Qwen3-1.7B"
MODEL_REPO_ALIASES = {
    DEFAULT_LOCAL_LLM_DISPLAY_NAME: DEFAULT_LOCAL_LLM_REPO_ID,
}


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "__").replace("\\", "__").replace(":", "_")


def resolve_model_repo_id(model_id: str) -> str:
    return MODEL_REPO_ALIASES.get(model_id, model_id)


def resolve_model_storage_dir(app_home: Path, model_id: str) -> Path:
    repo_id = resolve_model_repo_id(model_id)
    return app_home / "models" / sanitize_model_id(repo_id)


__all__ = [
    "DEFAULT_LOCAL_LLM_DISPLAY_NAME",
    "DEFAULT_LOCAL_LLM_REPO_ID",
    "MODEL_REPO_ALIASES",
    "resolve_model_repo_id",
    "resolve_model_storage_dir",
    "sanitize_model_id",
]
