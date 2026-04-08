from __future__ import annotations

from pathlib import Path

DEFAULT_LOCAL_LLM_DISPLAY_NAME = "qwen3:4b-q4_K_M"
DEFAULT_LOCAL_LLM_REPO_ID = "qwen3:4b-q4_K_M"
DEFAULT_LIGHT_LOCAL_LLM_DISPLAY_NAME = "qwen3:0.6b"
DEFAULT_LIGHT_LOCAL_LLM_REPO_ID = "qwen3:0.6b"
LEGACY_TRANSFORMERS_LOCAL_LLM_DISPLAY_NAME = "Qwen3-1.7B"
LEGACY_TRANSFORMERS_LOCAL_LLM_REPO_ID = "Qwen/Qwen3-1.7B"
LIGHT_TRANSFORMERS_LOCAL_LLM_DISPLAY_NAME = "Qwen3-0.6B"
LIGHT_TRANSFORMERS_LOCAL_LLM_REPO_ID = "Qwen/Qwen3-0.6B"
MODEL_REPO_ALIASES = {
    LEGACY_TRANSFORMERS_LOCAL_LLM_DISPLAY_NAME: LEGACY_TRANSFORMERS_LOCAL_LLM_REPO_ID,
    LIGHT_TRANSFORMERS_LOCAL_LLM_DISPLAY_NAME: LIGHT_TRANSFORMERS_LOCAL_LLM_REPO_ID,
}
OLLAMA_MODEL_ALIASES = {
    DEFAULT_LOCAL_LLM_DISPLAY_NAME: DEFAULT_LOCAL_LLM_REPO_ID,
    "qwen3:4b": DEFAULT_LOCAL_LLM_REPO_ID,
    DEFAULT_LIGHT_LOCAL_LLM_DISPLAY_NAME: DEFAULT_LIGHT_LOCAL_LLM_REPO_ID,
    LEGACY_TRANSFORMERS_LOCAL_LLM_DISPLAY_NAME: DEFAULT_LOCAL_LLM_REPO_ID,
    LEGACY_TRANSFORMERS_LOCAL_LLM_REPO_ID: DEFAULT_LOCAL_LLM_REPO_ID,
    LIGHT_TRANSFORMERS_LOCAL_LLM_DISPLAY_NAME: DEFAULT_LIGHT_LOCAL_LLM_REPO_ID,
    LIGHT_TRANSFORMERS_LOCAL_LLM_REPO_ID: DEFAULT_LIGHT_LOCAL_LLM_REPO_ID,
}


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "__").replace("\\", "__").replace(":", "_")


def resolve_model_repo_id(model_id: str) -> str:
    return MODEL_REPO_ALIASES.get(model_id, model_id)


def resolve_ollama_model_id(model_id: str) -> str:
    return OLLAMA_MODEL_ALIASES.get(model_id, model_id)


def infer_local_llm_backend(model_id: str | None, default: str = "ollama") -> str:
    if not model_id:
        return default
    if ":" in model_id:
        return "ollama"
    if "/" in model_id or model_id in MODEL_REPO_ALIASES:
        return "transformers"
    if model_id in OLLAMA_MODEL_ALIASES:
        return "ollama"
    return default


def resolve_model_storage_dir(app_home: Path, model_id: str) -> Path:
    repo_id = resolve_model_repo_id(model_id)
    return app_home / "models" / sanitize_model_id(repo_id)


__all__ = [
    "DEFAULT_LOCAL_LLM_DISPLAY_NAME",
    "DEFAULT_LOCAL_LLM_REPO_ID",
    "DEFAULT_LIGHT_LOCAL_LLM_DISPLAY_NAME",
    "DEFAULT_LIGHT_LOCAL_LLM_REPO_ID",
    "LEGACY_TRANSFORMERS_LOCAL_LLM_DISPLAY_NAME",
    "LEGACY_TRANSFORMERS_LOCAL_LLM_REPO_ID",
    "LIGHT_TRANSFORMERS_LOCAL_LLM_DISPLAY_NAME",
    "LIGHT_TRANSFORMERS_LOCAL_LLM_REPO_ID",
    "MODEL_REPO_ALIASES",
    "OLLAMA_MODEL_ALIASES",
    "infer_local_llm_backend",
    "resolve_model_repo_id",
    "resolve_ollama_model_id",
    "resolve_model_storage_dir",
    "sanitize_model_id",
]
