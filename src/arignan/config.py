from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from arignan.model_registry import (
    DEFAULT_EMBEDDING_MODEL_REPO_ID,
    DEFAULT_LIGHT_LOCAL_LLM_REPO_ID,
    DEFAULT_LOCAL_LLM_REPO_ID,
    DEFAULT_RERANKER_MODEL_REPO_ID,
    LEGACY_EMBEDDING_MODEL_REPO_ID,
    LEGACY_MODERNBERT_RERANKER_MODEL_REPO_ID,
    infer_local_llm_backend,
)
from arignan.paths import resolve_app_home, resolve_settings_path

APP_HOME_ENV = "ARIGNAN_HOME"


@dataclass(slots=True)
class ChunkingConfig:
    chunk_size: int = 5600
    chunk_overlap: int = 80


@dataclass(slots=True)
class RetrievalConfig:
    dense_top_k: int = 10
    lexical_top_k: int = 10
    map_top_k: int = 4
    fused_top_k: int = 16
    rerank_top_k: int = 8
    answer_context_top_k_default: int = 8
    answer_context_top_k_light: int = 6
    answer_context_top_k_none: int = 8
    answer_context_top_k_raw: int = 8


@dataclass(slots=True)
class SessionConfig:
    kv_cache_enabled: bool = True
    idle_timeout_minutes: int = 30
    soft_token_limit: int = 18000
    keep_recent_turns: int = 10


@dataclass(slots=True)
class MarkdownConfig:
    max_md_length: int = 5000


@dataclass(slots=True)
class AppConfig:
    ask_route_backend: str = "llm"
    local_llm_backend: str = "ollama"
    local_llm_model: str = DEFAULT_LOCAL_LLM_REPO_ID
    local_llm_light_model: str = DEFAULT_LIGHT_LOCAL_LLM_REPO_ID
    local_llm_endpoint: str = "http://127.0.0.1:11434"
    local_llm_keep_alive: str = "30m"
    local_llm_timeout_seconds: int = 300
    local_llm_context_window: int = 6144
    local_llm_flash_attention: bool = True
    local_llm_kv_cache_type: str = "q8_0"
    local_llm_num_parallel: int = 1
    local_llm_max_loaded_models: int = 1
    embedding_model: str = DEFAULT_EMBEDDING_MODEL_REPO_ID
    reranker_model: str = DEFAULT_RERANKER_MODEL_REPO_ID
    default_hat: str = "default"
    app_home: Path = field(default_factory=resolve_app_home)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    markdown: MarkdownConfig = field(default_factory=MarkdownConfig)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["app_home"] = str(self.app_home)
        return data


def _merge_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    for key, value in updates.items():
        current = getattr(instance, key)
        if dataclass_is_instance(current) and isinstance(value, dict):
            _merge_dataclass(current, value)
            continue
        setattr(instance, key, value)
    return instance


def dataclass_is_instance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__") and not isinstance(value, type)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(
    settings_path: Path | None = None,
    app_home: Path | None = None,
    environ: dict[str, str] | None = None,
) -> AppConfig:
    env = environ or os.environ
    resolved_home = resolve_app_home(app_home=app_home, environ=env)
    resolved_settings = resolve_settings_path(settings_path=settings_path, app_home=resolved_home)
    if not resolved_settings.exists():
        write_default_settings(settings_path=resolved_settings, app_home=resolved_home, overwrite=False)

    config = AppConfig(app_home=resolved_home)
    raw = _load_json(resolved_settings)
    if not raw:
        return config

    if "local_llm_backend" not in raw:
        raw["local_llm_backend"] = infer_local_llm_backend(raw.get("local_llm_model"), default=config.local_llm_backend)

    if raw.get("embedding_model") == LEGACY_EMBEDDING_MODEL_REPO_ID:
        raw["embedding_model"] = DEFAULT_EMBEDDING_MODEL_REPO_ID
    if raw.get("reranker_model") == LEGACY_MODERNBERT_RERANKER_MODEL_REPO_ID:
        raw["reranker_model"] = DEFAULT_RERANKER_MODEL_REPO_ID

    if "app_home" in raw:
        raw["app_home"] = Path(raw["app_home"])

    return _merge_dataclass(config, raw)


def write_default_settings(
    settings_path: Path | None = None,
    app_home: Path | None = None,
    overwrite: bool = False,
) -> Path:
    resolved_home = resolve_app_home(app_home=app_home)
    resolved_settings = resolve_settings_path(settings_path=settings_path, app_home=resolved_home)
    resolved_settings.parent.mkdir(parents=True, exist_ok=True)

    if resolved_settings.exists() and not overwrite:
        return resolved_settings

    config = AppConfig(app_home=resolved_home)
    with resolved_settings.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)
        handle.write("\n")
    return resolved_settings
