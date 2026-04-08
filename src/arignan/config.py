from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from arignan.model_registry import DEFAULT_LIGHT_LOCAL_LLM_REPO_ID, DEFAULT_LOCAL_LLM_REPO_ID, infer_local_llm_backend
from arignan.paths import resolve_app_home, resolve_settings_path

APP_HOME_ENV = "ARIGNAN_HOME"


@dataclass(slots=True)
class ChunkingConfig:
    chunk_size: int = 2800
    chunk_overlap: int = 160


@dataclass(slots=True)
class RetrievalConfig:
    dense_top_k: int = 14
    lexical_top_k: int = 14
    map_top_k: int = 8
    fused_top_k: int = 20
    rerank_top_k: int = 14
    answer_context_top_k_default: int = 10
    answer_context_top_k_light: int = 8
    answer_context_top_k_none: int = 10
    answer_context_top_k_raw: int = 10


@dataclass(slots=True)
class SessionConfig:
    kv_cache_enabled: bool = True
    idle_timeout_minutes: int = 30
    soft_token_limit: int = 12000
    keep_recent_turns: int = 8


@dataclass(slots=True)
class MarkdownConfig:
    max_md_length: int = 4000


@dataclass(slots=True)
class AppConfig:
    local_llm_backend: str = "ollama"
    local_llm_model: str = DEFAULT_LOCAL_LLM_REPO_ID
    local_llm_light_model: str = DEFAULT_LIGHT_LOCAL_LLM_REPO_ID
    local_llm_endpoint: str = "http://127.0.0.1:11434"
    local_llm_keep_alive: str = "10m"
    local_llm_timeout_seconds: int = 120
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
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

    config = AppConfig(app_home=resolved_home)
    raw = _load_json(resolved_settings)
    if not raw:
        return config

    if "local_llm_backend" not in raw:
        raw["local_llm_backend"] = infer_local_llm_backend(raw.get("local_llm_model"), default=config.local_llm_backend)

    if "embedding_model" in raw and raw["embedding_model"] != config.embedding_model:
        raise ValueError("embedding_model is fixed at build time and cannot be overridden in settings.json")

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
