from __future__ import annotations

import json
from pathlib import Path

from arignan.config import AppConfig, load_config, write_default_settings


def test_load_config_uses_defaults_when_settings_missing(app_home: Path) -> None:
    config = load_config(app_home=app_home, environ={})

    assert isinstance(config, AppConfig)
    assert config.app_home == app_home.resolve()
    assert config.ask_route_backend == "llm"
    assert config.mcp_llm_backend == "client"
    assert config.local_llm_backend == "ollama"
    assert config.local_llm_model == "qwen3:4b-q4_K_M"
    assert config.local_llm_light_model == "qwen3:0.6b"
    assert config.local_llm_endpoint == "http://127.0.0.1:11434"
    assert config.embedding_model == "BAAI/bge-base-en-v1.5"
    assert config.reranker_model == "mixedbread-ai/mxbai-rerank-base-v1"
    assert config.local_llm_keep_alive == "30m"
    assert config.local_llm_timeout_seconds == 300
    assert config.local_llm_context_window == 6144
    assert config.local_llm_flash_attention is True
    assert config.local_llm_kv_cache_type == "q8_0"
    assert config.chunking.chunk_size == 5600
    assert config.chunking.chunk_overlap == 80
    assert config.retrieval.dense_top_k == 10
    assert config.retrieval.fused_top_k == 16
    assert config.retrieval.rerank_top_k == 8
    assert config.retrieval.answer_context_top_k_default == 8
    assert config.retrieval.answer_context_top_k_light == 6
    assert config.retrieval.answer_context_top_k_none == 8
    assert config.retrieval.answer_context_top_k_raw == 8
    assert config.session.soft_token_limit == 18000
    assert config.session.keep_recent_turns == 10
    assert config.markdown.max_md_length == 5000
    assert (app_home / "settings.json").exists()


def test_load_config_recreates_missing_settings_file(app_home: Path) -> None:
    settings_path = app_home / "settings.json"
    if settings_path.exists():
        settings_path.unlink()

    config = load_config(app_home=app_home, environ={})

    assert config.app_home == app_home.resolve()
    assert settings_path.exists()
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["ask_route_backend"] == "llm"
    assert payload["mcp_llm_backend"] == "client"
    assert payload["local_llm_model"] == "qwen3:4b-q4_K_M"


def test_load_config_merges_nested_overrides(app_home: Path) -> None:
    settings_path = app_home / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(
            {
                "ask_route_backend": "embedding",
                "mcp_llm_backend": "local",
                "mcp_retrieval_keep_alive_seconds": 45,
                "local_llm_backend": "transformers",
                "local_llm_model": "custom-llm",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "reranker_model": "mixedbread-ai/mxbai-rerank-xsmall-v1",
                "retrieval": {"dense_top_k": 12},
                "session": {"soft_token_limit": 4096},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(app_home=app_home)

    assert config.ask_route_backend == "embedding"
    assert config.mcp_llm_backend == "local"
    assert config.local_llm_backend == "transformers"
    assert config.local_llm_model == "custom-llm"
    assert config.local_llm_light_model == "qwen3:0.6b"
    assert config.embedding_model == "BAAI/bge-small-en-v1.5"
    assert config.reranker_model == "mixedbread-ai/mxbai-rerank-xsmall-v1"
    assert config.retrieval.dense_top_k == 12
    assert config.session.soft_token_limit == 4096


def test_load_config_migrates_legacy_modernbert_defaults(app_home: Path) -> None:
    settings_path = app_home / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(
            {
                "embedding_model": "Alibaba-NLP/gte-modernbert-base",
                "reranker_model": "Alibaba-NLP/gte-reranker-modernbert-base",
            }
        ),
        encoding="utf-8",
    )

    config = load_config(app_home=app_home)

    assert config.embedding_model == "BAAI/bge-base-en-v1.5"
    assert config.reranker_model == "mixedbread-ai/mxbai-rerank-base-v1"


def test_write_default_settings_creates_file(app_home: Path) -> None:
    settings_path = write_default_settings(app_home=app_home)

    assert settings_path.exists()
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["app_home"] == str(app_home.resolve())
    assert payload["ask_route_backend"] == "llm"
    assert payload["mcp_llm_backend"] == "client"
    assert payload["local_llm_backend"] == "ollama"
    assert payload["local_llm_model"] == "qwen3:4b-q4_K_M"
    assert payload["local_llm_light_model"] == "qwen3:0.6b"
    assert payload["embedding_model"] == "BAAI/bge-base-en-v1.5"
    assert payload["reranker_model"] == "mixedbread-ai/mxbai-rerank-base-v1"
    assert payload["local_llm_keep_alive"] == "30m"
    assert payload["local_llm_timeout_seconds"] == 300
    assert payload["local_llm_context_window"] == 6144
    assert payload["local_llm_flash_attention"] is True
    assert payload["local_llm_kv_cache_type"] == "q8_0"
    assert payload["local_llm_num_parallel"] == 1
    assert payload["local_llm_max_loaded_models"] == 1
    assert payload["chunking"]["chunk_size"] == 5600
    assert payload["chunking"]["chunk_overlap"] == 80
    assert payload["retrieval"]["dense_top_k"] == 10
    assert payload["retrieval"]["fused_top_k"] == 16
    assert payload["retrieval"]["rerank_top_k"] == 8
    assert payload["retrieval"]["answer_context_top_k_default"] == 8
    assert payload["retrieval"]["answer_context_top_k_light"] == 6
    assert payload["retrieval"]["answer_context_top_k_none"] == 8
    assert payload["retrieval"]["answer_context_top_k_raw"] == 8
    assert payload["session"]["soft_token_limit"] == 18000
    assert payload["session"]["keep_recent_turns"] == 10
    assert payload["markdown"]["max_md_length"] == 5000
