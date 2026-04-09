from __future__ import annotations

import json
from pathlib import Path

from arignan.config import AppConfig, load_config, write_default_settings


def test_load_config_uses_defaults_when_settings_missing(app_home: Path) -> None:
    config = load_config(app_home=app_home, environ={})

    assert isinstance(config, AppConfig)
    assert config.app_home == app_home.resolve()
    assert config.local_llm_backend == "ollama"
    assert config.local_llm_model == "qwen3:4b-q4_K_M"
    assert config.local_llm_light_model == "qwen3:0.6b"
    assert config.local_llm_endpoint == "http://127.0.0.1:11434"
    assert config.embedding_model == "BAAI/bge-base-en-v1.5"
    assert config.reranker_model == "mixedbread-ai/mxbai-rerank-base-v1"
    assert config.chunking.chunk_size == 2800
    assert config.chunking.chunk_overlap == 160
    assert config.retrieval.dense_top_k == 14
    assert config.retrieval.answer_context_top_k_default == 10
    assert config.retrieval.answer_context_top_k_light == 8
    assert config.retrieval.answer_context_top_k_none == 10
    assert config.retrieval.answer_context_top_k_raw == 10


def test_load_config_merges_nested_overrides(app_home: Path) -> None:
    settings_path = app_home / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(
            {
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

    assert config.local_llm_backend == "transformers"
    assert config.local_llm_model == "custom-llm"
    assert config.local_llm_light_model == "qwen3:0.6b"
    assert config.embedding_model == "BAAI/bge-small-en-v1.5"
    assert config.reranker_model == "mixedbread-ai/mxbai-rerank-xsmall-v1"
    assert config.retrieval.dense_top_k == 12
    assert config.session.soft_token_limit == 4096


def test_write_default_settings_creates_file(app_home: Path) -> None:
    settings_path = write_default_settings(app_home=app_home)

    assert settings_path.exists()
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["app_home"] == str(app_home.resolve())
    assert payload["local_llm_backend"] == "ollama"
    assert payload["local_llm_model"] == "qwen3:4b-q4_K_M"
    assert payload["local_llm_light_model"] == "qwen3:0.6b"
    assert payload["embedding_model"] == "BAAI/bge-base-en-v1.5"
    assert payload["reranker_model"] == "mixedbread-ai/mxbai-rerank-base-v1"
    assert payload["chunking"]["chunk_size"] == 2800
    assert payload["chunking"]["chunk_overlap"] == 160
    assert payload["retrieval"]["dense_top_k"] == 14
    assert payload["retrieval"]["answer_context_top_k_default"] == 10
    assert payload["retrieval"]["answer_context_top_k_light"] == 8
    assert payload["retrieval"]["answer_context_top_k_none"] == 10
    assert payload["retrieval"]["answer_context_top_k_raw"] == 10
    assert payload["markdown"]["max_md_length"] == 4000
