from __future__ import annotations

from pathlib import Path

import pytest
from arignan.indexing.embedding import HashingEmbedder
from arignan.retrieval.reranking import HeuristicReranker


@pytest.fixture()
def app_home(tmp_path: Path) -> Path:
    return tmp_path / ".arignan"


@pytest.fixture(autouse=True)
def _stub_application_retrieval_runtime(monkeypatch):
    def fake_create_embedder(config, **kwargs):
        return HashingEmbedder(model_name=config.embedding_model)

    def fake_create_reranker(config, **kwargs):
        return HeuristicReranker(model_name=config.reranker_model)

    monkeypatch.setattr("arignan.application.create_embedder", fake_create_embedder)
    monkeypatch.setattr("arignan.application.create_reranker", fake_create_reranker)
