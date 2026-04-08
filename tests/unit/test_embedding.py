from __future__ import annotations

import sys
import types

from arignan.indexing import DEFAULT_EMBEDDING_MODEL, HashingEmbedder, cosine_similarity
from arignan.indexing.embedding import SentenceTransformerEmbedder


def test_hashing_embedder_is_deterministic() -> None:
    embedder = HashingEmbedder(dimension=12)

    first = embedder.embed_query("joint embedding predictive architecture")
    second = embedder.embed_query("joint embedding predictive architecture")

    assert first == second
    assert len(first) == 12
    assert embedder.model_name == DEFAULT_EMBEDDING_MODEL


def test_cosine_similarity_prefers_related_texts() -> None:
    embedder = HashingEmbedder(dimension=16)

    query = embedder.embed_query("keyword dense retrieval")
    related = embedder.embed_query("dense retrieval with keyword fusion")
    unrelated = embedder.embed_query("fresh mango smoothie recipe")

    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


def test_sentence_transformer_embedder_prefers_cuda_when_available(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str) -> None:
            captured["model_name"] = model_name
            captured["device"] = device

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr("arignan.indexing.embedding.preferred_torch_device", lambda: "cuda")

    embedder = SentenceTransformerEmbedder()

    assert embedder.device == "cuda"
    assert captured == {"model_name": DEFAULT_EMBEDDING_MODEL, "device": "cuda"}
