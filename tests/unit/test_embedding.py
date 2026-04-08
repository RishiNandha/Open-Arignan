from __future__ import annotations

from arignan.indexing import DEFAULT_EMBEDDING_MODEL, HashingEmbedder, cosine_similarity


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
