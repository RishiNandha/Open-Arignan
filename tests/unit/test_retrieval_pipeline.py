from __future__ import annotations

from arignan.models import ChunkMetadata, RetrievalHit, RetrievalSource
from arignan.retrieval import QueryExpander, reciprocal_rank_fusion


def test_query_expander_adds_abbreviation_expansions() -> None:
    expanded = QueryExpander().expand("JEPA for RAG")

    assert "joint" in expanded
    assert "retrieval" in expanded
    assert "augmented" in expanded


def test_reciprocal_rank_fusion_rewards_overlap_across_channels() -> None:
    shared = RetrievalHit(
        chunk_id="shared",
        text="shared text",
        score=0.9,
        source=RetrievalSource.DENSE,
        metadata=ChunkMetadata(load_id="load-1", hat="default", source_uri="a.md"),
    )
    lexical_shared = RetrievalHit(
        chunk_id="shared",
        text="shared text",
        score=1.4,
        source=RetrievalSource.LEXICAL,
        metadata=ChunkMetadata(load_id="load-1", hat="default", source_uri="a.md"),
    )
    map_only = RetrievalHit(
        chunk_id="map-only",
        text="map text",
        score=0.7,
        source=RetrievalSource.MAP,
        metadata=ChunkMetadata(load_id="map", hat="default", source_uri="map.md"),
    )

    fused = reciprocal_rank_fusion([[shared], [lexical_shared], [map_only]], limit=3)

    assert fused[0].chunk_id == "shared"
    assert fused[0].extras["channels"] == ["dense", "lexical"]
