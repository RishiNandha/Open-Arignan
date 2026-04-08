from __future__ import annotations

from arignan.models import ChunkMetadata, RetrievalHit, RetrievalSource
from arignan.retrieval import DEFAULT_RERANKER_MODEL, HeuristicReranker


def test_heuristic_reranker_reorders_hits_by_query_overlap() -> None:
    reranker = HeuristicReranker()
    hits = [
        RetrievalHit(
            chunk_id="unrelated",
            text="Therapy notes and cognitive biases",
            score=0.9,
            source=RetrievalSource.LEXICAL,
            metadata=ChunkMetadata(load_id="load-1", hat="default", source_uri="psych.md"),
        ),
        RetrievalHit(
            chunk_id="related",
            text="Joint embedding predictive architecture notes",
            score=0.4,
            source=RetrievalSource.DENSE,
            metadata=ChunkMetadata(load_id="load-2", hat="default", source_uri="jepa.md"),
        ),
    ]

    reranked = reranker.rerank("joint embedding architecture", hits, limit=2)

    assert reranker.model_name == DEFAULT_RERANKER_MODEL
    assert reranked[0].chunk_id == "related"


def test_heuristic_reranker_prunes_below_threshold() -> None:
    reranker = HeuristicReranker()
    hits = [
        RetrievalHit(
            chunk_id="keep",
            text="retrieval augmented generation pipeline",
            score=0.3,
            source=RetrievalSource.DENSE,
            metadata=ChunkMetadata(load_id="load-1", hat="default", source_uri="rag.md"),
        ),
        RetrievalHit(
            chunk_id="drop",
            text="fresh mango smoothie recipe",
            score=0.8,
            source=RetrievalSource.DENSE,
            metadata=ChunkMetadata(load_id="load-2", hat="default", source_uri="food.md"),
        ),
    ]

    reranked = reranker.rerank("retrieval pipeline", hits, limit=5, min_score=0.4)

    assert [hit.chunk_id for hit in reranked] == ["keep"]
