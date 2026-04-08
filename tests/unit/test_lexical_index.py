from __future__ import annotations

from pathlib import Path

from arignan.indexing import LexicalIndex, tokenize
from arignan.models import ChunkMetadata, ChunkRecord


def test_tokenize_normalizes_words() -> None:
    assert tokenize("Dense+Keyword Retrieval, JEPA!") == ["dense", "keyword", "retrieval", "jepa"]


def test_lexical_index_scores_keyword_matches(tmp_path: Path) -> None:
    index = LexicalIndex(tmp_path / "bm25")
    chunks = [
        ChunkRecord(
            chunk_id="chunk-1",
            text="JEPA uses joint embedding prediction for world models",
            metadata=ChunkMetadata(load_id="load-1", hat="default", source_uri="a.md"),
        ),
        ChunkRecord(
            chunk_id="chunk-2",
            text="Mango smoothie recipes with yogurt",
            metadata=ChunkMetadata(load_id="load-2", hat="default", source_uri="b.md"),
        ),
    ]

    index.upsert(chunks)
    hits = index.search("joint embedding model", limit=2)

    assert hits
    assert hits[0].chunk_id == "chunk-1"
