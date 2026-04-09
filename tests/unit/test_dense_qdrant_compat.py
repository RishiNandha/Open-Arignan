from __future__ import annotations

from pathlib import Path

from arignan.indexing import DenseIndexer, HashingEmbedder
from arignan.indexing.dense import LocalDenseIndex
from arignan.models import ChunkMetadata, ChunkRecord


class _FakeScoredPoint:
    def __init__(self) -> None:
        self.payload = {
            "chunk_id": "chunk-1",
            "text": "Joint embedding predictive architecture.",
            "metadata": {
                "load_id": "load-1",
                "hat": "default",
                "source_uri": "memory://doc",
                "source_type": "markdown",
                "source_path": "doc.md",
            },
        }
        self.score = 0.87


class _FakeQueryResponse:
    def __init__(self) -> None:
        self.points = [_FakeScoredPoint()]


class _FakeQdrantClientWithoutSearch:
    def collection_exists(self, name: str) -> bool:
        return True

    def query_points(self, *, collection_name: str, query: list[float], limit: int, with_payload: bool):
        assert collection_name == "arignan_chunks"
        assert query == [0.1, 0.2, 0.3]
        assert limit == 2
        assert with_payload is True
        return _FakeQueryResponse()


def test_local_dense_index_supports_qdrant_query_points_without_search(tmp_path: Path) -> None:
    index = LocalDenseIndex(tmp_path / "vector_index")
    index._qdrant_client = _FakeQdrantClientWithoutSearch()

    hits = index.search([0.1, 0.2, 0.3], limit=2)

    assert len(hits) == 1
    assert hits[0].chunk_id == "chunk-1"
    assert hits[0].score == 0.87
    assert hits[0].metadata.load_id == "load-1"


def test_local_dense_index_recreates_qdrant_collection_when_vector_size_changes(tmp_path: Path) -> None:
    index = LocalDenseIndex(tmp_path / "vector_index")
    chunks = [
        ChunkRecord(
            chunk_id="chunk-old",
            text="old retrieval chunk",
            metadata=ChunkMetadata(load_id="load-old", hat="default", source_uri="old.md"),
        )
    ]
    DenseIndexer(HashingEmbedder(dimension=24), index).index_chunks(chunks)

    new_chunks = [
        ChunkRecord(
            chunk_id="chunk-new",
            text="new retrieval chunk",
            metadata=ChunkMetadata(load_id="load-new", hat="default", source_uri="new.md"),
        )
    ]
    DenseIndexer(HashingEmbedder(dimension=768), index).index_chunks(new_chunks)

    records = index.all_chunks()
    assert [record.chunk_id for record in records] == ["chunk-new"]
    assert len(records[0].embedding or []) == 768
