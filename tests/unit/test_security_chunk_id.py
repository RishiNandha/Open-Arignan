"""Tests for SHA-256 chunk IDs and parameter bounds (Fixes #15 and #16)."""
from __future__ import annotations

from pathlib import Path

import pytest

from arignan.indexing import Chunker
from arignan.indexing.chunking import _MAX_CHUNK_SIZE, _MAX_CHUNK_OVERLAP
from arignan.indexing.embedding import HashingEmbedder, _MAX_HASHING_EMBEDDER_DIMENSION
from arignan.models import DocumentSection, ParsedDocument, SourceDocument, SourceType


def _make_document(text: str = "Hello world. This is a test document with some content.") -> ParsedDocument:
    return ParsedDocument(
        load_id="load-1",
        hat="default",
        source=SourceDocument(
            source_type=SourceType.MARKDOWN,
            source_uri="test.md",
            local_path=Path("test.md"),
        ),
        full_text=text,
        sections=[],
        keywords=[],
    )


class TestChunkIdFormat:
    def test_chunk_id_uses_sha256_prefix(self) -> None:
        chunker = Chunker()
        doc = _make_document()
        chunks = chunker.chunk_document(doc)
        assert chunks
        for chunk in chunks:
            assert chunk.chunk_id.startswith("chunk-")
            hex_part = chunk.chunk_id[len("chunk-"):]
            assert len(hex_part) == 20, f"Expected 20 hex chars, got {len(hex_part)}: {hex_part!r}"
            int(hex_part, 16)

    def test_chunk_ids_are_deterministic(self) -> None:
        chunker = Chunker()
        doc = _make_document()
        chunks1 = chunker.chunk_document(doc)
        chunks2 = chunker.chunk_document(doc)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_chunk_ids_unique_in_bulk(self) -> None:
        chunker = Chunker(chunk_size=50, chunk_overlap=10)
        long_text = " ".join(f"Sentence number {i} with some extra words to fill space." for i in range(200))
        doc = _make_document(long_text)
        chunks = chunker.chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"


class TestChunkerBounds:
    def test_chunk_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            Chunker(chunk_size=0)

    def test_chunk_size_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            Chunker(chunk_size=-1)

    def test_chunk_size_exceeds_max_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must not exceed"):
            Chunker(chunk_size=_MAX_CHUNK_SIZE + 1)

    def test_chunk_overlap_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must not be negative"):
            Chunker(chunk_size=100, chunk_overlap=-1)

    def test_chunk_overlap_exceeds_max_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must not exceed"):
            Chunker(chunk_size=_MAX_CHUNK_SIZE, chunk_overlap=_MAX_CHUNK_OVERLAP + 1)

    def test_chunk_overlap_gte_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
            Chunker(chunk_size=100, chunk_overlap=100)

    def test_valid_params_accepted(self) -> None:
        chunker = Chunker(chunk_size=1000, chunk_overlap=100)
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 100

    def test_max_chunk_size_constant_is_sane(self) -> None:
        assert _MAX_CHUNK_SIZE == 32_768

    def test_max_chunk_overlap_constant_is_sane(self) -> None:
        assert _MAX_CHUNK_OVERLAP == 8_192


class TestHashingEmbedderBounds:
    def test_dimension_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="dimension must be positive"):
            HashingEmbedder(dimension=0)

    def test_dimension_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="dimension must be positive"):
            HashingEmbedder(dimension=-5)

    def test_dimension_exceeds_max_raises(self) -> None:
        with pytest.raises(ValueError, match=f"dimension must not exceed {_MAX_HASHING_EMBEDDER_DIMENSION}"):
            HashingEmbedder(dimension=_MAX_HASHING_EMBEDDER_DIMENSION + 1)

    def test_absurdly_large_dimension_raises(self) -> None:
        with pytest.raises(ValueError):
            HashingEmbedder(dimension=10_000_000)

    def test_valid_dimension_accepted(self) -> None:
        embedder = HashingEmbedder(dimension=128)
        assert embedder.dimension == 128

    def test_max_dimension_boundary_accepted(self) -> None:
        embedder = HashingEmbedder(dimension=_MAX_HASHING_EMBEDDER_DIMENSION)
        assert embedder.dimension == _MAX_HASHING_EMBEDDER_DIMENSION
