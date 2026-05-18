"""Tests for thread-safe file locking in LocalDenseIndex (Fix #8)."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from arignan.indexing.dense import LocalDenseIndex
from arignan.models import ChunkMetadata, ChunkRecord


def _make_chunk(chunk_id: str, text: str = "hello") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(load_id="load-1", hat="default", source_uri="a.md"),
        embedding=[0.1, 0.2, 0.3],
    )


def _json_index(tmp_path: Path) -> LocalDenseIndex:
    """Return a LocalDenseIndex that always uses the JSON backend (no Qdrant)."""
    with patch("arignan.indexing.dense.LocalDenseIndex._try_create_qdrant_client", return_value=None):
        return LocalDenseIndex(tmp_path)


class TestLocalDenseIndexLocking:
    def test_upsert_requires_embedding(self, tmp_path: Path) -> None:
        index = _json_index(tmp_path)
        chunk = ChunkRecord(
            chunk_id="c1",
            text="text",
            metadata=ChunkMetadata(load_id="load-1", hat="default", source_uri="a.md"),
            embedding=None,
        )
        with pytest.raises(ValueError, match="embedding"):
            index.upsert([chunk])

    def test_upsert_persists_chunk(self, tmp_path: Path) -> None:
        index = _json_index(tmp_path)
        index.upsert([_make_chunk("c1")])
        all_chunks = index.all_chunks()
        assert any(c.chunk_id == "c1" for c in all_chunks)

    def test_concurrent_upserts_do_not_corrupt(self, tmp_path: Path) -> None:
        """Multiple threads upserting simultaneously must not corrupt the JSON index."""
        index = _json_index(tmp_path)
        errors: list[Exception] = []

        def worker(chunk_id: str) -> None:
            try:
                index.upsert([_make_chunk(chunk_id, text=f"text for {chunk_id}")])
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(f"c{i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in concurrent upserts: {errors}"
        all_chunks = index.all_chunks()
        ids = {c.chunk_id for c in all_chunks}
        assert len(ids) == 20, f"Expected 20 unique chunks, got {len(ids)}: {ids}"

    def test_atomic_write_no_partial_file(self, tmp_path: Path) -> None:
        """Storage file must be valid JSON after every upsert (atomic rename)."""
        index = _json_index(tmp_path)
        for i in range(5):
            index.upsert([_make_chunk(f"c{i}")])

        content = index.storage_path.read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert isinstance(parsed, list)
        assert len(parsed) == 5

    def test_delete_load_removes_correct_chunks(self, tmp_path: Path) -> None:
        index = _json_index(tmp_path)
        c1 = ChunkRecord(
            chunk_id="keep",
            text="keep",
            metadata=ChunkMetadata(load_id="load-keep", hat="default", source_uri="a.md"),
            embedding=[0.1, 0.2, 0.3],
        )
        c2 = ChunkRecord(
            chunk_id="remove",
            text="remove",
            metadata=ChunkMetadata(load_id="load-remove", hat="default", source_uri="b.md"),
            embedding=[0.4, 0.5, 0.6],
        )
        index.upsert([c1, c2])
        index.delete_load("load-remove")
        remaining = {c.chunk_id for c in index.all_chunks()}
        assert "keep" in remaining
        assert "remove" not in remaining

    def test_write_lock_exists(self, tmp_path: Path) -> None:
        """LocalDenseIndex must have a threading.Lock for the JSON path."""
        import threading
        index = _json_index(tmp_path)
        assert hasattr(index, "_write_lock")
        assert isinstance(index._write_lock, type(threading.Lock()))
