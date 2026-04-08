from __future__ import annotations

import json
import math
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from arignan.models import ChunkMetadata, ChunkRecord, RetrievalHit, RetrievalSource

from .embedding import Embedder


class DenseIndex(Protocol):
    def upsert(self, chunks: list[ChunkRecord]) -> None:
        """Persist chunks with embeddings."""

    def search(self, query_embedding: list[float], limit: int) -> list[RetrievalHit]:
        """Search by embedding."""

    def delete_load(self, load_id: str) -> None:
        """Delete all chunks associated with a load."""

    def all_chunks(self) -> list[ChunkRecord]:
        """Return all persisted chunks."""


class LocalDenseIndex:
    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.storage_path = self.storage_dir / "dense_index.json"
        self.collection_name = "arignan_chunks"
        self._qdrant_client = self._try_create_qdrant_client()
        if self._qdrant_client is None and not self.storage_path.exists():
            self.storage_path.write_text("[]\n", encoding="utf-8")

    def upsert(self, chunks: list[ChunkRecord]) -> None:
        if self._qdrant_client is not None:
            self._upsert_qdrant(chunks)
            return
        existing = {chunk.chunk_id: chunk for chunk in self.all_chunks()}
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError("chunk embedding must be set before indexing")
            existing[chunk.chunk_id] = chunk
        payload = [chunk.to_dict() for chunk in existing.values()]
        with self.storage_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    def search(self, query_embedding: list[float], limit: int) -> list[RetrievalHit]:
        if self._qdrant_client is not None:
            return self._search_qdrant(query_embedding, limit)
        hits: list[RetrievalHit] = []
        for chunk in self.all_chunks():
            if chunk.embedding is None:
                continue
            score = cosine_similarity(query_embedding, chunk.embedding)
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=score,
                    source=RetrievalSource.DENSE,
                    metadata=chunk.metadata,
                )
            )
        return sorted(hits, key=lambda item: item.score, reverse=True)[:limit]

    def delete_load(self, load_id: str) -> None:
        if self._qdrant_client is not None:
            self._delete_load_qdrant(load_id)
            return
        remaining = [chunk for chunk in self.all_chunks() if chunk.metadata.load_id != load_id]
        with self.storage_path.open("w", encoding="utf-8") as handle:
            json.dump([chunk.to_dict() for chunk in remaining], handle, indent=2)
            handle.write("\n")

    def all_chunks(self) -> list[ChunkRecord]:
        if self._qdrant_client is not None:
            return self._all_chunks_qdrant()
        with self.storage_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return [ChunkRecord.from_dict(item) for item in payload]

    def _try_create_qdrant_client(self):
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            return None
        return QdrantClient(path=str(self.storage_dir))

    def _upsert_qdrant(self, chunks: list[ChunkRecord]) -> None:
        from qdrant_client.http import models

        if not chunks:
            return
        first_embedding = chunks[0].embedding
        if first_embedding is None:
            raise ValueError("chunk embedding must be set before indexing")
        if not self._qdrant_client.collection_exists(self.collection_name):
            self._qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=len(first_embedding),
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(),
                ),
            )

        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError("chunk embedding must be set before indexing")
            points.append(
                models.PointStruct(
                    id=_qdrant_id(chunk.chunk_id),
                    vector=chunk.embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "metadata": chunk.metadata.to_dict(),
                    },
                )
            )
        self._qdrant_client.upsert(collection_name=self.collection_name, points=points)

    def _search_qdrant(self, query_embedding: list[float], limit: int) -> list[RetrievalHit]:
        if not self._qdrant_client.collection_exists(self.collection_name):
            return []
        results = self._qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True,
        )
        hits: list[RetrievalHit] = []
        for item in results:
            payload = item.payload or {}
            hits.append(
                RetrievalHit(
                    chunk_id=str(payload["chunk_id"]),
                    text=str(payload["text"]),
                    score=float(item.score),
                    source=RetrievalSource.DENSE,
                    metadata=ChunkMetadata.from_dict(payload["metadata"]),
                )
            )
        return hits

    def _delete_load_qdrant(self, load_id: str) -> None:
        from qdrant_client.http import models

        if not self._qdrant_client.collection_exists(self.collection_name):
            return
        self._qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.load_id",
                            match=models.MatchValue(value=load_id),
                        )
                    ]
                )
            ),
        )

    def _all_chunks_qdrant(self) -> list[ChunkRecord]:
        if not self._qdrant_client.collection_exists(self.collection_name):
            return []
        points, _ = self._qdrant_client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            with_vectors=True,
            limit=10_000,
        )
        records: list[ChunkRecord] = []
        for item in points:
            payload = item.payload or {}
            records.append(
                ChunkRecord.from_dict(
                    {
                        "chunk_id": payload["chunk_id"],
                        "text": payload["text"],
                        "metadata": payload["metadata"],
                        "embedding": list(item.vector) if item.vector is not None else None,
                    }
                )
            )
        return records


class DenseIndexer:
    def __init__(self, embedder: Embedder, index: DenseIndex) -> None:
        self.embedder = embedder
        self.index = index

    def index_chunks(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        embeddings = self.embedder.embed_texts([chunk.text for chunk in chunks])
        embedded_chunks = [
            replace(chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self.index.upsert(embedded_chunks)
        return embedded_chunks

    def search(self, query: str, limit: int) -> list[RetrievalHit]:
        query_embedding = self.embedder.embed_query(query)
        return self.index.search(query_embedding, limit)

    def delete_load(self, load_id: str) -> None:
        self.index.delete_load(load_id)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        raise ValueError("embeddings must be non-empty and have the same dimension")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _qdrant_id(chunk_id: str) -> int:
    return int(hashlib.sha1(chunk_id.encode("utf-8")).hexdigest()[:15], 16)
