from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from arignan.models import ChunkRecord, RetrievalHit, RetrievalSource

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


class LexicalIndex:
    def __init__(self, storage_dir: Path, k1: float = 1.5, b: float = 0.75) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.storage_path = self.storage_dir / "bm25_index.json"
        self.k1 = k1
        self.b = b
        if not self.storage_path.exists():
            self.storage_path.write_text("[]\n", encoding="utf-8")

    def upsert(self, chunks: list[ChunkRecord]) -> None:
        existing = {chunk.chunk_id: chunk for chunk in self.all_chunks()}
        for chunk in chunks:
            existing[chunk.chunk_id] = chunk
        with self.storage_path.open("w", encoding="utf-8") as handle:
            json.dump([chunk.to_dict() for chunk in existing.values()], handle, indent=2)
            handle.write("\n")

    def all_chunks(self) -> list[ChunkRecord]:
        with self.storage_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return [ChunkRecord.from_dict(item) for item in payload]

    def search(self, query: str, limit: int) -> list[RetrievalHit]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
        chunks = self.all_chunks()
        if not chunks:
            return []

        tokenized_docs = [tokenize(chunk.text) for chunk in chunks]
        avg_doc_length = sum(len(tokens) for tokens in tokenized_docs) / len(tokenized_docs)
        doc_frequencies = self._document_frequencies(tokenized_docs)

        hits: list[RetrievalHit] = []
        for chunk, doc_tokens in zip(chunks, tokenized_docs):
            score = self._bm25_score(query_terms, doc_tokens, avg_doc_length, len(chunks), doc_frequencies)
            if score <= 0:
                continue
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=score,
                    source=RetrievalSource.LEXICAL,
                    metadata=chunk.metadata,
                )
            )

        return sorted(hits, key=lambda item: item.score, reverse=True)[:limit]

    def delete_load(self, load_id: str) -> None:
        remaining = [chunk for chunk in self.all_chunks() if chunk.metadata.load_id != load_id]
        with self.storage_path.open("w", encoding="utf-8") as handle:
            json.dump([chunk.to_dict() for chunk in remaining], handle, indent=2)
            handle.write("\n")

    def _document_frequencies(self, tokenized_docs: list[list[str]]) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for tokens in tokenized_docs:
            for token in set(tokens):
                frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies

    def _bm25_score(
        self,
        query_terms: list[str],
        doc_tokens: list[str],
        avg_doc_length: float,
        document_count: int,
        doc_frequencies: dict[str, int],
    ) -> float:
        term_counts = Counter(doc_tokens)
        doc_length = max(len(doc_tokens), 1)
        score = 0.0
        for term in query_terms:
            term_frequency = term_counts.get(term, 0)
            if term_frequency == 0:
                continue
            doc_frequency = doc_frequencies.get(term, 0)
            idf = math.log(1 + ((document_count - doc_frequency + 0.5) / (doc_frequency + 0.5)))
            numerator = term_frequency * (self.k1 + 1)
            denominator = term_frequency + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
            score += idf * (numerator / denominator)
        return score


class LexicalIndexer:
    def __init__(self, index: LexicalIndex) -> None:
        self.index = index

    def index_chunks(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        self.index.upsert(chunks)
        return chunks

    def search(self, query: str, limit: int) -> list[RetrievalHit]:
        return self.index.search(query, limit)

    def delete_load(self, load_id: str) -> None:
        self.index.delete_load(load_id)
