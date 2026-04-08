from __future__ import annotations

from typing import Protocol

from arignan.indexing import tokenize
from arignan.models import RetrievalHit

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker(Protocol):
    model_name: str

    def rerank(self, query: str, hits: list[RetrievalHit], limit: int, min_score: float = 0.0) -> list[RetrievalHit]:
        """Rerank candidate retrieval hits."""


class HeuristicReranker:
    def __init__(self) -> None:
        self.model_name = DEFAULT_RERANKER_MODEL

    def rerank(self, query: str, hits: list[RetrievalHit], limit: int, min_score: float = 0.0) -> list[RetrievalHit]:
        query_terms = set(tokenize(query))
        rescored: list[tuple[float, RetrievalHit]] = []
        for hit in hits:
            hit_terms = set(tokenize(f"{hit.metadata.heading or ''} {hit.text}"))
            overlap = len(query_terms & hit_terms)
            score = overlap / max(len(query_terms), 1)
            hit.extras["rerank_score"] = score
            if score >= min_score:
                rescored.append((score, hit))
        rescored.sort(key=lambda item: item[0], reverse=True)
        return [hit for _, hit in rescored[:limit]]


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - exercised only when dependency is installed
            raise RuntimeError(
                "sentence-transformers is required for CrossEncoderReranker; "
                "install the optional ml dependencies"
            ) from exc
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[RetrievalHit], limit: int, min_score: float = 0.0) -> list[RetrievalHit]:
        pairs = [[query, hit.text] for hit in hits]
        scores = self._model.predict(pairs)
        rescored: list[tuple[float, RetrievalHit]] = []
        for hit, score in zip(hits, scores):
            numeric_score = float(score)
            hit.extras["rerank_score"] = numeric_score
            if numeric_score >= min_score:
                rescored.append((numeric_score, hit))
        rescored.sort(key=lambda item: item[0], reverse=True)
        return [hit for _, hit in rescored[:limit]]
