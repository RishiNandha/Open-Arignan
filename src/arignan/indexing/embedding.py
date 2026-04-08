from __future__ import annotations

import hashlib
import math
from typing import Protocol

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


class Embedder(Protocol):
    model_name: str
    backend_name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""


class HashingEmbedder:
    def __init__(self, dimension: int = 24) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.model_name = DEFAULT_EMBEDDING_MODEL
        self.backend_name = "hashing-embedder"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = text.lower().split()
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(self.dimension):
                raw = digest[index % len(digest)]
                vector[index] += (raw / 255.0) * 2.0 - 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.backend_name = "sentence-transformers"
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only when dependency is installed
            raise RuntimeError(
                "sentence-transformers is required for SentenceTransformerEmbedder; "
                "install the optional ml dependencies"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        encoded = self._model.encode(texts, normalize_embeddings=True)
        return [list(vector) for vector in encoded]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
