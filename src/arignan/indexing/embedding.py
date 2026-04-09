from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from arignan.compute import preferred_torch_device
from arignan.model_registry import DEFAULT_EMBEDDING_MODEL_REPO_ID, resolve_model_storage_dir

if False:  # pragma: no cover
    from arignan.config import AppConfig
    from arignan.session import SessionExceptionLogger

DEFAULT_EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL_REPO_ID


class Embedder(Protocol):
    model_name: str
    backend_name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""


class HashingEmbedder:
    def __init__(self, dimension: int = 24, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.model_name = model_name
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
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, model_source: str | Path | None = None) -> None:
        self.model_name = model_name
        self.backend_name = "sentence-transformers"
        self.device = preferred_torch_device()
        self.model_source = str(model_source or model_name)
        self._model = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        encoded = model.encode(texts, normalize_embeddings=True)
        return [list(vector) for vector in encoded]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only when dependency is installed
            raise RuntimeError(
                "sentence-transformers is required for SentenceTransformerEmbedder; "
                "install the optional ml dependencies"
            ) from exc
        self._model = SentenceTransformer(self.model_source, device=self.device)
        return self._model


def create_embedder(
    config: "AppConfig",
    *,
    progress_sink: Callable[[str], None] | None = None,
    exception_logger: "SessionExceptionLogger | None" = None,
) -> Embedder:
    model_dir = resolve_model_storage_dir(config.app_home, config.embedding_model)
    if not model_dir.exists():
        return HashingEmbedder(model_name=config.embedding_model)
    try:
        if progress_sink is not None:
            progress_sink(f"Preparing local embedding model ({config.embedding_model})...")
        return SentenceTransformerEmbedder(model_name=config.embedding_model, model_source=model_dir)
    except Exception as exc:
        if exception_logger is not None:
            log_path = exception_logger.log_exception(
                component="embedder",
                task="embedding model load",
                exc=exc,
                context={"model_name": config.embedding_model, "model_source": str(model_dir)},
            )
            if progress_sink is not None:
                progress_sink(
                    f"Local embedding model unavailable; using hashing fallback. Log: {log_path.resolve()}"
                )
        return HashingEmbedder(model_name=config.embedding_model)
