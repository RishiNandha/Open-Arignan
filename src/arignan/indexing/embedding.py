from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from arignan.compute import format_torch_cuda_memory, preferred_torch_device, release_torch_cuda_memory
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
        encoded = self._encode(texts, is_query=False)
        return [list(vector) for vector in encoded]

    def embed_query(self, text: str) -> list[float]:
        return list(self._encode([text], is_query=True)[0])

    def _encode(self, texts: list[str], *, is_query: bool):
        model = self._ensure_model()
        encode_kwargs = {"normalize_embeddings": True}
        query_prompt = _query_prompt_for_model(self.model_name)
        if is_query and query_prompt:
            encode_kwargs["prompt"] = query_prompt
        try:
            return model.encode(texts, **encode_kwargs)
        except TypeError:
            encode_kwargs.pop("prompt", None)
            return model.encode(texts, **encode_kwargs)

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

    def release_device_memory(self) -> bool:
        model = self._model
        self._model = None
        if model is None:
            return False
        if self.device == "cuda" and hasattr(model, "cpu"):
            try:
                model.cpu()
            except Exception:  # pragma: no cover - depends on local runtime
                pass
        del model
        release_torch_cuda_memory()
        return True


def _query_prompt_for_model(model_name: str) -> str | None:
    normalized = model_name.lower()
    if "bge-" in normalized or "/bge-" in normalized:
        return "Represent this sentence for searching relevant passages: "
    return None


def create_embedder(
    config: "AppConfig",
    *,
    progress_sink: Callable[[str], None] | None = None,
    exception_logger: "SessionExceptionLogger | None" = None,
) -> Embedder:
    model_dir = resolve_model_storage_dir(config.app_home, config.embedding_model)
    if progress_sink is not None:
        progress_sink(f"Preparing local embedding model ({config.embedding_model})...")
    if not model_dir.exists():
        raise RuntimeError(_missing_embedder_model_error(config.embedding_model, model_dir))
    try:
        embedder = SentenceTransformerEmbedder(model_name=config.embedding_model, model_source=model_dir)
        embedder._ensure_model()
        if progress_sink is not None and embedder.device == "cuda":
            message = format_torch_cuda_memory(f"GPU after embedding model load ({config.embedding_model})")
            if message:
                progress_sink(message)
        return embedder
    except Exception as exc:
        log_path = None
        if exception_logger is not None:
            log_path = exception_logger.log_exception(
                component="embedder",
                task="embedding model load",
                exc=exc,
                context={"model_name": config.embedding_model, "model_source": str(model_dir)},
            )
        raise RuntimeError(_embedder_runtime_error(config.embedding_model, model_dir, log_path=log_path)) from exc


def _embedder_runtime_error(model_name: str, model_dir: Path, *, log_path: Path | None = None) -> str:
    message = [
        "Arignan requires the Python retrieval ML stack for embeddings; hashing fallback is disabled.",
        f"Configured embedding model: {model_name}",
        f"Expected local model directory: {model_dir}",
        "Required packages in this Python environment: "
        "transformers>=4.48,<4.50, accelerate>=0.30,<1, sentence-transformers>=3.0,<4",
        "Arignan will not auto-install or change your existing Torch/CUDA setup.",
        "If the model files are missing, rerun `python setup.py --app-home <your app home>`.",
    ]
    if log_path is not None:
        message.append(f"See exception log: {log_path.resolve()}")
    return " ".join(message)


def _missing_embedder_model_error(model_name: str, model_dir: Path) -> str:
    message = [
        "Arignan could not find the cached embedding model files on disk.",
        f"Configured embedding model: {model_name}",
        f"Expected local model directory: {model_dir}",
        "This is a model-cache problem, not proof that your Python packages are missing.",
        "If you already installed the required Python packages, rerun `python setup.py --app-home <your app home>` to download the retrieval models into the app home.",
        "Arignan will not auto-install or change your existing Torch/CUDA setup.",
    ]
    return " ".join(message)
