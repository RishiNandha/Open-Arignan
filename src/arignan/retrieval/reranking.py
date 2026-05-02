from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from arignan.compute import (
    _report_best_effort_exception,
    format_torch_cuda_memory,
    preferred_torch_device,
    release_torch_cuda_memory,
)
from arignan.indexing import tokenize
from arignan.model_registry import DEFAULT_RERANKER_MODEL_REPO_ID, resolve_model_storage_dir
from arignan.models import RetrievalHit

if False:  # pragma: no cover
    from arignan.config import AppConfig
    from arignan.session import SessionExceptionLogger

DEFAULT_RERANKER_MODEL = DEFAULT_RERANKER_MODEL_REPO_ID


class Reranker(Protocol):
    model_name: str
    backend_name: str

    def rerank(self, query: str, hits: list[RetrievalHit], limit: int, min_score: float = 0.0) -> list[RetrievalHit]:
        """Rerank candidate retrieval hits."""


class HeuristicReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.model_name = model_name
        self.backend_name = "heuristic-reranker"

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
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, model_source: str | Path | None = None) -> None:
        self.model_name = model_name
        self.backend_name = "cross-encoder"
        self.device = preferred_torch_device()
        self.model_source = str(model_source or model_name)
        self._model = None

    def rerank(self, query: str, hits: list[RetrievalHit], limit: int, min_score: float = 0.0) -> list[RetrievalHit]:
        if not hits or limit <= 0:
            return []
        pairs = [[query, hit.text] for hit in hits]
        scores = self._ensure_model().predict(pairs)
        rescored: list[tuple[float, RetrievalHit]] = []
        for hit, score in zip(hits, scores):
            numeric_score = float(score)
            hit.extras["rerank_score"] = numeric_score
            if numeric_score >= min_score:
                rescored.append((numeric_score, hit))
        rescored.sort(key=lambda item: item[0], reverse=True)
        return [hit for _, hit in rescored[:limit]]

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - exercised only when dependency is installed
            raise RuntimeError(
                "sentence-transformers is required for CrossEncoderReranker; "
                "install the optional ml dependencies"
            ) from exc
        self._model = CrossEncoder(self.model_source, device=self.device)
        return self._model

    def release_device_memory(self) -> bool:
        model = self._model
        self._model = None
        if model is None:
            return False
        if self.device == "cuda" and hasattr(model, "model") and hasattr(model.model, "cpu"):
            try:
                model.model.cpu()
            except Exception as exc:  # pragma: no cover - depends on local runtime
                _report_best_effort_exception("reranker model GPU offload", exc)
        del model
        release_torch_cuda_memory()
        return True


def create_reranker(
    config: "AppConfig",
    *,
    progress_sink: Callable[[str], None] | None = None,
    exception_logger: "SessionExceptionLogger | None" = None,
) -> Reranker:
    model_dir = resolve_model_storage_dir(config.app_home, config.reranker_model)
    if progress_sink is not None:
        progress_sink(f"Preparing local reranker model ({config.reranker_model})...")
    if not model_dir.exists():
        raise RuntimeError(_missing_reranker_model_error(config.reranker_model, model_dir))
    try:
        reranker = CrossEncoderReranker(model_name=config.reranker_model, model_source=model_dir)
        reranker._ensure_model()
        if progress_sink is not None and reranker.device == "cuda":
            message = format_torch_cuda_memory(f"GPU after reranker load ({config.reranker_model})")
            if message:
                progress_sink(message)
        return reranker
    except Exception as exc:
        log_path = None
        if exception_logger is not None:
            log_path = exception_logger.log_exception(
                component="reranker",
                task="reranker model load",
                exc=exc,
                context={"model_name": config.reranker_model, "model_source": str(model_dir)},
            )
        raise RuntimeError(_reranker_runtime_error(config.reranker_model, model_dir, log_path=log_path)) from exc


def _reranker_runtime_error(model_name: str, model_dir: Path, *, log_path: Path | None = None) -> str:
    message = [
        "Arignan requires the Python retrieval ML stack for reranking; heuristic fallback is disabled.",
        f"Configured reranker model: {model_name}",
        f"Expected local model directory: {model_dir}",
        "Required packages in this Python environment: "
        "transformers>=4.48,<4.50, accelerate>=0.30,<1, sentence-transformers>=3.0,<4",
        "Arignan will not auto-install or change your existing Torch/CUDA setup.",
        "If the model files are missing, rerun `python setup.py --app-home <your app home>`.",
    ]
    if log_path is not None:
        message.append(f"See exception log: {log_path.resolve()}")
    return " ".join(message)


def _missing_reranker_model_error(model_name: str, model_dir: Path) -> str:
    message = [
        "Arignan could not find the cached reranker model files on disk.",
        f"Configured reranker model: {model_name}",
        f"Expected local model directory: {model_dir}",
        "This is a model-cache problem, not proof that your Python packages are missing.",
        "If you already installed the required Python packages, rerun `python setup.py --app-home <your app home>` to download the retrieval models into the app home.",
        "Arignan will not auto-install or change your existing Torch/CUDA setup.",
    ]
    return " ".join(message)
