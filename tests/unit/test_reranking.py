from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from arignan.config import load_config, write_default_settings
from arignan.models import ChunkMetadata, RetrievalHit, RetrievalSource
from arignan.retrieval import DEFAULT_RERANKER_MODEL, CrossEncoderReranker, HeuristicReranker, create_reranker


def test_heuristic_reranker_reorders_hits_by_query_overlap() -> None:
    reranker = HeuristicReranker()
    hits = [
        RetrievalHit(
            chunk_id="unrelated",
            text="Therapy notes and cognitive biases",
            score=0.9,
            source=RetrievalSource.LEXICAL,
            metadata=ChunkMetadata(load_id="load-1", hat="default", source_uri="psych.md"),
        ),
        RetrievalHit(
            chunk_id="related",
            text="Joint embedding predictive architecture notes",
            score=0.4,
            source=RetrievalSource.DENSE,
            metadata=ChunkMetadata(load_id="load-2", hat="default", source_uri="jepa.md"),
        ),
    ]

    reranked = reranker.rerank("joint embedding architecture", hits, limit=2)

    assert reranker.model_name == DEFAULT_RERANKER_MODEL
    assert reranked[0].chunk_id == "related"


def test_heuristic_reranker_prunes_below_threshold() -> None:
    reranker = HeuristicReranker()
    hits = [
        RetrievalHit(
            chunk_id="keep",
            text="retrieval augmented generation pipeline",
            score=0.3,
            source=RetrievalSource.DENSE,
            metadata=ChunkMetadata(load_id="load-1", hat="default", source_uri="rag.md"),
        ),
        RetrievalHit(
            chunk_id="drop",
            text="fresh mango smoothie recipe",
            score=0.8,
            source=RetrievalSource.DENSE,
            metadata=ChunkMetadata(load_id="load-2", hat="default", source_uri="food.md"),
        ),
    ]

    reranked = reranker.rerank("retrieval pipeline", hits, limit=5, min_score=0.4)

    assert [hit.chunk_id for hit in reranked] == ["keep"]


def test_cross_encoder_reranker_prefers_cuda_when_available(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeCrossEncoder:
        def __init__(self, model_name: str, device: str) -> None:
            captured["model_name"] = model_name
            captured["device"] = device

    fake_module = types.SimpleNamespace(CrossEncoder=FakeCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr("arignan.retrieval.reranking.preferred_torch_device", lambda: "cuda")

    reranker = CrossEncoderReranker()
    reranker._ensure_model()

    assert reranker.device == "cuda"
    assert captured == {"model_name": DEFAULT_RERANKER_MODEL, "device": "cuda"}


def test_create_reranker_uses_cross_encoder_when_model_cached(tmp_path: Path, monkeypatch) -> None:
    app_home = tmp_path / ".arignan"
    write_default_settings(app_home=app_home)
    model_dir = app_home / "models" / "mixedbread-ai__mxbai-rerank-base-v1"
    model_dir.mkdir(parents=True, exist_ok=True)

    captured: dict[str, object] = {}

    class FakeCrossEncoder:
        def __init__(self, model_name: str, device: str) -> None:
            captured["model_name"] = model_name
            captured["device"] = device

        def predict(self, pairs):
            return [0.9 for _ in pairs]

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=FakeCrossEncoder))
    monkeypatch.setattr("arignan.retrieval.reranking.preferred_torch_device", lambda: "cuda")

    reranker = create_reranker(load_config(app_home=app_home))
    reranker.rerank(
        "question",
        [
            RetrievalHit(
                chunk_id="c1",
                text="Joint embedding predictive architecture",
                score=0.5,
                source=RetrievalSource.DENSE,
                metadata=ChunkMetadata(load_id="load-1", hat="default", source_uri="jepa.md"),
            )
        ],
        limit=1,
    )

    assert reranker.backend_name == "cross-encoder"
    assert captured == {"model_name": str(model_dir), "device": "cuda"}


def test_create_reranker_emits_gpu_telemetry_when_loaded_on_cuda(tmp_path: Path, monkeypatch) -> None:
    app_home = tmp_path / ".arignan"
    write_default_settings(app_home=app_home)
    model_dir = app_home / "models" / "mixedbread-ai__mxbai-rerank-base-v1"
    model_dir.mkdir(parents=True, exist_ok=True)
    progress: list[str] = []

    class FakeCrossEncoder:
        def __init__(self, model_name: str, device: str) -> None:
            return None

        def predict(self, pairs):
            return [0.9 for _ in pairs]

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=FakeCrossEncoder))
    monkeypatch.setattr("arignan.retrieval.reranking.preferred_torch_device", lambda: "cuda")
    monkeypatch.setattr(
        "arignan.retrieval.reranking.format_torch_cuda_memory",
        lambda label: f"{label}: torch cuda allocated=0.70 GiB, reserved=0.90 GiB, total=4.00 GiB",
    )

    create_reranker(load_config(app_home=app_home), progress_sink=progress.append)

    assert any("GPU after reranker load" in message for message in progress)


def test_create_reranker_requires_local_ml_runtime_when_model_missing(tmp_path: Path) -> None:
    app_home = tmp_path / ".arignan"
    write_default_settings(app_home=app_home)

    with pytest.raises(RuntimeError) as exc_info:
        create_reranker(load_config(app_home=app_home))

    message = str(exc_info.value)
    assert "could not find the cached reranker model files on disk" in message
    assert "model-cache problem" in message
    assert "Arignan will not auto-install or change your existing Torch/CUDA setup." in message


def test_cross_encoder_reranker_returns_empty_for_empty_hits(monkeypatch) -> None:
    class FakeCrossEncoder:
        def __init__(self, model_name: str, device: str) -> None:
            raise AssertionError("model should not load for empty hit lists")

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=FakeCrossEncoder))

    reranker = CrossEncoderReranker()

    assert reranker.rerank("question", [], limit=3) == []
