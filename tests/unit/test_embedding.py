from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from arignan.config import load_config, write_default_settings
from arignan.indexing import DEFAULT_EMBEDDING_MODEL, HashingEmbedder, cosine_similarity
from arignan.indexing.embedding import SentenceTransformerEmbedder, create_embedder


def test_hashing_embedder_is_deterministic() -> None:
    embedder = HashingEmbedder(dimension=12)

    first = embedder.embed_query("joint embedding predictive architecture")
    second = embedder.embed_query("joint embedding predictive architecture")

    assert first == second
    assert len(first) == 12
    assert embedder.model_name == DEFAULT_EMBEDDING_MODEL


def test_cosine_similarity_prefers_related_texts() -> None:
    embedder = HashingEmbedder(dimension=16)

    query = embedder.embed_query("keyword dense retrieval")
    related = embedder.embed_query("dense retrieval with keyword fusion")
    unrelated = embedder.embed_query("fresh mango smoothie recipe")

    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


def test_sentence_transformer_embedder_prefers_cuda_when_available(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str) -> None:
            captured["model_name"] = model_name
            captured["device"] = device

        def encode(self, texts, normalize_embeddings=True):
            return [[1.0, 0.0] for _ in texts]

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr("arignan.indexing.embedding.preferred_torch_device", lambda: "cuda")

    embedder = SentenceTransformerEmbedder()
    embedder.embed_texts(["hello world"])

    assert embedder.device == "cuda"
    assert captured == {"model_name": DEFAULT_EMBEDDING_MODEL, "device": "cuda"}


def test_sentence_transformer_embedder_uses_query_prompt_for_bge(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str) -> None:
            captured["model_name"] = model_name
            captured["device"] = device

        def encode(self, texts, normalize_embeddings=True, prompt=None):
            captured["prompt"] = prompt
            return [[1.0, 0.0] for _ in texts]

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr("arignan.indexing.embedding.preferred_torch_device", lambda: "cuda")

    embedder = SentenceTransformerEmbedder(model_name="BAAI/bge-small-en-v1.5")
    embedder.embed_query("what is jepa")

    assert captured["prompt"] == "Represent this sentence for searching relevant passages: "


def test_create_embedder_uses_sentence_transformer_when_model_cached(tmp_path: Path, monkeypatch) -> None:
    app_home = tmp_path / ".arignan"
    write_default_settings(app_home=app_home)
    model_dir = app_home / "models" / "BAAI__bge-base-en-v1.5"
    model_dir.mkdir(parents=True, exist_ok=True)

    captured: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str) -> None:
            captured["model_name"] = model_name
            captured["device"] = device

        def encode(self, texts, normalize_embeddings=True):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))
    monkeypatch.setattr("arignan.indexing.embedding.preferred_torch_device", lambda: "cuda")

    embedder = create_embedder(load_config(app_home=app_home))
    embedder.embed_query("test")

    assert embedder.backend_name == "sentence-transformers"
    assert captured == {"model_name": str(model_dir), "device": "cuda"}


def test_create_embedder_emits_gpu_telemetry_when_loaded_on_cuda(tmp_path: Path, monkeypatch) -> None:
    app_home = tmp_path / ".arignan"
    write_default_settings(app_home=app_home)
    model_dir = app_home / "models" / "BAAI__bge-base-en-v1.5"
    model_dir.mkdir(parents=True, exist_ok=True)
    progress: list[str] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str) -> None:
            return None

        def encode(self, texts, normalize_embeddings=True):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))
    monkeypatch.setattr("arignan.indexing.embedding.preferred_torch_device", lambda: "cuda")
    monkeypatch.setattr(
        "arignan.indexing.embedding.format_torch_cuda_memory",
        lambda label: f"{label}: torch cuda allocated=0.40 GiB, reserved=0.50 GiB, total=4.00 GiB",
    )

    create_embedder(load_config(app_home=app_home), progress_sink=progress.append)

    assert any("GPU after embedding model load" in message for message in progress)


def test_create_embedder_requires_local_ml_runtime_when_model_missing(tmp_path: Path) -> None:
    app_home = tmp_path / ".arignan"
    write_default_settings(app_home=app_home)

    with pytest.raises(RuntimeError) as exc_info:
        create_embedder(load_config(app_home=app_home))

    message = str(exc_info.value)
    assert "could not find the cached embedding model files on disk" in message
    assert "model-cache problem" in message
    assert "Arignan will not auto-install or change your existing Torch/CUDA setup." in message
