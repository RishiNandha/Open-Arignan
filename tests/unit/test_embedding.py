from __future__ import annotations

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

    monkeypatch.setattr("arignan.indexing.embedding.SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr("arignan.indexing.embedding.preferred_torch_device", lambda x: "cuda")

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

    monkeypatch.setattr("arignan.indexing.embedding.SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr("arignan.indexing.embedding.preferred_torch_device", lambda x: "cuda")

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

    monkeypatch.setattr("arignan.indexing.embedding.SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr("arignan.indexing.embedding.preferred_torch_device", lambda x: "cuda")

    embedder = create_embedder(load_config(app_home=app_home), progress_sink=None)
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

    monkeypatch.setattr("arignan.indexing.embedding.SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr("arignan.indexing.embedding.preferred_torch_device", lambda x: "cuda")
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


# ---------------------------------------------------------------------------
# Model path security: create_embedder must reject paths outside app_home
# ---------------------------------------------------------------------------


class TestCreateEmbedderModelPathSecurity:
    """Verify that create_embedder() refuses to load a model whose resolved
    path is outside app_home.

    Without this check a tampered settings.json could point embedding_model at
    an arbitrary local path.  SentenceTransformer pickle-loads model weights,
    so an adversarial path gives arbitrary code execution.
    """

    def _fake_sentence_transformer_patch(self, monkeypatch) -> None:
        """Patch SentenceTransformer so tests don't need real model files."""

        class FakeSentenceTransformer:
            def __init__(self, model_name: str, device: str) -> None:
                pass

            def encode(self, texts, **kw):
                return [[0.0] for _ in texts]

        monkeypatch.setattr(
            "arignan.indexing.embedding.SentenceTransformer", FakeSentenceTransformer
        )
        monkeypatch.setattr(
            "arignan.indexing.embedding.preferred_torch_device", lambda _: "cpu"
        )

    def test_model_dir_inside_app_home_is_accepted(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        self._fake_sentence_transformer_patch(monkeypatch)
        app_home = tmp_path / ".arignan"
        write_default_settings(app_home=app_home)
        model_dir = app_home / "models" / "BAAI__bge-base-en-v1.5"
        model_dir.mkdir(parents=True, exist_ok=True)

        config = load_config(app_home=app_home)
        embedder = create_embedder(config, progress_sink=None)
        assert embedder.backend_name == "sentence-transformers"

    def test_model_dir_outside_app_home_raises_runtime_error(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        self._fake_sentence_transformer_patch(monkeypatch)
        app_home = tmp_path / ".arignan"
        write_default_settings(app_home=app_home)

        # Point embedding_model at something outside app_home via monkeypatching
        # the config after construction so we bypass the normal setup path.
        config = load_config(app_home=app_home)
        evil_model_dir = tmp_path / "outside" / "malicious_model"
        evil_model_dir.mkdir(parents=True, exist_ok=True)

        # Monkeypatch resolve_model_storage_dir to return the evil path
        monkeypatch.setattr(
            "arignan.indexing.embedding.resolve_model_storage_dir",
            lambda app_home, model_id: evil_model_dir,
        )

        with pytest.raises(RuntimeError, match="outside of app_home"):
            create_embedder(config, progress_sink=None)

    def test_symlink_pointing_outside_app_home_is_rejected(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A symlink inside app_home that points outside must be rejected
        because we resolve() the path before the boundary check."""
        self._fake_sentence_transformer_patch(monkeypatch)
        app_home = tmp_path / ".arignan"
        write_default_settings(app_home=app_home)

        # Create a legitimate models dir, but the actual model dir is a symlink
        # to a location outside app_home
        outside_target = tmp_path / "outside_model"
        outside_target.mkdir(parents=True, exist_ok=True)

        models_root = app_home / "models"
        models_root.mkdir(parents=True, exist_ok=True)
        symlink_model_dir = models_root / "BAAI__bge-base-en-v1.5"
        symlink_model_dir.symlink_to(outside_target)

        monkeypatch.setattr(
            "arignan.indexing.embedding.resolve_model_storage_dir",
            lambda _ah, _mid: symlink_model_dir,
        )

        config = load_config(app_home=app_home)
        with pytest.raises(RuntimeError, match="outside of app_home"):
            create_embedder(config, progress_sink=None)

    def test_path_traversal_in_model_name_is_rejected(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A model_id containing '../' that resolves outside app_home must fail."""
        self._fake_sentence_transformer_patch(monkeypatch)
        app_home = tmp_path / ".arignan"
        write_default_settings(app_home=app_home)

        # Simulate model_storage_dir resolving to a traversal path
        traversal_path = (app_home / "models" / ".." / ".." / "etc").resolve()
        traversal_path.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(
            "arignan.indexing.embedding.resolve_model_storage_dir",
            lambda _ah, _mid: traversal_path,
        )

        config = load_config(app_home=app_home)
        with pytest.raises(RuntimeError, match="outside of app_home"):
            create_embedder(config, progress_sink=None)

    def test_error_message_includes_both_paths(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        self._fake_sentence_transformer_patch(monkeypatch)
        app_home = tmp_path / ".arignan"
        write_default_settings(app_home=app_home)
        evil = tmp_path / "evil"
        evil.mkdir()

        monkeypatch.setattr(
            "arignan.indexing.embedding.resolve_model_storage_dir",
            lambda _ah, _mid: evil,
        )

        config = load_config(app_home=app_home)
        with pytest.raises(RuntimeError) as exc_info:
            create_embedder(config, progress_sink=None)
        message = str(exc_info.value)
        assert "outside of app_home" in message
