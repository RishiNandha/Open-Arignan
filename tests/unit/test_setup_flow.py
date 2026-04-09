from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from arignan.config import write_default_settings
from arignan.paths import read_persisted_app_home
from arignan.setup_flow import (
    SetupResult,
    _emit,
    _display_path,
    create_launchers,
    download_required_models,
    initialize_local_state,
    install_package,
    install_target,
    render_summary,
    resolve_ollama_model_id,
    resolve_model_repo_id,
    sanitize_model_id,
)


def test_install_target_switches_for_dev() -> None:
    assert install_target(dev=False) == "."
    assert install_target(dev=True) == ".[dev]"


def test_install_package_uses_standard_pip_build_isolation(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[list[str], Path, bool]] = []

    def fake_run(command, cwd=None, check=None):
        calls.append((list(command), Path(cwd), bool(check)))
        return None

    monkeypatch.setattr("arignan.setup_flow.subprocess.run", fake_run)
    monkeypatch.setattr(sys, "executable", str((tmp_path / "python.exe").resolve()))

    target = install_package(root=tmp_path, dev=True)

    assert target == ".[dev]"
    assert calls == [
        ([str((tmp_path / "python.exe").resolve()), "-m", "pip", "install", ".[dev]"], tmp_path.resolve(), True)
    ]


def test_create_launchers_writes_bin_scripts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "executable", str((tmp_path / "python.exe").resolve()))

    bin_dir, windows_launcher, posix_launcher = create_launchers(root=tmp_path)

    assert bin_dir == tmp_path / "bin"
    assert windows_launcher.exists()
    assert posix_launcher.exists()
    assert "-m arignan.cli" in windows_launcher.read_text(encoding="utf-8")
    assert "-m arignan.cli" in posix_launcher.read_text(encoding="utf-8")
    assert "--app-home" not in windows_launcher.read_text(encoding="utf-8")


def test_create_launchers_can_pin_app_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "executable", str((tmp_path / "python.exe").resolve()))
    app_home = (tmp_path / "custom-home").resolve()

    _, windows_launcher, posix_launcher = create_launchers(root=tmp_path, app_home=app_home)

    assert f'--app-home "{app_home}"' in windows_launcher.read_text(encoding="utf-8")
    assert f"--app-home '{app_home}'" in posix_launcher.read_text(encoding="utf-8")


def test_render_summary_mentions_next_steps(tmp_path: Path) -> None:
    result = SetupResult(
        install_target=".",
        app_home=tmp_path / ".arignan",
        settings_path=tmp_path / ".arignan" / "settings.json",
        models_dir=tmp_path / ".arignan" / "models",
        local_llm_backend="ollama",
        local_llm_model="qwen3:4b-q4_K_M",
        local_llm_light_model="qwen3:0.6b",
        embedding_model="BAAI/bge-base-en-v1.5",
        reranker_model="mixedbread-ai/mxbai-rerank-base-v1",
        bin_dir=tmp_path / "bin",
        windows_launcher=tmp_path / "bin" / "arignan.cmd",
        posix_launcher=tmp_path / "bin" / "arignan",
    )

    summary = render_summary(result)

    assert "Arignan setup complete." in summary
    assert "Next steps:" in summary
    assert str(result.bin_dir) in summary


def test_display_path_strips_windows_extended_prefix() -> None:
    assert _display_path(r"\\?\D:\Code\Open Arignan\bin") == r"D:\Code\Open Arignan\bin"
    assert _display_path(r"\\?\UNC\server\share\folder") == r"\\server\share\folder"


def test_sanitize_model_id_normalizes_paths() -> None:
    assert sanitize_model_id("BAAI/bge-base-en-v1.5") == "BAAI__bge-base-en-v1.5"


def test_resolve_model_repo_id_maps_readme_default() -> None:
    assert resolve_model_repo_id("Qwen3-1.7B") == "Qwen/Qwen3-1.7B"
    assert resolve_model_repo_id("Qwen/Qwen3-1.7B") == "Qwen/Qwen3-1.7B"
    assert resolve_model_repo_id("BAAI/bge-base-en-v1.5") == "BAAI/bge-base-en-v1.5"


def test_initialize_local_state_can_override_local_llm_model(tmp_path: Path) -> None:
    original_home = Path.home
    Path.home = staticmethod(lambda: tmp_path)
    try:
        app_home, settings_path = initialize_local_state(
            app_home=tmp_path / ".arignan",
            local_llm_backend="ollama",
            local_llm_model="qwen3:4b-q4_K_M",
        )

        payload = json.loads(settings_path.read_text(encoding="utf-8"))

        assert app_home == (tmp_path / ".arignan").resolve()
        assert payload["local_llm_backend"] == "ollama"
        assert payload["local_llm_model"] == "qwen3:4b-q4_K_M"
        assert payload["local_llm_light_model"] == "qwen3:0.6b"
        assert payload["embedding_model"] == "BAAI/bge-base-en-v1.5"
        assert payload["reranker_model"] == "mixedbread-ai/mxbai-rerank-base-v1"
        assert read_persisted_app_home() == (tmp_path / ".arignan").resolve()
    finally:
        Path.home = original_home


def test_initialize_local_state_can_pin_lightweight_full_model(tmp_path: Path) -> None:
    original_home = Path.home
    Path.home = staticmethod(lambda: tmp_path)
    try:
        _, settings_path = initialize_local_state(
            app_home=tmp_path / ".arignan",
            local_llm_backend="ollama",
            local_llm_model="qwen3:0.6b",
            local_llm_light_model="qwen3:0.6b",
            embedding_model="BAAI/bge-small-en-v1.5",
            reranker_model="mixedbread-ai/mxbai-rerank-xsmall-v1",
        )

        payload = json.loads(settings_path.read_text(encoding="utf-8"))

        assert payload["local_llm_model"] == "qwen3:0.6b"
        assert payload["local_llm_light_model"] == "qwen3:0.6b"
        assert payload["embedding_model"] == "BAAI/bge-small-en-v1.5"
        assert payload["reranker_model"] == "mixedbread-ai/mxbai-rerank-xsmall-v1"
    finally:
        Path.home = original_home


def test_initialize_local_state_refreshes_existing_settings_to_current_defaults(tmp_path: Path) -> None:
    original_home = Path.home
    Path.home = staticmethod(lambda: tmp_path)
    try:
        app_home = tmp_path / ".arignan"
        settings_path = write_default_settings(app_home=app_home)
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
        payload["local_llm_backend"] = "transformers"
        payload["local_llm_model"] = "some-custom-old-model"
        payload["local_llm_light_model"] = "another-old-model"
        payload["embedding_model"] = "legacy-embedding"
        payload["reranker_model"] = "legacy-reranker"
        settings_path.write_text(json.dumps(payload), encoding="utf-8")

        _, refreshed_settings_path = initialize_local_state(app_home=app_home)

        refreshed = json.loads(refreshed_settings_path.read_text(encoding="utf-8"))
        assert refreshed["local_llm_backend"] == "ollama"
        assert refreshed["local_llm_model"] == "qwen3:4b-q4_K_M"
        assert refreshed["local_llm_light_model"] == "qwen3:0.6b"
        assert refreshed["embedding_model"] == "BAAI/bge-base-en-v1.5"
        assert refreshed["reranker_model"] == "mixedbread-ai/mxbai-rerank-base-v1"
    finally:
        Path.home = original_home


def test_download_required_models_pulls_default_ollama_model(tmp_path: Path, monkeypatch) -> None:
    app_home = tmp_path / ".arignan"
    write_default_settings(app_home=app_home)
    provisioned: list[Path] = []
    ensured: list[tuple[Path, str, str, float]] = []

    def fake_provision(app_home_arg: Path, progress=None) -> Path:
        provisioned.append(app_home_arg)
        executable = app_home_arg / "runtime" / "local_llm" / "ollama.exe"
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text("", encoding="utf-8")
        return executable

    def fake_ensure(app_home_arg: Path, endpoint: str, model: str, progress=None, timeout_seconds: float = 1800.0) -> None:
        ensured.append((app_home_arg, endpoint, model, timeout_seconds))

    downloaded_transformer_models: list[tuple[str, Path, bool]] = []

    class FakeHubModule:
        @staticmethod
        def snapshot_download(*, repo_id: str, local_dir: Path, local_dir_use_symlinks: bool, ignore_patterns=None) -> None:
            downloaded_transformer_models.append((repo_id, local_dir, local_dir_use_symlinks))
            local_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHubModule())

    monkeypatch.setattr("arignan.setup_flow.provision_managed_runtime", fake_provision)
    monkeypatch.setattr("arignan.setup_flow.ensure_model_available", fake_ensure)

    models_dir = download_required_models(app_home)

    assert models_dir == app_home / "models"
    assert provisioned == [app_home]
    assert ensured == [
        (app_home, "http://127.0.0.1:11434", "qwen3:4b-q4_K_M", 1800.0),
        (app_home, "http://127.0.0.1:11434", "qwen3:0.6b", 1800.0),
    ]
    manifest = json.loads((models_dir / "local_llm_manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "local_llm_backend": "ollama",
        "local_llm_model": "qwen3:4b-q4_K_M",
        "local_llm_light_model": "qwen3:0.6b",
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "reranker_model": "mixedbread-ai/mxbai-rerank-base-v1",
    }
    assert [repo_id for repo_id, _, _ in downloaded_transformer_models] == [
        "BAAI/bge-base-en-v1.5",
        "mixedbread-ai/mxbai-rerank-base-v1",
    ]


def test_download_required_models_surfaces_managed_runtime_provision_error(tmp_path: Path, monkeypatch) -> None:
    app_home = tmp_path / ".arignan"
    write_default_settings(app_home=app_home)

    def fake_provision(app_home_arg: Path, progress=None) -> Path:
        raise RuntimeError("local model runtime failed to install")

    monkeypatch.setattr("arignan.setup_flow.provision_managed_runtime", fake_provision)

    with pytest.raises(RuntimeError) as exc_info:
        download_required_models(app_home)

    assert "local model runtime failed to install" in str(exc_info.value)


def test_download_required_models_supports_transformers_backend(tmp_path: Path, monkeypatch) -> None:
    app_home = tmp_path / ".arignan"
    settings_path = write_default_settings(app_home=app_home)
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    payload["local_llm_backend"] = "transformers"
    payload["local_llm_model"] = "Qwen3-1.7B"
    settings_path.write_text(json.dumps(payload), encoding="utf-8")

    class FakeRepositoryNotFoundError(Exception):
        pass

    class FakeHubModule:
        @staticmethod
        def snapshot_download(*, repo_id: str, local_dir: Path, local_dir_use_symlinks: bool, ignore_patterns=None) -> None:
            raise FakeRepositoryNotFoundError(f"401 for {repo_id}")

    class FakeErrorsModule:
        RepositoryNotFoundError = FakeRepositoryNotFoundError
        GatedRepoError = type("FakeGatedRepoError", (Exception,), {})
        HfHubHTTPError = type("FakeHfHubHTTPError", (Exception,), {})

    monkeypatch.setattr("arignan.setup_flow.provision_managed_runtime", lambda app_home_arg, progress=None: app_home_arg / "runtime" / "local_llm" / "ollama.exe")
    monkeypatch.setattr("arignan.setup_flow.ensure_model_available", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHubModule())
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", FakeErrorsModule)

    with pytest.raises(RuntimeError) as exc_info:
        download_required_models(app_home)

    message = str(exc_info.value)
    assert "Failed to download model 'Qwen3-1.7B'." in message
    assert "Resolved Hugging Face repo: Qwen/Qwen3-1.7B" in message
    assert f"401 for {resolve_model_repo_id('Qwen3-1.7B')}" in message


def test_emit_forwards_progress_messages() -> None:
    messages: list[str] = []

    _emit(messages.append, "[1/4] Installing Python package...")

    assert messages == ["[1/4] Installing Python package..."]
