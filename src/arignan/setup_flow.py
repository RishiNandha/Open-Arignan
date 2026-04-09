from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from arignan.model_registry import (
    DEFAULT_EMBEDDING_MODEL_REPO_ID,
    DEFAULT_LOCAL_LLM_DISPLAY_NAME,
    DEFAULT_LOCAL_LLM_REPO_ID,
    DEFAULT_LIGHT_EMBEDDING_MODEL_REPO_ID,
    DEFAULT_LIGHT_LOCAL_LLM_DISPLAY_NAME,
    DEFAULT_LIGHT_LOCAL_LLM_REPO_ID,
    DEFAULT_LIGHT_RERANKER_MODEL_REPO_ID,
    DEFAULT_RERANKER_MODEL_REPO_ID,
    LEGACY_TRANSFORMERS_LOCAL_LLM_DISPLAY_NAME,
    LEGACY_TRANSFORMERS_LOCAL_LLM_REPO_ID,
    LEGACY_RERANKER_MODEL_REPO_ID,
    infer_local_llm_backend,
    resolve_ollama_model_id,
    resolve_model_repo_id,
    sanitize_model_id,
)


@dataclass(frozen=True, slots=True)
class SetupResult:
    install_target: str
    app_home: Path
    settings_path: Path
    models_dir: Path
    local_llm_backend: str
    local_llm_model: str
    local_llm_light_model: str
    embedding_model: str
    reranker_model: str
    bin_dir: Path
    windows_launcher: Path
    posix_launcher: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def install_target(dev: bool = False) -> str:
    return ".[dev]" if dev else "."


def ensure_repo_on_syspath(root: Path | None = None) -> Path:
    resolved = (root or repo_root()).resolve()
    src_dir = resolved / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return src_dir


def install_package(root: Path | None = None, dev: bool = False) -> str:
    resolved_root = (root or repo_root()).resolve()
    target = install_target(dev=dev)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", target],
        cwd=resolved_root,
        check=True,
    )
    return target
def update_local_llm_settings(
    settings_path: Path,
    local_llm_backend: str | None,
    local_llm_model: str | None,
    local_llm_light_model: str | None = None,
    embedding_model: str | None = None,
    reranker_model: str | None = None,
) -> None:
    if (
        local_llm_model is None
        and local_llm_backend is None
        and local_llm_light_model is None
        and embedding_model is None
        and reranker_model is None
    ):
        _migrate_legacy_local_llm_defaults(settings_path)
        _migrate_legacy_retrieval_defaults(settings_path)
        return
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    if local_llm_backend is not None:
        payload["local_llm_backend"] = local_llm_backend
    if local_llm_model is not None:
        payload["local_llm_model"] = local_llm_model
    if local_llm_light_model is not None:
        payload["local_llm_light_model"] = local_llm_light_model
    if embedding_model is not None:
        payload["embedding_model"] = embedding_model
    if reranker_model is not None:
        payload["reranker_model"] = reranker_model
    settings_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def initialize_local_state(
    app_home: Path | None = None,
    local_llm_backend: str | None = None,
    local_llm_model: str | None = None,
    local_llm_light_model: str | None = None,
    embedding_model: str | None = None,
    reranker_model: str | None = None,
) -> tuple[Path, Path]:
    from arignan.config import write_default_settings
    from arignan.paths import write_persisted_app_home
    from arignan.storage import StorageLayout

    settings_path = write_default_settings(app_home=app_home, overwrite=True)
    resolved_home = settings_path.parent.resolve()
    write_persisted_app_home(resolved_home)
    update_local_llm_settings(
        settings_path,
        local_llm_backend=local_llm_backend,
        local_llm_model=local_llm_model,
        local_llm_light_model=local_llm_light_model,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
    )
    layout = StorageLayout.from_home(app_home).ensure()
    return layout.root, settings_path


def provision_managed_runtime(app_home: Path, progress: Callable[[str], None] | None = None) -> Path:
    from arignan.llm.service import provision_managed_runtime as _provision_managed_runtime

    return _provision_managed_runtime(app_home, progress=progress)


def ensure_model_available(
    app_home: Path,
    endpoint: str,
    model: str,
    *,
    context_window: int | None = None,
    flash_attention: bool | None = None,
    kv_cache_type: str | None = None,
    num_parallel: int | None = None,
    max_loaded_models: int | None = None,
    progress: Callable[[str], None] | None = None,
    timeout_seconds: float = 1800.0,
) -> None:
    from arignan.llm.service import ensure_model_available as _ensure_model_available

    _ensure_model_available(
        app_home,
        endpoint,
        model,
        context_window=context_window,
        flash_attention=flash_attention,
        kv_cache_type=kv_cache_type,
        num_parallel=num_parallel,
        max_loaded_models=max_loaded_models,
        progress=progress,
        timeout_seconds=timeout_seconds,
    )


def download_required_models(app_home: Path, progress: Callable[[str], None] | None = None) -> Path:
    from arignan.config import load_config

    config = load_config(app_home=app_home)
    models_dir = app_home / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    requested_models = _configured_local_models(config)
    backends = {
        backend: models
        for backend, models in _group_models_by_backend(requested_models, default_backend=config.local_llm_backend).items()
        if models
    }
    if "ollama" in backends:
        provision_managed_runtime(app_home, progress=progress)
        for model in backends["ollama"]:
            ensure_model_available(
                app_home,
                config.local_llm_endpoint,
                model,
                context_window=config.local_llm_context_window,
                flash_attention=config.local_llm_flash_attention,
                kv_cache_type=config.local_llm_kv_cache_type,
                num_parallel=config.local_llm_num_parallel,
                max_loaded_models=config.local_llm_max_loaded_models,
                progress=progress,
                timeout_seconds=1800.0,
            )
    if any(backend in {"transformers", "huggingface"} for backend in backends):
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("huggingface_hub is required to bootstrap the local model bundle") from exc

        for model_id in backends.get("transformers", []) + backends.get("huggingface", []):
            repo_id = resolve_model_repo_id(model_id)
            target_dir = models_dir / sanitize_model_id(repo_id)
            try:
                snapshot_download(repo_id=repo_id, local_dir=target_dir, local_dir_use_symlinks=False)
            except (RepositoryNotFoundError, GatedRepoError, HfHubHTTPError) as exc:
                raise RuntimeError(_format_model_download_error(app_home, model_id, repo_id, exc)) from exc
    _download_retrieval_models(app_home, models_dir=models_dir, progress=progress)
    unsupported = [backend for backend in backends if backend not in {"ollama", "transformers", "huggingface"}]
    if unsupported:
        raise RuntimeError(f"Unsupported local_llm_backend '{', '.join(unsupported)}'")
    _write_runtime_manifest(
        models_dir,
        backend=config.local_llm_backend,
        model=config.local_llm_model,
        light_model=config.local_llm_light_model,
        embedding_model=config.embedding_model,
        reranker_model=config.reranker_model,
    )
    return models_dir


def _write_runtime_manifest(
    models_dir: Path,
    *,
    backend: str,
    model: str,
    light_model: str,
    embedding_model: str,
    reranker_model: str,
) -> None:
    payload = {
        "local_llm_backend": backend,
        "local_llm_model": model,
        "local_llm_light_model": light_model,
        "embedding_model": embedding_model,
        "reranker_model": reranker_model,
    }
    (models_dir / "local_llm_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _migrate_legacy_local_llm_defaults(settings_path: Path) -> None:
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    current_model = payload.get("local_llm_model")
    current_backend = payload.get("local_llm_backend")
    legacy_models = {
        None,
        LEGACY_TRANSFORMERS_LOCAL_LLM_DISPLAY_NAME,
        LEGACY_TRANSFORMERS_LOCAL_LLM_REPO_ID,
    }
    if current_backend not in {None, "transformers"}:
        return
    if current_model not in legacy_models:
        return
    payload["local_llm_backend"] = "ollama"
    payload["local_llm_model"] = DEFAULT_LOCAL_LLM_REPO_ID
    payload.setdefault("local_llm_light_model", DEFAULT_LIGHT_LOCAL_LLM_REPO_ID)
    settings_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _migrate_legacy_retrieval_defaults(settings_path: Path) -> None:
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    changed = False
    if payload.get("embedding_model") in {None, ""}:
        payload["embedding_model"] = DEFAULT_EMBEDDING_MODEL_REPO_ID
        changed = True
    if payload.get("reranker_model") in {None, "", LEGACY_RERANKER_MODEL_REPO_ID}:
        payload["reranker_model"] = DEFAULT_RERANKER_MODEL_REPO_ID
        changed = True
    if changed:
        settings_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def create_launchers(root: Path | None = None, app_home: Path | None = None) -> tuple[Path, Path, Path]:
    resolved_root = (root or repo_root()).resolve()
    bin_dir = resolved_root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    python_executable = Path(sys.executable).resolve()
    app_home_arg_windows = f' --app-home "{Path(app_home).resolve()}"' if app_home is not None else ""
    app_home_arg_posix = f" --app-home {_quote_posix_argument(str(Path(app_home).resolve()))}" if app_home is not None else ""
    windows_launcher = bin_dir / "arignan.cmd"
    windows_launcher.write_text(
        "@echo off\r\n"
        f"\"{python_executable}\" -m arignan.cli{app_home_arg_windows} %*\r\n",
        encoding="utf-8",
    )

    posix_launcher = bin_dir / "arignan"
    posix_launcher.write_text(
        "#!/usr/bin/env sh\n"
        f"{_quote_posix_argument(str(python_executable))} -m arignan.cli{app_home_arg_posix} \"$@\"\n",
        encoding="utf-8",
    )
    current_mode = posix_launcher.stat().st_mode
    posix_launcher.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, windows_launcher, posix_launcher


def run_setup(
    dev: bool = False,
    app_home: Path | None = None,
    llm_backend: str | None = None,
    llm_model: str | None = None,
    lightweight: bool = False,
    progress: Callable[[str], None] | None = None,
) -> SetupResult:
    root = repo_root()
    ensure_repo_on_syspath(root)
    effective_llm_model = DEFAULT_LIGHT_LOCAL_LLM_REPO_ID if lightweight else llm_model
    effective_light_model = DEFAULT_LIGHT_LOCAL_LLM_REPO_ID if lightweight else None
    effective_embedding_model = DEFAULT_LIGHT_EMBEDDING_MODEL_REPO_ID if lightweight else None
    effective_reranker_model = DEFAULT_LIGHT_RERANKER_MODEL_REPO_ID if lightweight else None
    _emit(progress, "[1/4] Installing Python package...")
    target = install_package(root=root, dev=dev)
    _emit(progress, "[2/4] Initializing local Arignan state...")
    resolved_home, settings_path = initialize_local_state(
        app_home=app_home,
        local_llm_backend=llm_backend,
        local_llm_model=effective_llm_model,
        local_llm_light_model=effective_light_model,
        embedding_model=effective_embedding_model,
        reranker_model=effective_reranker_model,
    )
    _emit(progress, "[3/4] Downloading required models...")
    models_dir = download_required_models(resolved_home, progress=progress)
    _emit(progress, "[4/4] Creating CLI launchers...")
    pinned_app_home = resolved_home if app_home is not None else None
    bin_dir, windows_launcher, posix_launcher = create_launchers(root=root, app_home=pinned_app_home)
    from arignan.config import load_config

    config = load_config(app_home=resolved_home)
    _emit(progress, "[done] Setup steps completed.")
    return SetupResult(
        install_target=target,
        app_home=resolved_home,
        settings_path=settings_path,
        models_dir=models_dir,
        local_llm_backend=config.local_llm_backend,
        local_llm_model=config.local_llm_model,
        local_llm_light_model=config.local_llm_light_model,
        embedding_model=config.embedding_model,
        reranker_model=config.reranker_model,
        bin_dir=bin_dir,
        windows_launcher=windows_launcher,
        posix_launcher=posix_launcher,
    )


def render_summary(result: SetupResult) -> str:
    path_command = "arignan --help"
    direct_command = _display_path(result.bin_dir / ("arignan.cmd" if os.name == "nt" else "arignan")) + " --help"
    lines = [
        "Arignan setup complete.",
        f"- Installed package target: {result.install_target}",
        f"- App home: {_display_path(result.app_home)}",
        f"- Settings: {_display_path(result.settings_path)}",
        f"- Models directory: {_display_path(result.models_dir)}",
        f"- Local LLM backend: {result.local_llm_backend}",
        f"- Local LLM model: {result.local_llm_model}",
        f"- Light local LLM model: {result.local_llm_light_model}",
        f"- Embedding model: {result.embedding_model}",
        f"- Reranker model: {result.reranker_model}",
        f"- Bin directory: {_display_path(result.bin_dir)}",
        f"- Windows launcher: {_display_path(result.windows_launcher)}",
        f"- POSIX launcher: {_display_path(result.posix_launcher)}",
        "",
        "Next steps:",
        f"1. Add '{_display_path(result.bin_dir)}' to PATH if you want to run just: {path_command}",
        f"2. Or run directly from the repo with: {direct_command}",
        f"3. Python fallback: {_display_path(sys.executable)} -m arignan.cli --help",
    ]
    return "\n".join(lines)


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _format_model_download_error(app_home: Path, configured_model: str, repo_id: str, exc: Exception) -> str:
    settings_path = app_home / "settings.json"
    lines = [
        f"Failed to download model '{configured_model}'.",
    ]
    if repo_id != configured_model:
        lines.append(f"Resolved Hugging Face repo: {repo_id}")
    lines.append(f"Reason: {exc}")
    lines.append(f"You can also edit {settings_path} and rerun python setup.py.")
    lines.append("If the model repo is gated, authenticate with `huggingface-cli login` before retrying.")
    return "\n".join(lines)


def _quote_posix_argument(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _display_path(value: Path | str) -> str:
    text = str(value)
    if text.startswith("\\\\?\\UNC\\"):
        return "\\\\" + text[8:]
    if text.startswith("\\\\?\\"):
        return text[4:]
    return text


def _configured_local_models(config) -> list[str]:
    models: list[str] = []
    for model in [config.local_llm_model, config.local_llm_light_model]:
        candidate = str(model).strip()
        if not candidate or candidate in models:
            continue
        models.append(candidate)
    return models


def _download_retrieval_models(
    app_home: Path,
    *,
    models_dir: Path,
    progress: Callable[[str], None] | None = None,
) -> None:
    from arignan.config import load_config

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required to bootstrap retrieval models") from exc

    config = load_config(app_home=app_home)
    retrieval_models = [
        ("embedding model", config.embedding_model),
        ("reranker model", config.reranker_model),
    ]
    for label, model_id in retrieval_models:
        repo_id = resolve_model_repo_id(model_id)
        target_dir = models_dir / sanitize_model_id(repo_id)
        if target_dir.exists():
            continue
        _emit(progress, f"Downloading local {label} ({model_id})...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["onnx/*", "openvino/*", "gguf/*", "*.onnx", "*.xml"],
        )


def _group_models_by_backend(models: list[str], *, default_backend: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for model in models:
        backend = infer_local_llm_backend(model, default=default_backend).strip().lower()
        normalized = resolve_ollama_model_id(model) if backend == "ollama" else model
        grouped.setdefault(backend, [])
        if normalized not in grouped[backend]:
            grouped[backend].append(normalized)
    return grouped
