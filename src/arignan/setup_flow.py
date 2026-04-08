from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DEFAULT_LOCAL_LLM_DISPLAY_NAME = "Qwen3-1.7B"
DEFAULT_LOCAL_LLM_REPO_ID = "Qwen/Qwen3-1.7B"
MODEL_REPO_ALIASES = {
    DEFAULT_LOCAL_LLM_DISPLAY_NAME: DEFAULT_LOCAL_LLM_REPO_ID,
}


@dataclass(frozen=True, slots=True)
class SetupResult:
    install_target: str
    app_home: Path
    settings_path: Path
    models_dir: Path
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
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", target],
        cwd=resolved_root,
        check=True,
    )
    return target


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "__").replace("\\", "__").replace(":", "_")


def resolve_model_repo_id(model_id: str) -> str:
    return MODEL_REPO_ALIASES.get(model_id, model_id)


def update_local_llm_model(settings_path: Path, local_llm_model: str | None) -> None:
    if local_llm_model is None:
        return
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    payload["local_llm_model"] = local_llm_model
    settings_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def initialize_local_state(
    app_home: Path | None = None,
    local_llm_model: str | None = None,
) -> tuple[Path, Path]:
    from arignan.config import write_default_settings
    from arignan.paths import write_persisted_app_home
    from arignan.storage import StorageLayout

    settings_path = write_default_settings(app_home=app_home)
    resolved_home = settings_path.parent.resolve()
    write_persisted_app_home(resolved_home)
    update_local_llm_model(settings_path, local_llm_model=local_llm_model)
    layout = StorageLayout.from_home(app_home).ensure()
    return layout.root, settings_path


def download_required_models(app_home: Path) -> Path:
    from arignan.config import load_config

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required to bootstrap the local model bundle") from exc

    config = load_config(app_home=app_home)
    models_dir = app_home / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_ids = [
        config.local_llm_model,
        config.embedding_model,
        config.reranker_model,
    ]
    for model_id in model_ids:
        repo_id = resolve_model_repo_id(model_id)
        target_dir = models_dir / sanitize_model_id(repo_id)
        try:
            snapshot_download(repo_id=repo_id, local_dir=target_dir, local_dir_use_symlinks=False)
        except (RepositoryNotFoundError, GatedRepoError, HfHubHTTPError) as exc:
            raise RuntimeError(_format_model_download_error(app_home, model_id, repo_id, exc)) from exc
    return models_dir


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
    llm_model: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> SetupResult:
    root = repo_root()
    ensure_repo_on_syspath(root)
    _emit(progress, "[1/4] Installing Python package...")
    target = install_package(root=root, dev=dev)
    _emit(progress, "[2/4] Initializing local Arignan state...")
    resolved_home, settings_path = initialize_local_state(app_home=app_home, local_llm_model=llm_model)
    _emit(progress, "[3/4] Downloading required models...")
    models_dir = download_required_models(resolved_home)
    _emit(progress, "[4/4] Creating CLI launchers...")
    pinned_app_home = resolved_home if app_home is not None else None
    bin_dir, windows_launcher, posix_launcher = create_launchers(root=root, app_home=pinned_app_home)
    _emit(progress, "[done] Setup steps completed.")
    return SetupResult(
        install_target=target,
        app_home=resolved_home,
        settings_path=settings_path,
        models_dir=models_dir,
        bin_dir=bin_dir,
        windows_launcher=windows_launcher,
        posix_launcher=posix_launcher,
    )


def render_summary(result: SetupResult) -> str:
    path_command = "arignan --help"
    direct_command = str(result.bin_dir / ("arignan.cmd" if os.name == "nt" else "arignan")) + " --help"
    lines = [
        "Arignan setup complete.",
        f"- Installed package target: {result.install_target}",
        f"- App home: {result.app_home}",
        f"- Settings: {result.settings_path}",
        f"- Models directory: {result.models_dir}",
        f"- Bin directory: {result.bin_dir}",
        f"- Windows launcher: {result.windows_launcher}",
        f"- POSIX launcher: {result.posix_launcher}",
        "",
        "Next steps:",
        f"1. Add '{result.bin_dir}' to PATH if you want to run just: {path_command}",
        f"2. Or run directly from the repo with: {direct_command}",
        f"3. Python fallback: {sys.executable} -m arignan.cli --help",
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
