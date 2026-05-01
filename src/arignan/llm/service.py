from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import httpx

WINDOWS_OLLAMA_AMD64_ZIP_URL = "https://ollama.com/download/ollama-windows-amd64.zip"


def _is_windows_platform() -> bool:
    return os.name == "nt"


def managed_runtime_dir(app_home: Path) -> Path:
    return app_home / "runtime" / "local_llm"


def managed_runtime_logs_dir(app_home: Path) -> Path:
    return managed_runtime_dir(app_home) / "logs"


def bundled_ollama_executable(app_home: Path) -> Path:
    executable = "ollama.exe" if _is_windows_platform() else "ollama"
    return managed_runtime_dir(app_home) / executable


def system_ollama_executable() -> Path | None:
    discovered = shutil.which("ollama")
    if not discovered:
        return None
    return Path(discovered).resolve()


def resolve_ollama_executable(app_home: Path) -> Path:
    system = system_ollama_executable()
    if system is not None:
        return system
    bundled = bundled_ollama_executable(app_home)
    if bundled.exists():
        return bundled.resolve()
    raise RuntimeError(
        "The local model runtime is not provisioned. Re-run `python setup.py --app-home <install dir>`."
    )


def provision_managed_runtime(
    app_home: Path,
    progress: Callable[[str], None] | None = None,
) -> Path:
    system = system_ollama_executable()
    if system is not None:
        _emit(progress, "Using existing local model runtime from PATH...")
        return system
    executable = bundled_ollama_executable(app_home)
    if executable.exists():
        return executable
    if _is_windows_platform():
        _emit(progress, "Installing local model runtime...")
        return _install_windows_runtime(app_home)
    raise RuntimeError(
        "Automatic local model runtime provisioning is only bundled for Windows right now. "
        "Re-run setup on Windows or install Ollama manually on this platform."
    )


def ensure_service_running(
    app_home: Path,
    endpoint: str,
    *,
    context_window: int | None = None,
    flash_attention: bool | None = None,
    kv_cache_type: str | None = None,
    num_parallel: int | None = None,
    max_loaded_models: int | None = None,
    progress: Callable[[str], None] | None = None,
    ready_timeout_seconds: float = 20.0,
) -> None:
    if is_service_ready(endpoint):
        return
    executable = resolve_ollama_executable(app_home)
    bundled_executable = bundled_ollama_executable(app_home).resolve()
    if executable != bundled_executable:
        raise RuntimeError(
            "A system Ollama installation was found on PATH, but no running Ollama service was reachable at "
            f"{endpoint}. Arignan will not launch a separate system-Ollama daemon during setup or model preparation, "
            "because that can point at a different model store than the one your existing Ollama app is using. "
            "Start your normal Ollama app/service first, then rerun the command."
        )
    log_dir = managed_runtime_logs_dir(app_home)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "service.log"
    pid_path = managed_runtime_dir(app_home) / "service.pid"
    _emit(progress, "Starting local model runtime...")
    env = os.environ.copy()
    if executable == bundled_executable:
        env["OLLAMA_MODELS"] = str((app_home / "models").resolve())
        env["OLLAMA_HOST"] = _ollama_host(endpoint)
        if flash_attention:
            env["OLLAMA_FLASH_ATTENTION"] = "1"
        if isinstance(context_window, int) and context_window > 0:
            env["OLLAMA_CONTEXT_LENGTH"] = str(context_window)
        if kv_cache_type:
            env["OLLAMA_KV_CACHE_TYPE"] = kv_cache_type
        if isinstance(num_parallel, int) and num_parallel > 0:
            env["OLLAMA_NUM_PARALLEL"] = str(num_parallel)
        if isinstance(max_loaded_models, int) and max_loaded_models > 0:
            env["OLLAMA_MAX_LOADED_MODELS"] = str(max_loaded_models)
    handle = log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [str(executable), "serve"],
            cwd=executable.parent,
            env=env,
            stdout=handle,
            stderr=handle,
            **_background_process_kwargs(),
        )
    finally:
        handle.close()
    pid_path.write_text(str(process.pid), encoding="utf-8")
    if _wait_for_service(endpoint, timeout_seconds=ready_timeout_seconds):
        return
    raise RuntimeError(f"Local model runtime failed to start. Log: {log_path.resolve()}")


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
    ensure_service_running(
        app_home,
        endpoint,
        context_window=context_window,
        flash_attention=flash_attention,
        kv_cache_type=kv_cache_type,
        num_parallel=num_parallel,
        max_loaded_models=max_loaded_models,
        progress=progress,
    )
    if model in list_available_models(endpoint):
        return
    _emit(progress, f"Configured local model '{model}' is not cached locally yet. Downloading it now...")
    try:
        with httpx.stream(
            "POST",
            endpoint.rstrip("/") + "/api/pull",
            json={"model": model, "stream": True},
            timeout=timeout_seconds,
        ) as response:
            response.raise_for_status()
            _stream_ollama_pull_progress(response, model=model, progress=progress)
    except httpx.HTTPError as exc:
        details = exc.response.text if exc.response is not None else str(exc)
        raise RuntimeError(f"Failed to prepare local model '{model}' from the local runtime: {details}") from exc
    _emit(progress, f"Local model '{model}' is ready.")


def list_available_models(endpoint: str) -> set[str]:
    response = httpx.get(endpoint.rstrip("/") + "/api/tags", timeout=5.0)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("models", [])
    names: set[str] = set()
    if not isinstance(models, list):
        return names
    for item in models:
        if isinstance(item, dict):
            for key in ("name", "model"):
                value = item.get(key)
                if isinstance(value, str):
                    names.add(value)
    return names


def list_running_models(endpoint: str) -> list[str]:
    response = httpx.get(endpoint.rstrip("/") + "/api/ps", timeout=5.0)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("models", [])
    names: list[str] = []
    if not isinstance(models, list):
        return names
    for item in models:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model"):
            value = item.get(key)
            if isinstance(value, str) and value not in names:
                names.append(value)
    return names


def unload_model(endpoint: str, model: str) -> None:
    response = httpx.post(
        endpoint.rstrip("/") + "/api/generate",
        json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
        timeout=30.0,
    )
    response.raise_for_status()


def release_running_models(
    endpoint: str,
    *,
    progress: Callable[[str], None] | None = None,
    exclude: set[str] | None = None,
) -> list[str]:
    excluded = exclude or set()
    released: list[str] = []
    for model in list_running_models(endpoint):
        if model in excluded:
            continue
        unload_model(endpoint, model)
        released.append(model)
        _emit(progress, f"Unloaded local model runtime model '{model}' to recover memory.")
    return released


def describe_running_models(endpoint: str) -> list[str]:
    response = httpx.get(endpoint.rstrip("/") + "/api/ps", timeout=5.0)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("models", [])
    descriptions: list[str] = []
    if not isinstance(models, list):
        return descriptions
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
        if not isinstance(name, str) or not name:
            continue
        size_vram = item.get("size_vram")
        if isinstance(size_vram, (int, float)):
            descriptions.append(f"{name} ({float(size_vram) / (1024 ** 3):.2f} GiB VRAM)")
            continue
        descriptions.append(name)
    return descriptions


def is_service_ready(endpoint: str, timeout_seconds: float = 1.0) -> bool:
    try:
        response = httpx.get(endpoint.rstrip("/") + "/api/version", timeout=timeout_seconds)
        response.raise_for_status()
    except httpx.HTTPError:
        return False
    return True


def _install_windows_runtime(app_home: Path) -> Path:
    runtime_dir = managed_runtime_dir(app_home)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    archive_path = runtime_dir / "ollama-windows-amd64.zip"
    with httpx.stream("GET", WINDOWS_OLLAMA_AMD64_ZIP_URL, timeout=120.0, follow_redirects=True) as response:
        response.raise_for_status()
        with archive_path.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(runtime_dir)
    archive_path.unlink(missing_ok=True)
    executable = bundled_ollama_executable(app_home)
    if not executable.exists():
        raise RuntimeError(f"Managed local model runtime install did not produce {executable.name}.")
    return executable


def _background_process_kwargs() -> dict[str, object]:
    if _is_windows_platform():
        return {
            "creationflags": (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
            )
        }
    return {"start_new_session": True}


def _ollama_host(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    return parsed.netloc or endpoint


def _wait_for_service(endpoint: str, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if is_service_ready(endpoint):
            return True
        time.sleep(0.5)
    return False


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _stream_ollama_pull_progress(
    response,
    *,
    model: str,
    progress: Callable[[str], None] | None = None,
) -> None:
    last_status: str | None = None
    last_percent = -1
    for line in response.iter_lines():
        if not line:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = str(payload.get("status", "")).strip()
        completed = payload.get("completed")
        total = payload.get("total")
        percent = None
        if isinstance(completed, int) and isinstance(total, int) and total > 0:
            percent = int((completed / total) * 100)
        if percent is not None:
            bucket = percent // 10
            if bucket != last_percent:
                _emit(progress, f"Downloading local model ({model})... {percent}%")
                last_percent = bucket
                last_status = status or last_status
                continue
        if status and status != last_status:
            _emit(progress, f"Downloading local model ({model})... {status}")
            last_status = status
