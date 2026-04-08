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


def managed_runtime_dir(app_home: Path) -> Path:
    return app_home / "runtime" / "local_llm"


def managed_runtime_logs_dir(app_home: Path) -> Path:
    return managed_runtime_dir(app_home) / "logs"


def bundled_ollama_executable(app_home: Path) -> Path:
    executable = "ollama.exe" if os.name == "nt" else "ollama"
    return managed_runtime_dir(app_home) / executable


def resolve_ollama_executable(app_home: Path) -> Path:
    bundled = bundled_ollama_executable(app_home)
    if bundled.exists():
        return bundled.resolve()
    discovered = shutil.which("ollama")
    if discovered:
        return Path(discovered).resolve()
    raise RuntimeError(
        "The local model runtime is not provisioned. Re-run `python setup.py --app-home <install dir>`."
    )


def provision_managed_runtime(
    app_home: Path,
    progress: Callable[[str], None] | None = None,
) -> Path:
    executable = bundled_ollama_executable(app_home)
    if executable.exists():
        return executable
    if os.name == "nt":
        _emit(progress, "Installing local model runtime...")
        return _install_windows_runtime(app_home)
    discovered = shutil.which("ollama")
    if discovered:
        return Path(discovered).resolve()
    raise RuntimeError(
        "Automatic local model runtime provisioning is only bundled for Windows right now. "
        "Re-run setup on Windows or install Ollama manually on this platform."
    )


def ensure_service_running(
    app_home: Path,
    endpoint: str,
    progress: Callable[[str], None] | None = None,
    ready_timeout_seconds: float = 20.0,
) -> None:
    if is_service_ready(endpoint):
        return
    executable = resolve_ollama_executable(app_home)
    log_dir = managed_runtime_logs_dir(app_home)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "service.log"
    pid_path = managed_runtime_dir(app_home) / "service.pid"
    _emit(progress, "Starting local model runtime...")
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = str((app_home / "models").resolve())
    env["OLLAMA_HOST"] = _ollama_host(endpoint)
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
    progress: Callable[[str], None] | None = None,
    timeout_seconds: float = 1800.0,
) -> None:
    ensure_service_running(app_home, endpoint, progress=progress)
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
            name = item.get("name")
            if isinstance(name, str):
                names.add(name)
    return names


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
    if os.name == "nt":
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
