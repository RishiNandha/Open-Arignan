from __future__ import annotations

import io
from pathlib import Path

import httpx

from arignan.llm.service import (
    bundled_ollama_executable,
    ensure_model_available,
    ensure_service_running,
    is_service_ready,
    list_available_models,
    managed_runtime_dir,
    provision_managed_runtime,
)


def test_provision_managed_runtime_downloads_windows_bundle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("arignan.llm.service.os.name", "nt")
    archive_bytes = io.BytesIO()
    captured: dict[str, object] = {}
    import zipfile

    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("ollama.exe", "binary")

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        @staticmethod
        def raise_for_status() -> None:
            return None

        def iter_bytes(self):
            yield archive_bytes.getvalue()

    def fake_stream(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeStream()

    monkeypatch.setattr("arignan.llm.service.httpx.stream", fake_stream)

    executable = provision_managed_runtime(tmp_path)

    assert executable == bundled_ollama_executable(tmp_path)
    assert executable.exists()
    assert captured["args"] == ("GET", "https://ollama.com/download/ollama-windows-amd64.zip")
    assert captured["kwargs"]["follow_redirects"] is True


def test_ensure_service_running_starts_background_runtime_when_endpoint_is_down(tmp_path: Path, monkeypatch) -> None:
    executable = bundled_ollama_executable(tmp_path)
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_text("", encoding="utf-8")
    calls: list[dict[str, object]] = []
    readiness = iter([False, True])

    class FakeProcess:
        pid = 3210

    monkeypatch.setattr("arignan.llm.service.is_service_ready", lambda endpoint, timeout_seconds=1.0: next(readiness))
    monkeypatch.setattr(
        "arignan.llm.service.subprocess.Popen",
        lambda command, **kwargs: calls.append({"command": command, "kwargs": kwargs}) or FakeProcess(),
    )

    ensure_service_running(
        tmp_path,
        "http://127.0.0.1:11434",
        context_window=6144,
        flash_attention=True,
        kv_cache_type="q8_0",
        num_parallel=1,
        max_loaded_models=1,
    )

    assert calls[0]["command"] == [str(executable), "serve"]
    env = calls[0]["kwargs"]["env"]
    assert env["OLLAMA_FLASH_ATTENTION"] == "1"
    assert env["OLLAMA_CONTEXT_LENGTH"] == "6144"
    assert env["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    assert env["OLLAMA_NUM_PARALLEL"] == "1"
    assert env["OLLAMA_MAX_LOADED_MODELS"] == "1"
    assert (managed_runtime_dir(tmp_path) / "service.pid").read_text(encoding="utf-8") == "3210"


def test_list_available_models_reads_tag_names(monkeypatch) -> None:
    class FakeResponse:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict[str, object]:
            return {"models": [{"name": "qwen3:4b-q4_K_M"}, {"name": "other"}]}

    monkeypatch.setattr("arignan.llm.service.httpx.get", lambda *args, **kwargs: FakeResponse())

    assert list_available_models("http://127.0.0.1:11434") == {"qwen3:4b-q4_K_M", "other"}


def test_is_service_ready_returns_false_on_http_error(monkeypatch) -> None:
    def fake_get(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr("arignan.llm.service.httpx.get", fake_get)

    assert is_service_ready("http://127.0.0.1:11434") is False


def test_ensure_model_available_pulls_when_model_missing(monkeypatch, tmp_path: Path) -> None:
    ensured: list[str] = []
    pulled: list[dict[str, object]] = []
    progress: list[str] = []

    monkeypatch.setattr(
        "arignan.llm.service.ensure_service_running",
        lambda app_home, endpoint, **kwargs: ensured.append(endpoint),
    )
    monkeypatch.setattr("arignan.llm.service.list_available_models", lambda endpoint: set())

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def iter_lines():
            return iter(
                [
                    '{"status":"pulling manifest"}',
                    '{"status":"pulling layers","completed":25,"total":100}',
                    '{"status":"pulling layers","completed":100,"total":100}',
                    '{"status":"success"}',
                ]
            )

    monkeypatch.setattr(
        "arignan.llm.service.httpx.stream",
        lambda method, url, json, timeout: pulled.append(
            {"method": method, "url": url, "json": json, "timeout": timeout}
        )
        or FakeStream(),
    )

    ensure_model_available(tmp_path, "http://127.0.0.1:11434", "qwen3:4b-q4_K_M", progress=progress.append)

    assert ensured == ["http://127.0.0.1:11434"]
    assert pulled == [
        {
            "method": "POST",
            "url": "http://127.0.0.1:11434/api/pull",
            "json": {"model": "qwen3:4b-q4_K_M", "stream": True},
            "timeout": 1800.0,
        }
    ]
    assert "is not cached locally yet" in progress[0]
    assert any("25%" in message or "100%" in message for message in progress)
    assert progress[-1] == "Local model 'qwen3:4b-q4_K_M' is ready."
