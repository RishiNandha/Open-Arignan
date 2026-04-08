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
    calls: list[list[str]] = []
    readiness = iter([False, True])

    class FakeProcess:
        pid = 3210

    monkeypatch.setattr("arignan.llm.service.is_service_ready", lambda endpoint, timeout_seconds=1.0: next(readiness))
    monkeypatch.setattr(
        "arignan.llm.service.subprocess.Popen",
        lambda command, **kwargs: calls.append(command) or FakeProcess(),
    )

    ensure_service_running(tmp_path, "http://127.0.0.1:11434")

    assert calls == [[str(executable), "serve"]]
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

    monkeypatch.setattr(
        "arignan.llm.service.ensure_service_running",
        lambda app_home, endpoint, progress=None, ready_timeout_seconds=20.0: ensured.append(endpoint),
    )
    monkeypatch.setattr("arignan.llm.service.list_available_models", lambda endpoint: set())

    class FakeResponse:
        @staticmethod
        def raise_for_status() -> None:
            return None

    monkeypatch.setattr(
        "arignan.llm.service.httpx.post",
        lambda url, json, timeout: pulled.append({"url": url, "json": json, "timeout": timeout}) or FakeResponse(),
    )

    ensure_model_available(tmp_path, "http://127.0.0.1:11434", "qwen3:4b-q4_K_M")

    assert ensured == ["http://127.0.0.1:11434"]
    assert pulled == [
        {
            "url": "http://127.0.0.1:11434/api/pull",
            "json": {"model": "qwen3:4b-q4_K_M", "stream": False},
            "timeout": 1800.0,
        }
    ]
