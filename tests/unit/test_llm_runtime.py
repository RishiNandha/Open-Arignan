from __future__ import annotations

import importlib
import os
import sys
import types

import arignan
import arignan.llm.runtime as llm_runtime
from arignan.config import AppConfig
from arignan.llm.runtime import OllamaTextGenerator, TransformersTextGenerator, create_local_text_generator
from arignan.runtime_env import TEXT_RUNTIME_ENVIRONMENT, configure_text_runtime_environment
import httpx


def test_configure_text_runtime_environment_overrides_conflicting_values(monkeypatch) -> None:
    for key in TEXT_RUNTIME_ENVIRONMENT:
        monkeypatch.setenv(key, "unexpected")

    applied = configure_text_runtime_environment()

    assert applied == TEXT_RUNTIME_ENVIRONMENT
    for key, value in TEXT_RUNTIME_ENVIRONMENT.items():
        assert os.environ[key] == value


def test_package_import_reapplies_text_runtime_environment(monkeypatch) -> None:
    for key in TEXT_RUNTIME_ENVIRONMENT:
        monkeypatch.setenv(key, "unexpected")

    importlib.reload(arignan)

    for key, value in TEXT_RUNTIME_ENVIRONMENT.items():
        assert os.environ[key] == value


def test_llm_runtime_import_reapplies_text_runtime_environment(monkeypatch) -> None:
    for key in TEXT_RUNTIME_ENVIRONMENT:
        monkeypatch.setenv(key, "unexpected")

    importlib.reload(llm_runtime)

    for key, value in TEXT_RUNTIME_ENVIRONMENT.items():
        assert os.environ[key] == value


def test_create_local_text_generator_uses_ollama_by_default(app_home) -> None:
    generator = create_local_text_generator(AppConfig(app_home=app_home))

    assert generator.__class__.__name__ == "OllamaTextGenerator"
    assert generator.backend_name == "ollama-local"
    assert generator.model_name == "qwen3:4b-q4_K_M"


def test_create_local_text_generator_can_override_model(app_home) -> None:
    generator = create_local_text_generator(AppConfig(app_home=app_home), model_name="qwen3:0.6b")

    assert generator.__class__.__name__ == "OllamaTextGenerator"
    assert generator.backend_name == "ollama-local"
    assert generator.model_name == "qwen3:0.6b"


def test_ollama_text_generator_posts_chat_request_and_strips_think_blocks(app_home) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        @staticmethod
        def json() -> dict[str, object]:
            return {"message": {"content": "<think>hidden reasoning</think>\nVisible answer."}}

    class FakeClient:
        def post(self, url: str, json: dict[str, object]) -> FakeResponse:
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    generator = OllamaTextGenerator(AppConfig(app_home=app_home))
    generator._model_ready = True
    generator._client = FakeClient()  # type: ignore[assignment]

    output = generator.generate(
        system_prompt="System prompt",
        user_prompt="User prompt",
        chat_messages=[
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
        ],
        max_new_tokens=256,
        temperature=0.0,
        response_format={"type": "object"},
    )

    assert output == "Visible answer."
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    payload = captured["json"]
    assert payload["model"] == "qwen3:4b-q4_K_M"
    assert payload["options"]["num_predict"] == 256
    assert payload["options"]["num_ctx"] == 6144
    assert payload["format"] == {"type": "object"}
    assert payload["think"] is False
    assert payload["messages"][1:] == [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "User prompt"},
    ]


def test_ollama_text_generator_retries_once_after_memory_pressure(app_home, monkeypatch) -> None:
    progress: list[str] = []
    release_calls: list[str] = []
    released_models: list[dict[str, object]] = []

    class RetryResponse:
        def __init__(self, *, text: str = "", ok: bool = False) -> None:
            self.text = text
            self._ok = ok

        def raise_for_status(self) -> None:
            if self._ok:
                return None
            request = httpx.Request("POST", "http://127.0.0.1:11434/api/chat")
            response = httpx.Response(500, request=request, text=self.text)
            raise httpx.HTTPStatusError("boom", request=request, response=response)

        @staticmethod
        def json() -> dict[str, object]:
            return {"message": {"content": "Recovered answer."}}

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, url: str, json: dict[str, object]):
            self.calls += 1
            if self.calls == 1:
                return RetryResponse(text='{"error":"model requires more system memory (7.1 GiB) than is available (4.7 GiB)"}')
            return RetryResponse(ok=True)

    monkeypatch.setattr(
        "arignan.llm.runtime.release_running_models",
        lambda endpoint, progress=None, exclude=None: released_models.append(
            {"endpoint": endpoint, "exclude": exclude}
        )
        or ["gemma4:e2b"],
    )

    generator = OllamaTextGenerator(
        AppConfig(app_home=app_home, local_llm_model="qwen3:4b-q4_K_M"),
        progress_sink=progress.append,
        memory_recovery=lambda reason: release_calls.append(reason) or True,
    )
    generator._model_ready = True
    generator._client = FakeClient()  # type: ignore[assignment]

    output = generator.generate(system_prompt="System prompt", user_prompt="User prompt")

    assert output == "Recovered answer."
    assert len(release_calls) == 1
    assert "requires more system memory" in release_calls[0]
    assert released_models == [
        {"endpoint": "http://127.0.0.1:11434", "exclude": {"qwen3:4b-q4_K_M"}}
    ]
    assert any("retrying once" in message for message in progress)


def test_ollama_text_generator_retries_once_after_connect_failure(app_home, monkeypatch) -> None:
    progress: list[str] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        @staticmethod
        def json() -> dict[str, object]:
            return {"message": {"content": "Recovered answer."}}

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, url: str, json: dict[str, object]):
            self.calls += 1
            if self.calls == 1:
                raise httpx.ConnectError("connection reset by peer")
            return FakeResponse()

    generator = OllamaTextGenerator(AppConfig(app_home=app_home), progress_sink=progress.append)
    generator._model_ready = True
    generator._client = FakeClient()  # type: ignore[assignment]
    ensure_calls: list[None] = []

    def fake_ensure_model_ready(self) -> None:
        ensure_calls.append(None)

    monkeypatch.setattr(OllamaTextGenerator, "_ensure_model_ready", fake_ensure_model_ready)

    output = generator.generate(system_prompt="System prompt", user_prompt="User prompt")

    assert output == "Recovered answer."
    assert any("stopped mid-generation" in message for message in progress)
    assert len(ensure_calls) == 2


def test_ollama_text_generator_reports_gpu_state_once_after_success(app_home, monkeypatch) -> None:
    progress: list[str] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        @staticmethod
        def json() -> dict[str, object]:
            return {"message": {"content": "Visible answer."}}

    class FakeClient:
        def post(self, url: str, json: dict[str, object]) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(
        "arignan.llm.runtime.describe_running_models",
        lambda endpoint: ["qwen3:4b-q4_K_M (2.70 GiB VRAM)"],
    )
    monkeypatch.setattr(
        "arignan.llm.runtime.format_torch_cuda_memory",
        lambda label: f"{label}: torch cuda allocated=0.20 GiB, reserved=0.30 GiB, total=4.00 GiB",
    )

    generator = OllamaTextGenerator(AppConfig(app_home=app_home), progress_sink=progress.append)
    generator._model_ready = True
    generator._client = FakeClient()  # type: ignore[assignment]

    generator.generate(system_prompt="System prompt", user_prompt="User prompt")
    generator.generate(system_prompt="System prompt", user_prompt="User prompt")

    assert progress.count("Ollama GPU state: qwen3:4b-q4_K_M (2.70 GiB VRAM)") == 1
    assert sum("GPU after first LLM response" in message for message in progress) == 1


def test_ollama_text_generator_streams_thinking_and_answer_separately(app_home) -> None:
    thinking_chunks: list[str] = []
    answer_chunks: list[str] = []

    class FakeStreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self) -> None:
            return None

        @staticmethod
        def iter_lines():
            return iter(
                [
                    b'{"message":{"thinking":"Let me think."},"done":false}',
                    b'{"message":{"thinking":" More reasoning."},"done":false}',
                    b'{"message":{"content":"Final"},"done":false}',
                    b'{"message":{"content":" answer."},"done":false}',
                    b'{"done":true,"total_duration":2000000000,"eval_count":12}',
                ]
            )

    class FakeClient:
        def stream(self, method: str, url: str, json: dict[str, object]):
            assert method == "POST"
            assert json["stream"] is True
            assert json["think"] is True
            return FakeStreamResponse()

    generator = OllamaTextGenerator(AppConfig(app_home=app_home))
    generator._model_ready = True
    generator._client = FakeClient()  # type: ignore[assignment]
    generator.thinking_sink = thinking_chunks.append
    generator.stream_sink = answer_chunks.append

    output = generator.generate(system_prompt="System prompt", user_prompt="User prompt")

    assert output == "Final answer."
    assert "".join(thinking_chunks) == "Let me think. More reasoning."
    assert "".join(answer_chunks) == "Final answer."
    assert generator.last_thinking == "Let me think. More reasoning."
    assert generator.last_usage == {"total_duration": 2000000000, "eval_count": 12}


def test_ollama_text_generator_reports_empty_stream_after_thinking_with_detail(app_home, monkeypatch) -> None:
    class FakeStreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self) -> None:
            return None

        @staticmethod
        def iter_lines():
            return iter(
                [
                    b'{"message":{"thinking":"Let me think."},"done":false}',
                    b'{"done":true,"done_reason":"stop"}',
                ]
            )

    class FakeClient:
        def stream(self, method: str, url: str, json: dict[str, object]):
            return FakeStreamResponse()

    generator = OllamaTextGenerator(AppConfig(app_home=app_home))
    generator._model_ready = True
    generator._client = FakeClient()  # type: ignore[assignment]
    generator.thinking_sink = lambda text: None
    generator.stream_sink = lambda text: None
    ensure_calls: list[None] = []

    def fake_ensure_model_ready(self) -> None:
        ensure_calls.append(None)

    monkeypatch.setattr(OllamaTextGenerator, "_ensure_model_ready", fake_ensure_model_ready)

    try:
        generator.generate(system_prompt="System prompt", user_prompt="User prompt")
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected an empty-stream runtime error")

    assert "finished without answer text" in message
    assert "after producing thinking tokens" in message
    assert len(ensure_calls) == 2
    assert generator.last_failure_detail is not None
    assert "after producing thinking tokens" in generator.last_failure_detail
    assert generator._model_ready is False


def test_ollama_text_generator_retries_once_after_memory_pressure(app_home, monkeypatch) -> None:
    progress: list[str] = []
    release_calls: list[str] = []
    released_models: list[dict[str, object]] = []

    class RetryResponse:
        def __init__(self, *, text: str = "", ok: bool = False) -> None:
            self.text = text
            self._ok = ok

        def raise_for_status(self) -> None:
            if self._ok:
                return None
            request = httpx.Request("POST", "http://127.0.0.1:11434/api/chat")
            response = httpx.Response(500, request=request, text=self.text)
            raise httpx.HTTPStatusError("boom", request=request, response=response)

        @staticmethod
        def json() -> dict[str, object]:
            return {"message": {"content": "Recovered answer."}}

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, url: str, json: dict[str, object]):
            self.calls += 1
            if self.calls == 1:
                return RetryResponse(text='{"error":"model requires more system memory (7.1 GiB) than is available (4.7 GiB)"}')
            return RetryResponse(ok=True)

    monkeypatch.setattr(
        "arignan.llm.runtime.release_running_models",
        lambda endpoint, progress=None, exclude=None: released_models.append(
            {"endpoint": endpoint, "exclude": exclude}
        )
        or ["gemma4:e2b"],
    )

    generator = OllamaTextGenerator(
        AppConfig(app_home=app_home, local_llm_model="qwen3:4b-q4_K_M"),
        progress_sink=progress.append,
        memory_recovery=lambda reason: release_calls.append(reason) or True,
    )
    generator._model_ready = True
    generator._client = FakeClient()  # type: ignore[assignment]

    output = generator.generate(system_prompt="System prompt", user_prompt="User prompt")

    assert output == "Recovered answer."
    assert len(release_calls) == 1
    assert "requires more system memory" in release_calls[0]
    assert released_models == [
        {"endpoint": "http://127.0.0.1:11434", "exclude": {"qwen3:4b-q4_K_M"}}
    ]
    assert any("retrying once" in message for message in progress)


def test_ollama_text_generator_reports_gpu_state_once_after_success(app_home, monkeypatch) -> None:
    progress: list[str] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        @staticmethod
        def json() -> dict[str, object]:
            return {"message": {"content": "Visible answer."}}

    class FakeClient:
        def post(self, url: str, json: dict[str, object]) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(
        "arignan.llm.runtime.describe_running_models",
        lambda endpoint: ["qwen3:4b-q4_K_M (2.70 GiB VRAM)"],
    )
    monkeypatch.setattr(
        "arignan.llm.runtime.format_torch_cuda_memory",
        lambda label: f"{label}: torch cuda allocated=0.20 GiB, reserved=0.30 GiB, total=4.00 GiB",
    )

    generator = OllamaTextGenerator(AppConfig(app_home=app_home), progress_sink=progress.append)
    generator._model_ready = True
    generator._client = FakeClient()  # type: ignore[assignment]

    generator.generate(system_prompt="System prompt", user_prompt="User prompt")
    generator.generate(system_prompt="System prompt", user_prompt="User prompt")

    assert progress.count("Ollama GPU state: qwen3:4b-q4_K_M (2.70 GiB VRAM)") == 1
    assert sum("GPU after first LLM response" in message for message in progress) == 1


def test_transformers_text_generator_prefers_cuda_load_when_available(app_home, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeTokenizer:
        pad_token_id = None
        eos_token_id = 7

        @staticmethod
        def from_pretrained(source, **kwargs):
            captured["tokenizer_source"] = source
            captured["tokenizer_kwargs"] = kwargs
            return FakeTokenizer()

    class FakeModel:
        def to(self, device: str):
            captured["moved_to"] = device
            return self

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(source, **kwargs):
            captured["model_source"] = source
            captured["model_kwargs"] = kwargs
            return FakeModel()

    fake_module = types.SimpleNamespace(
        AutoTokenizer=FakeTokenizer,
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    monkeypatch.setattr("arignan.llm.runtime.preferred_torch_device", lambda: "cuda")
    local_dir = app_home / "models" / "Qwen__Qwen3-1.7B"
    local_dir.mkdir(parents=True, exist_ok=True)

    generator = TransformersTextGenerator(
        AppConfig(app_home=app_home, local_llm_backend="transformers", local_llm_model="Qwen/Qwen3-1.7B")
    )

    generator._ensure_loaded()

    assert captured["tokenizer_kwargs"]["local_files_only"] is True
    assert captured["model_kwargs"]["dtype"] == "auto"
    assert captured["moved_to"] == "cuda"


def test_transformers_text_generator_downloads_missing_local_model_with_warning(app_home, monkeypatch) -> None:
    captured: dict[str, object] = {}
    progress: list[str] = []

    class FakeTokenizer:
        pad_token_id = None
        eos_token_id = 7

        @staticmethod
        def from_pretrained(source, **kwargs):
            captured["tokenizer_source"] = source
            captured["tokenizer_kwargs"] = kwargs
            return FakeTokenizer()

    class FakeModel:
        def to(self, device: str):
            captured["moved_to"] = device
            return self

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(source, **kwargs):
            captured["model_source"] = source
            captured["model_kwargs"] = kwargs
            return FakeModel()

    fake_module = types.SimpleNamespace(
        AutoTokenizer=FakeTokenizer,
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    monkeypatch.setattr("arignan.llm.runtime.preferred_torch_device", lambda: "cpu")

    download_dir = app_home / "models" / "Qwen__Qwen3-1.7B"

    def fake_download(config, *, progress_sink=None):
        if progress_sink is not None:
            progress_sink("Configured local LLM model 'Qwen/Qwen3-1.7B' is not cached locally yet. Downloading it now...")
            progress_sink("Local LLM model 'Qwen/Qwen3-1.7B' is ready.")
        download_dir.mkdir(parents=True, exist_ok=True)
        return str(download_dir)

    monkeypatch.setattr("arignan.llm.runtime._download_transformers_model", fake_download)

    generator = TransformersTextGenerator(
        AppConfig(app_home=app_home, local_llm_backend="transformers", local_llm_model="Qwen/Qwen3-1.7B"),
        progress_sink=progress.append,
    )

    generator._ensure_loaded()

    assert captured["tokenizer_source"] == str(download_dir)
    assert captured["tokenizer_kwargs"]["local_files_only"] is True
    assert "Downloading it now" in progress[0]
    assert "is ready" in progress[-1]
