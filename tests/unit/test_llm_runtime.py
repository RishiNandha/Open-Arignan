from __future__ import annotations

import importlib
import os

import arignan
import arignan.llm.runtime as llm_runtime
from arignan.config import AppConfig
from arignan.llm.runtime import OllamaTextGenerator, create_local_text_generator
from arignan.runtime_env import TEXT_RUNTIME_ENVIRONMENT, configure_text_runtime_environment


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
        max_new_tokens=256,
        temperature=0.0,
        response_format={"type": "object"},
    )

    assert output == "Visible answer."
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    payload = captured["json"]
    assert payload["model"] == "qwen3:4b-q4_K_M"
    assert payload["options"]["num_predict"] == 256
    assert payload["format"] == {"type": "object"}
    assert payload["think"] is False
