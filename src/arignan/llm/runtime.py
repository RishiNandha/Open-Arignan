from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any
from typing import Callable, Protocol

import httpx

from arignan.config import AppConfig
from arignan.llm.service import ensure_model_available
from arignan.model_registry import (
    infer_local_llm_backend,
    resolve_model_repo_id,
    resolve_model_storage_dir,
    resolve_ollama_model_id,
    sanitize_model_id,
)
from arignan.runtime_env import configure_text_runtime_environment


configure_text_runtime_environment()


class LocalTextGenerator(Protocol):
    model_name: str
    backend_name: str

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 800,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Generate text from a local model."""


@dataclass(slots=True)
class OllamaTextGenerator:
    config: AppConfig
    progress_sink: Callable[[str], None] | None = None
    model_name: str = field(init=False)
    backend_name: str = field(init=False)
    _client: httpx.Client | None = field(default=None, init=False, repr=False)
    _model_ready: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_name = resolve_ollama_model_id(self.config.local_llm_model)
        self.backend_name = "ollama-local"

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 800,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        if not self._model_ready:
            ensure_model_available(
                self.config.app_home,
                self.config.local_llm_endpoint,
                self.model_name,
                progress=self.progress_sink,
                timeout_seconds=float(max(self.config.local_llm_timeout_seconds, 1800)),
            )
            self._model_ready = True
        client = self._ensure_client()
        endpoint = self.config.local_llm_endpoint.rstrip("/") + "/api/chat"
        messages = _build_ollama_messages(self.model_name, system_prompt=system_prompt, user_prompt=user_prompt)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": self.config.local_llm_keep_alive,
            "options": {
                "temperature": max(temperature, 0.0),
                "num_predict": max_new_tokens,
            },
        }
        if response_format is not None:
            payload["format"] = response_format
        if _supports_no_think(self.model_name):
            payload["think"] = False
        try:
            response = client.post(endpoint, json=payload)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"failed to reach the local model runtime at {self.config.local_llm_endpoint}. "
                "Re-run setup if the managed runtime was not provisioned correctly."
            ) from exc
        except httpx.HTTPError as exc:
            details = exc.response.text if exc.response is not None else str(exc)
            raise RuntimeError(f"local model runtime request failed for model {self.model_name}: {details}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("Ollama returned a non-JSON response") from exc
        content = payload.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Ollama returned an empty response for model {self.model_name}")
        return _strip_think_blocks(content).strip()

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.config.local_llm_timeout_seconds)
        return self._client


@dataclass(slots=True)
class TransformersTextGenerator:
    config: AppConfig
    model_name: str = field(init=False)
    backend_name: str = field(init=False)
    _tokenizer: object | None = None
    _model: object | None = None

    def __post_init__(self) -> None:
        self.model_name = self.config.local_llm_model
        self.backend_name = "transformers-local"

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 800,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        tokenizer, model = self._ensure_loaded()
        prompt = self._build_prompt(tokenizer, system_prompt=system_prompt, user_prompt=user_prompt)
        model_inputs, prompt_length = self._prepare_inputs(tokenizer, model, prompt)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
        }
        if temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
        else:
            generation_kwargs["do_sample"] = False

        output_ids = model.generate(**model_inputs, **generation_kwargs)
        generated_ids = output_ids[0][prompt_length:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _ensure_loaded(self):
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        configure_text_runtime_environment()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("transformers is required for local LLM generation") from exc

        source = resolve_local_model_source(self.config)
        load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
        }
        tokenizer = AutoTokenizer.from_pretrained(source, **load_kwargs)
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = None
        model_load_attempts = [
            {"torch_dtype": "auto", "device_map": "auto", "low_cpu_mem_usage": True},
            {"torch_dtype": "auto", "low_cpu_mem_usage": True},
            {},
        ]
        last_error: Exception | None = None
        for extra_kwargs in model_load_attempts:
            try:
                model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs, **extra_kwargs)
                break
            except Exception as exc:  # pragma: no cover - depends on local runtime
                last_error = exc
        if model is None:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"failed to load local LLM model from {source}: {last_error}") from last_error

        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model

    @staticmethod
    def _build_prompt(tokenizer, *, system_prompt: str, user_prompt: str):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"

    @staticmethod
    def _prepare_inputs(tokenizer, model, prompt: str):
        encoded = tokenizer(prompt, return_tensors="pt")
        device = _model_device(model)
        model_inputs = {name: value.to(device) for name, value in encoded.items()}
        prompt_length = model_inputs["input_ids"].shape[1]
        return model_inputs, prompt_length


def create_local_text_generator(
    config: AppConfig,
    progress_sink=None,
    *,
    model_name: str | None = None,
    backend: str | None = None,
) -> LocalTextGenerator:
    selected_model = model_name or config.local_llm_model
    selected_backend = (backend or infer_local_llm_backend(selected_model, default=config.local_llm_backend)).strip().lower()
    generator_config = replace(
        config,
        local_llm_backend=selected_backend,
        local_llm_model=selected_model,
    )
    backend = generator_config.local_llm_backend.strip().lower()
    if backend == "ollama":
        return OllamaTextGenerator(generator_config, progress_sink=progress_sink)
    if backend in {"transformers", "huggingface"}:
        return TransformersTextGenerator(generator_config)
    raise ValueError(f"unsupported local_llm_backend: {generator_config.local_llm_backend}")


def resolve_local_model_source(config: AppConfig) -> str:
    repo_id = resolve_model_repo_id(config.local_llm_model)
    model_dir = resolve_model_storage_dir(config.app_home, config.local_llm_model)
    if model_dir.exists():
        return str(model_dir)
    return repo_id


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return "cpu"


def _build_ollama_messages(model_name: str, *, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    system_content = system_prompt.strip()
    if _supports_no_think(model_name):
        system_content = "/no_think\n" + system_content
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt},
    ]


def _supports_no_think(model_name: str) -> bool:
    normalized = model_name.lower()
    return normalized.startswith("qwen3")


def _strip_think_blocks(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("<think>\n", "<think>").replace("\n</think>", "</think>")
    cleaned = cleaned.replace("<thinking>\n", "<thinking>").replace("\n</thinking>", "</thinking>")
    import re

    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()
