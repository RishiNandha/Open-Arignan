from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from typing import Callable, Protocol
import time

import httpx

from arignan.compute import format_torch_cuda_memory, preferred_torch_device
from arignan.config import AppConfig
from arignan.llm.service import describe_running_models, ensure_model_available, release_running_models
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
    stream_sink: Callable[[str], None] | None
    thinking_sink: Callable[[str], None] | None

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        chat_messages: list[dict[str, str]] | None = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Generate text from a local model."""


@dataclass(slots=True)
class OllamaTextGenerator:
    config: AppConfig
    progress_sink: Callable[[str], None] | None = None
    stream_sink: Callable[[str], None] | None = None
    thinking_sink: Callable[[str], None] | None = None
    memory_recovery: Callable[[str], bool] | None = None
    model_name: str = field(init=False)
    backend_name: str = field(init=False)
    _client: httpx.Client | None = field(default=None, init=False, repr=False)
    _model_ready: bool = field(default=False, init=False, repr=False)
    _gpu_reported: bool = field(default=False, init=False, repr=False)
    last_thinking: str = field(default="", init=False, repr=False)
    last_usage: dict[str, Any] | None = field(default=None, init=False, repr=False)
    last_failure_detail: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_name = resolve_ollama_model_id(self.config.local_llm_model)
        self.backend_name = "ollama-local"

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        chat_messages: list[dict[str, str]] | None = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        client = self._ensure_client()
        endpoint = self.config.local_llm_endpoint.rstrip("/") + "/api/chat"
        want_thinking_stream = self.thinking_sink is not None and response_format is None and _supports_thinking(self.model_name)
        disable_thinking = not want_thinking_stream and _supports_no_think(self.model_name)
        messages = _build_ollama_messages(
            self.model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            disable_thinking=disable_thinking,
            chat_messages=chat_messages,
        )
        use_stream = (self.stream_sink is not None or want_thinking_stream) and response_format is None
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": use_stream,
            "keep_alive": self.config.local_llm_keep_alive,
            "options": {
                "temperature": max(temperature, 0.0),
                "num_predict": max_new_tokens,
                "num_ctx": self.config.local_llm_context_window,
            },
        }
        stream_content = ""
        stream_error_detail: str | None = None
        self.last_thinking = ""
        self.last_usage = None
        self.last_failure_detail = None
        if response_format is not None:
            payload["format"] = response_format
        if want_thinking_stream:
            payload["think"] = True
        elif _supports_no_think(self.model_name):
            payload["think"] = False
        for attempt in range(2):
            try:
                self._ensure_model_ready()
                if use_stream:
                    with client.stream("POST", endpoint, json=payload) as response:
                        response.raise_for_status()
                        stream_content, stream_error_detail = self._collect_streamed_content(response)
                    content = stream_content
                else:
                    response = client.post(endpoint, json=payload)
                    response.raise_for_status()
                    try:
                        response_payload = response.json()
                    except ValueError as exc:
                        detail = "Ollama returned a non-JSON response"
                        if attempt == 0 and self._prepare_runtime_retry(detail):
                            time.sleep(0.5)
                            continue
                        raise RuntimeError(detail) from exc
                    content = response_payload.get("message", {}).get("content")
                if not isinstance(content, str) or not content.strip():
                    detail = stream_error_detail or "stream completed without answer text"
                    if self.last_thinking.strip():
                        detail += " after producing thinking tokens"
                    if attempt == 0 and self._prepare_runtime_retry(detail):
                        time.sleep(0.5)
                        continue
                    self._model_ready = False
                    self.last_failure_detail = detail
                    raise RuntimeError(f"Ollama finished without answer text for model {self.model_name}: {detail}")
                self._report_gpu_state_once()
                return _strip_think_blocks(content).strip()
            except httpx.ConnectError as exc:
                detail = str(exc)
                if attempt == 0 and self._prepare_runtime_retry(detail):
                    time.sleep(0.5)
                    continue
                self._model_ready = False
                self.last_failure_detail = detail
                raise RuntimeError(
                    f"failed to reach the local model runtime at {self.config.local_llm_endpoint}. "
                    "Re-run setup if the managed runtime was not provisioned correctly."
                ) from exc
            except httpx.HTTPError as exc:
                details = exc.response.text if exc.response is not None else str(exc)
                if attempt == 0 and self._recover_from_memory_pressure(details):
                    time.sleep(0.5)
                    continue
                if attempt == 0 and _is_retryable_runtime_http_error(exc, details) and self._prepare_runtime_retry(details):
                    time.sleep(0.5)
                    continue
                self._model_ready = False
                self.last_failure_detail = details
                raise RuntimeError(f"local model runtime request failed for model {self.model_name}: {details}") from exc

        raise RuntimeError(f"local model runtime request failed for model {self.model_name}: unknown runtime error")

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.config.local_llm_timeout_seconds)
        return self._client

    def _ensure_model_ready(self) -> None:
        if self._model_ready:
            return
        ensure_model_available(
            self.config.app_home,
            self.config.local_llm_endpoint,
            self.model_name,
            progress=self.progress_sink,
            timeout_seconds=float(max(self.config.local_llm_timeout_seconds, 1800)),
            context_window=self.config.local_llm_context_window,
            flash_attention=self.config.local_llm_flash_attention,
            kv_cache_type=self.config.local_llm_kv_cache_type,
            num_parallel=self.config.local_llm_num_parallel,
            max_loaded_models=self.config.local_llm_max_loaded_models,
        )
        self._model_ready = True

    def _prepare_runtime_retry(self, detail: str) -> bool:
        self._model_ready = False
        self.last_failure_detail = detail
        self._emit_progress(
            f"Local model '{self.model_name}' stopped mid-generation; re-checking the Ollama runtime and retrying once..."
        )
        return True

    def _recover_from_memory_pressure(self, details: str) -> bool:
        if not _is_ollama_memory_pressure(details):
            return False
        self._emit_progress(
            f"Local model '{self.model_name}' hit a memory limit; releasing CUDA retrieval state and retrying once..."
        )
        recovered = False
        if self.memory_recovery is not None and self.memory_recovery(details):
            recovered = True
        try:
            released_models = release_running_models(
                self.config.local_llm_endpoint,
                progress=self.progress_sink,
                exclude={self.model_name},
            )
        except httpx.HTTPError:
            released_models = []
        if released_models:
            recovered = True
        return recovered

    def _emit_progress(self, message: str) -> None:
        if self.progress_sink is not None:
            self.progress_sink(message)

    def _report_gpu_state_once(self) -> None:
        if self._gpu_reported:
            return
        self._gpu_reported = True
        descriptions: list[str] = []
        try:
            descriptions = describe_running_models(self.config.local_llm_endpoint)
        except httpx.HTTPError:
            descriptions = []
        if descriptions:
            self._emit_progress("Ollama GPU state: " + ", ".join(descriptions))
        message = format_torch_cuda_memory(f"GPU after first LLM response ({self.model_name})")
        if message:
            self._emit_progress(message)

    def _collect_streamed_content(self, response: httpx.Response) -> tuple[str, str | None]:
        parts: list[str] = []
        thinking_parts: list[str] = []
        usage: dict[str, Any] | None = None
        error_detail: str | None = None
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            try:
                payload = httpx.Response(200, content=raw_line).json()
            except ValueError:
                continue
            message = payload.get("message", {})
            thinking = message.get("thinking")
            if isinstance(thinking, str) and thinking:
                thinking_parts.append(thinking)
                if self.thinking_sink is not None:
                    self.thinking_sink(thinking)
            content = payload.get("message", {}).get("content")
            if isinstance(content, str) and content:
                parts.append(content)
                if self.stream_sink is not None:
                    self.stream_sink(content)
            error = payload.get("error")
            if isinstance(error, str) and error.strip():
                error_detail = error.strip()
            if payload.get("done"):
                usage = {
                    key: payload.get(key)
                    for key in (
                        "total_duration",
                        "load_duration",
                        "prompt_eval_count",
                        "prompt_eval_duration",
                        "eval_count",
                        "eval_duration",
                    )
                    if payload.get(key) is not None
                }
        self.last_thinking = "".join(thinking_parts)
        self.last_usage = usage
        return "".join(parts), error_detail


@dataclass(slots=True)
class TransformersTextGenerator:
    config: AppConfig
    progress_sink: Callable[[str], None] | None = None
    stream_sink: Callable[[str], None] | None = None
    thinking_sink: Callable[[str], None] | None = None
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
        chat_messages: list[dict[str, str]] | None = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        tokenizer, model = self._ensure_loaded()
        prompt = self._build_prompt(
            tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            chat_messages=chat_messages,
        )
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

        source, local_files_only = _resolve_transformers_model_source(self.config)
        if not local_files_only:
            source = _download_transformers_model(self.config, progress_sink=self.progress_sink)
            local_files_only = True
        load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        tokenizer = AutoTokenizer.from_pretrained(source, **load_kwargs)
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = None
        preferred_device = preferred_torch_device()
        model_load_attempts = []
        if preferred_device == "cuda":
            model_load_attempts.append(
                {
                    "dtype": "auto",
                    "low_cpu_mem_usage": True,
                    "__move_to_device__": "cuda",
                }
            )
            model_load_attempts.append({"dtype": "auto", "device_map": "auto", "low_cpu_mem_usage": True})
        model_load_attempts.extend(
            [
                {"dtype": "auto", "low_cpu_mem_usage": True},
                {},
            ]
        )
        last_error: Exception | None = None
        for extra_kwargs in model_load_attempts:
            try:
                move_to_device = extra_kwargs.pop("__move_to_device__", None)
                model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs, **extra_kwargs)
                if move_to_device is not None and hasattr(model, "to"):
                    model = model.to(move_to_device)
                break
            except Exception as exc:  # pragma: no cover - depends on local runtime
                last_error = exc
        if model is None:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"failed to load local LLM model from {source}: {last_error}") from last_error

        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model

    @staticmethod
    def _build_prompt(tokenizer, *, system_prompt: str, user_prompt: str, chat_messages: list[dict[str, str]] | None = None):
        messages = [{"role": "system", "content": system_prompt}]
        if chat_messages:
            messages.extend(chat_messages)
        messages.append({"role": "user", "content": user_prompt})
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
        return TransformersTextGenerator(generator_config, progress_sink=progress_sink)
    raise ValueError(f"unsupported local_llm_backend: {generator_config.local_llm_backend}")


def resolve_local_model_source(config: AppConfig) -> str:
    repo_id = resolve_model_repo_id(config.local_llm_model)
    model_dir = resolve_model_storage_dir(config.app_home, config.local_llm_model)
    if model_dir.exists():
        return str(model_dir)
    return repo_id


def _resolve_transformers_model_source(config: AppConfig) -> tuple[str, bool]:
    repo_id = resolve_model_repo_id(config.local_llm_model)
    model_dir = resolve_model_storage_dir(config.app_home, config.local_llm_model)
    if model_dir.exists():
        return str(model_dir), True
    return repo_id, False


def _download_transformers_model(
    config: AppConfig,
    *,
    progress_sink: Callable[[str], None] | None = None,
) -> str:
    repo_id = resolve_model_repo_id(config.local_llm_model)
    model_dir = resolve_model_storage_dir(config.app_home, config.local_llm_model)
    if progress_sink is not None:
        progress_sink(
            f"Configured local LLM model '{config.local_llm_model}' is not cached locally yet. Downloading it now..."
        )
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required for local transformers model downloads") from exc

    try:
        snapshot_download(repo_id=repo_id, local_dir=model_dir, local_dir_use_symlinks=False)
    except (RepositoryNotFoundError, GatedRepoError, HfHubHTTPError) as exc:
        raise RuntimeError(
            f"Failed to download local LLM model '{config.local_llm_model}' into {model_dir}: {exc}"
        ) from exc
    if progress_sink is not None:
        progress_sink(f"Local LLM model '{config.local_llm_model}' is ready.")
    return str(model_dir)


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return "cpu"


def _build_ollama_messages(
    model_name: str,
    *,
    system_prompt: str,
    user_prompt: str,
    disable_thinking: bool,
    chat_messages: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    system_content = system_prompt.strip()
    if disable_thinking and _supports_no_think(model_name):
        system_content = "/no_think\n" + system_content
    messages = [{"role": "system", "content": system_content}]
    if chat_messages:
        for message in chat_messages:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if role not in {"user", "assistant", "system"} or not content:
                continue
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _supports_no_think(model_name: str) -> bool:
    normalized = model_name.lower()
    return normalized.startswith("qwen3")


def _supports_thinking(model_name: str) -> bool:
    normalized = model_name.lower()
    return normalized.startswith(("qwen3", "deepseek-r1", "deepseek-v3.1", "gpt-oss"))


def _strip_think_blocks(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("<think>\n", "<think>").replace("\n</think>", "</think>")
    cleaned = cleaned.replace("<thinking>\n", "<thinking>").replace("\n</thinking>", "</thinking>")
    import re

    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _is_ollama_memory_pressure(details: str) -> bool:
    normalized = details.lower()
    return (
        "requires more system memory" in normalized
        or "out of memory" in normalized
        or "insufficient memory" in normalized
        or "cuda out of memory" in normalized
    )


def _is_retryable_runtime_http_error(exc: httpx.HTTPError, details: str) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is not None and int(status_code) >= 500:
        return True
    normalized = details.lower()
    return (
        "connection reset" in normalized
        or "connection refused" in normalized
        or "broken pipe" in normalized
        or "unexpected eof" in normalized
        or "eof" in normalized
    )
