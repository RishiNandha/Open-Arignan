from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from arignan.config import AppConfig
from arignan.model_registry import resolve_model_repo_id, resolve_model_storage_dir, sanitize_model_id


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
    ) -> str:
        """Generate text from a local model."""


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
