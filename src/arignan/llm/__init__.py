"""Local LLM runtime helpers."""

from __future__ import annotations

from typing import Any

__all__ = ["LocalTextGenerator", "OllamaTextGenerator", "TransformersTextGenerator", "create_local_text_generator"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from arignan.llm.runtime import (
            LocalTextGenerator,
            OllamaTextGenerator,
            TransformersTextGenerator,
            create_local_text_generator,
        )

        exports = {
            "LocalTextGenerator": LocalTextGenerator,
            "OllamaTextGenerator": OllamaTextGenerator,
            "TransformersTextGenerator": TransformersTextGenerator,
            "create_local_text_generator": create_local_text_generator,
        }
        return exports[name]
    raise AttributeError(name)
