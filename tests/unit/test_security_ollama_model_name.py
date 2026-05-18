"""Tests for Ollama model name validation (Fix #9)."""
from __future__ import annotations

import pytest

from arignan.llm.service import _validate_ollama_model_name


class TestValidateOllamaModelName:
    def test_simple_valid_name(self) -> None:
        _validate_ollama_model_name("llama3")

    def test_name_with_tag(self) -> None:
        _validate_ollama_model_name("qwen:4b")

    def test_name_with_namespace_and_tag(self) -> None:
        _validate_ollama_model_name("myorg/mymodel:latest")

    def test_name_with_dots(self) -> None:
        _validate_ollama_model_name("llama3.1:8b-instruct-q4_K_M")

    def test_semicolon_injection_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid Ollama model name"):
            _validate_ollama_model_name("qwen;rm -rf /")

    def test_shell_pipe_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid Ollama model name"):
            _validate_ollama_model_name("qwen|cat /etc/passwd")

    def test_spaces_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid Ollama model name"):
            _validate_ollama_model_name("llama 3")

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid Ollama model name"):
            _validate_ollama_model_name("")

    def test_ampersand_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid Ollama model name"):
            _validate_ollama_model_name("llama3 & evil")

    def test_excessively_long_name_rejected(self) -> None:
        long_name = "a" * 201
        with pytest.raises(ValueError, match="Invalid Ollama model name"):
            _validate_ollama_model_name(long_name)

    def test_max_length_boundary_passes(self) -> None:
        valid = "a" + "b" * 199
        _validate_ollama_model_name(valid)
