"""Tests for exception context sanitization to prevent path leaks (Fix #17)."""
from __future__ import annotations

from pathlib import Path

from arignan.markdown.writer import _sanitize_exception_context, _SENSITIVE_CONTEXT_KEYS


class TestSanitizeExceptionContext:
    def test_path_object_is_redacted(self) -> None:
        context = {"path": Path("/home/user/.arignan/data.json")}
        result = _sanitize_exception_context(context)
        assert result["path"] == "<redacted-path>"

    def test_non_sensitive_string_preserved(self) -> None:
        context = {"hat": "default", "model": "llama3", "count": 42}
        result = _sanitize_exception_context(context)
        assert result["hat"] == "default"
        assert result["model"] == "llama3"
        assert result["count"] == 42

    def test_sensitive_key_document_redacted(self) -> None:
        context = {"document": "full document text content here"}
        result = _sanitize_exception_context(context)
        assert result["document"] == "<redacted>"

    def test_sensitive_key_text_redacted(self) -> None:
        context = {"text": "some chunk text that should not leak"}
        result = _sanitize_exception_context(context)
        assert result["text"] == "<redacted>"

    def test_sensitive_key_content_redacted(self) -> None:
        context = {"content": "sensitive content"}
        result = _sanitize_exception_context(context)
        assert result["content"] == "<redacted>"

    def test_sensitive_key_chunk_redacted(self) -> None:
        context = {"chunk": "chunk data"}
        result = _sanitize_exception_context(context)
        assert result["chunk"] == "<redacted>"

    def test_sensitive_key_source_redacted(self) -> None:
        context = {"source": "document source text"}
        result = _sanitize_exception_context(context)
        assert result["source"] == "<redacted>"

    def test_mixed_context_redacts_only_sensitive(self) -> None:
        context = {
            "hat": "research",
            "path": Path("/private/home/docs.pdf"),
            "text": "document text",
            "task": "write_topic",
            "count": 5,
        }
        result = _sanitize_exception_context(context)
        assert result["hat"] == "research"
        assert result["path"] == "<redacted-path>"
        assert result["text"] == "<redacted>"
        assert result["task"] == "write_topic"
        assert result["count"] == 5

    def test_empty_context_returns_empty(self) -> None:
        assert _sanitize_exception_context({}) == {}

    def test_sensitive_context_keys_covers_expected_set(self) -> None:
        expected = {"document", "text", "content", "chunk", "path", "source"}
        assert expected.issubset(_SENSITIVE_CONTEXT_KEYS)
