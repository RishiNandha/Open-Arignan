"""Tests for markdown injection prevention in citation output (Fix #12)."""
from __future__ import annotations

import pytest

from arignan.markdown.rendering import _escape_code_span, markdown_table_cell


class TestEscapeCodeSpan:
    def test_normal_filename_unchanged(self) -> None:
        assert _escape_code_span("report.pdf") == "report.pdf"

    def test_backtick_stripped(self) -> None:
        result = _escape_code_span("file`with`backticks.md")
        assert "`" not in result

    def test_multiple_backticks_all_stripped(self) -> None:
        result = _escape_code_span("a`b`c`d")
        assert "`" not in result
        assert "abcd" in result

    def test_adversarial_code_span_breakout(self) -> None:
        adversarial = "` evil` [link](http://evil.com) `"
        result = _escape_code_span(adversarial)
        assert "`" not in result


class TestMarkdownTableCell:
    def test_plain_text_unchanged(self) -> None:
        assert markdown_table_cell("normal text") == "normal text"

    def test_pipe_character_escaped(self) -> None:
        result = markdown_table_cell("a | b")
        assert "\\|" in result
        assert result.count("|") == 1

    def test_bracket_escaped_to_prevent_link_injection(self) -> None:
        adversarial = "[click me](http://evil.com)"
        result = markdown_table_cell(adversarial)
        assert "\\[" in result
        assert "](http://evil.com)" in result  # link target preserved but not rendered

    def test_html_bracket_escaped(self) -> None:
        result = markdown_table_cell("<script>alert(1)</script>")
        assert "<script>" not in result
        assert "&lt;" in result

    def test_newline_replaced_with_space(self) -> None:
        result = markdown_table_cell("line1\nline2")
        assert "\n" not in result

    def test_image_injection_via_bracket_is_escaped(self) -> None:
        adversarial = "![evil](http://attacker.com/img.png)"
        result = markdown_table_cell(adversarial)
        assert "\\[" in result
