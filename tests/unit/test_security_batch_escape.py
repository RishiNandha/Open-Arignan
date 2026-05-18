"""Tests for Windows batch argument escaping in launchers (Fix #6)."""
from __future__ import annotations

import pytest

from arignan.setup_flow import _escape_batch_argument


class TestEscapeBatchArgument:
    def test_plain_path_unchanged(self) -> None:
        result = _escape_batch_argument(r"C:\Users\alice\.arignan")
        assert result == r"C:\Users\alice\.arignan"

    def test_double_quote_is_doubled(self) -> None:
        result = _escape_batch_argument('path with "quotes"')
        assert result == 'path with ""quotes""'

    def test_multiple_quotes_all_doubled(self) -> None:
        result = _escape_batch_argument('"a" "b"')
        assert result == '""a"" ""b""'

    def test_carriage_return_raises(self) -> None:
        with pytest.raises(ValueError, match="carriage return"):
            _escape_batch_argument("path\rwith\rcr")

    def test_newline_raises(self) -> None:
        with pytest.raises(ValueError, match="newline"):
            _escape_batch_argument("path\nwith\nnewline")

    def test_null_byte_raises(self) -> None:
        with pytest.raises(ValueError, match="null byte"):
            _escape_batch_argument("path\x00withnull")

    def test_unicode_path_passes_through(self) -> None:
        result = _escape_batch_argument("/home/ückermark/app")
        assert result == "/home/ückermark/app"

    def test_empty_string_unchanged(self) -> None:
        assert _escape_batch_argument("") == ""
