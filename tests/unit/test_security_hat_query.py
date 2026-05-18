"""Tests for hat name validation and query length caps (Fixes #7 and #11)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from arignan.retrieval.pipeline import QueryExpander, HatSelector, _MAX_QUERY_INPUT_CHARS, _MAX_QUERY_EXPANDED_CHARS
from arignan.storage.layout import StorageLayout


# ---------------------------------------------------------------------------
# Fix #11 — Unbounded query length
# ---------------------------------------------------------------------------

class TestQueryExpanderLengthCap:
    def test_normal_query_unchanged_by_cap(self) -> None:
        expander = QueryExpander()
        result = expander.expand("what is JEPA?")
        assert len(result) > 0

    def test_very_long_query_is_truncated_for_input(self) -> None:
        expander = QueryExpander()
        long_query = "a " * 10_000
        result = expander.expand(long_query)
        assert len(result) <= _MAX_QUERY_EXPANDED_CHARS

    def test_adversarial_10k_char_query_does_not_explode(self) -> None:
        expander = QueryExpander()
        adversarial = "x" * 10_000
        result = expander.expand(adversarial)
        assert isinstance(result, str)
        assert len(result) <= _MAX_QUERY_EXPANDED_CHARS

    def test_expanded_result_capped_at_max(self) -> None:
        expander = QueryExpander()
        query = "bm25 " * 2000
        result = expander.expand(query)
        assert len(result) <= _MAX_QUERY_EXPANDED_CHARS

    def test_constants_are_sane(self) -> None:
        assert _MAX_QUERY_INPUT_CHARS == 4_096
        assert _MAX_QUERY_EXPANDED_CHARS == 8_192


# ---------------------------------------------------------------------------
# Fix #7 — Hat name validation
# ---------------------------------------------------------------------------

def _make_layout_with_hats(tmp_path: Path, hats: list[str]) -> StorageLayout:
    layout = StorageLayout(
        root=tmp_path,
        settings_path=tmp_path / "settings.json",
        ingestion_log_path=tmp_path / "ingestion.jsonl",
        hats_dir=tmp_path / "hats",
        global_map_path=tmp_path / "hats" / "global_map.md",
    )
    for hat in hats:
        (tmp_path / "hats" / hat).mkdir(parents=True, exist_ok=True)
    return layout


class TestHatSelectorValidation:
    def test_valid_hat_name_passes(self, tmp_path: Path) -> None:
        layout = _make_layout_with_hats(tmp_path, ["myhat"])
        selector = HatSelector(layout)
        result = selector.select("some query", hat="myhat")
        assert result == "myhat"

    def test_path_traversal_dot_dot_rejected(self, tmp_path: Path) -> None:
        layout = _make_layout_with_hats(tmp_path, [])
        selector = HatSelector(layout)
        with pytest.raises(ValueError):
            selector.select("query", hat="../secret")

    def test_path_separator_rejected(self, tmp_path: Path) -> None:
        layout = _make_layout_with_hats(tmp_path, [])
        selector = HatSelector(layout)
        with pytest.raises(ValueError):
            selector.select("query", hat="foo/bar")

    def test_nonexistent_hat_raises_with_clear_message(self, tmp_path: Path) -> None:
        layout = _make_layout_with_hats(tmp_path, ["real"])
        selector = HatSelector(layout)
        with pytest.raises(ValueError, match="does not exist"):
            selector.select("query", hat="nonexistent")

    def test_auto_hat_bypasses_validation(self, tmp_path: Path) -> None:
        layout = _make_layout_with_hats(tmp_path, ["only"])
        selector = HatSelector(layout)
        result = selector.select("query", hat="auto")
        assert result == "only"
