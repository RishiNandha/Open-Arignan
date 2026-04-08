from __future__ import annotations

from pathlib import Path

import pytest

from arignan.ingestion import discover_sources, is_web_url
from arignan.models import SourceType


def test_is_web_url_detects_http_and_https() -> None:
    assert is_web_url("https://example.com/article")
    assert is_web_url("http://example.com")
    assert not is_web_url("ftp://example.com")
    assert not is_web_url("notes.md")


def test_discover_sources_for_markdown_fixture() -> None:
    source = discover_sources("tests/fixtures/markdown/sample_notes.md")[0]

    assert source.source_type is SourceType.MARKDOWN
    assert source.local_path is not None
    assert source.local_path.name == "sample_notes.md"


def test_discover_sources_for_directory(tmp_path: Path) -> None:
    (tmp_path / "one.md").write_text("# One", encoding="utf-8")
    (tmp_path / "two.pdf").write_text("fake", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")

    sources = discover_sources(tmp_path)

    assert [source.source_type for source in sources] == [SourceType.MARKDOWN, SourceType.PDF]


def test_discover_sources_rejects_empty_supported_directory(tmp_path: Path) -> None:
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError):
        discover_sources(tmp_path)
