from __future__ import annotations

from pathlib import Path

import pytest

from arignan.ingestion.parsers import DocumentParser, FetchedUrl
from arignan.models import SourceDocument, SourceType


class FakeUrlFetcher:
    def fetch(self, url: str) -> FetchedUrl:
        return FetchedUrl(
            url=url,
            html="""
            <html>
              <head><title>Sample Article</title></head>
              <body>
                <h1>Overview</h1>
                <p>Local-first retrieval matters.</p>
                <h2>Details</h2>
                <p>Keyword and dense search complement each other.</p>
              </body>
            </html>
            """,
        )


class FakePdfPage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class FakePdfReader:
    def __init__(self, path: str) -> None:
        self.pages = [FakePdfPage("Page one"), FakePdfPage("Page two")]


def test_markdown_parser_extracts_sections_from_headings(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Title\n\nIntro\n\n## Details\n\nMore detail", encoding="utf-8")

    parser = DocumentParser()
    parsed = parser.parse(
        SourceDocument(source_type=SourceType.MARKDOWN, source_uri=str(path), local_path=path),
        load_id="load-1",
        hat="default",
    )

    assert parsed.source.title == "Title"
    assert [section.heading for section in parsed.sections] == ["Title", "Details"]


def test_pdf_parser_creates_page_sections(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr("arignan.ingestion.parsers.PdfReader", FakePdfReader)

    parser = DocumentParser()
    parsed = parser.parse(
        SourceDocument(source_type=SourceType.PDF, source_uri=str(path), local_path=path),
        load_id="load-2",
        hat="default",
    )

    assert parsed.source.title == "paper"
    assert [section.page_number for section in parsed.sections] == [1, 2]
    assert "Page one" in parsed.full_text


def test_url_parser_extracts_html_title_and_sections() -> None:
    parser = DocumentParser(url_fetcher=FakeUrlFetcher())
    parsed = parser.parse(
        SourceDocument(source_type=SourceType.URL, source_uri="https://example.com/post"),
        load_id="load-3",
        hat="default",
    )

    assert parsed.source.title == "Sample Article"
    assert [section.heading for section in parsed.sections] == ["Overview", "Details"]
    assert "dense search" in parsed.full_text
