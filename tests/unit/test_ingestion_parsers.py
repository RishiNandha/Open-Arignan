from __future__ import annotations

from pathlib import Path

import pytest

from arignan.ingestion.parsers import DocumentParser, FetchedUrl, PdfOcrRequired
from arignan.models import ParsedDocument, SourceDocument, SourceType


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


class EmptyPdfReader:
    def __init__(self, path: str) -> None:
        self.pages = [FakePdfPage(""), FakePdfPage("")]


class FakePdfOcrEngine:
    def __init__(self, pages: dict[int, str] | None = None, error: str | None = None) -> None:
        self.pages = pages or {}
        self.error = error

    def extract_page_text(self, pdf_path: Path, page_index: int) -> str:
        if self.error is not None:
            raise RuntimeError(self.error)
        return self.pages.get(page_index, "")


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


def test_pdf_parser_falls_back_to_ocr_for_image_only_pages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr("arignan.ingestion.parsers.PdfReader", EmptyPdfReader)

    parser = DocumentParser(pdf_ocr_engine=FakePdfOcrEngine({0: "Scanned page one", 1: "Scanned page two"}))
    parsed = parser.parse(
        SourceDocument(source_type=SourceType.PDF, source_uri=str(path), local_path=path),
        load_id="load-ocr",
        hat="default",
    )

    assert parsed.source.metadata["parser"] == "pdf+ocr"
    assert "Scanned page one" in parsed.full_text
    assert [section.page_number for section in parsed.sections] == [1, 2]


def test_pdf_parser_can_signal_that_ocr_should_be_deferred(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr("arignan.ingestion.parsers.PdfReader", EmptyPdfReader)

    parser = DocumentParser(pdf_ocr_engine=FakePdfOcrEngine({0: "Scanned page one"}))

    with pytest.raises(PdfOcrRequired):
        parser.parse(
            SourceDocument(source_type=SourceType.PDF, source_uri=str(path), local_path=path),
            load_id="load-ocr-defer",
            hat="default",
            allow_ocr=False,
        )


def test_pdf_parser_raises_clear_error_when_pdf_is_image_only_and_ocr_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr("arignan.ingestion.parsers.PdfReader", EmptyPdfReader)

    parser = DocumentParser(pdf_ocr_engine=FakePdfOcrEngine(error="OCR unavailable"))

    with pytest.raises(ValueError) as exc_info:
        parser.parse(
            SourceDocument(source_type=SourceType.PDF, source_uri=str(path), local_path=path),
            load_id="load-ocr-fail",
            hat="default",
        )

    message = str(exc_info.value)
    assert "image-only" in message
    assert "OCR fallback did not succeed" in message
    assert "OCR unavailable" in message


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
