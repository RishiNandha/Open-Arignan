from __future__ import annotations

from pathlib import Path

import pytest

from arignan.ingestion import FetchedUrl, IngestionLog, IngestionService


class FakeUrlFetcher:
    def fetch(self, url: str) -> FetchedUrl:
        return FetchedUrl(
            url=url,
            html="<html><head><title>Blog</title></head><body><h1>Intro</h1><p>Web text.</p></body></html>",
        )


class FakePdfPage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class FakePdfReader:
    def __init__(self, path: str) -> None:
        self.pages = [FakePdfPage("PDF content")]


def test_ingestion_service_ingests_markdown_and_appends_log(tmp_path: Path) -> None:
    markdown = tmp_path / "notes.md"
    markdown.write_text("# Notes\n\nBody", encoding="utf-8")
    log = IngestionLog(tmp_path / "ingestion_log.jsonl")
    service = IngestionService(log)

    batch = service.ingest(markdown, hat="default", load_id="load-md")

    assert batch.load_id == "load-md"
    assert len(batch.documents) == 1
    assert log.find_by_load_id("load-md")[0].source_items == [str(markdown.resolve())]


def test_ingestion_service_ingests_folder_of_supported_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "one.md").write_text("# One\n\nAlpha", encoding="utf-8")
    (folder / "two.pdf").write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr("arignan.ingestion.parsers.PdfReader", FakePdfReader)
    service = IngestionService(IngestionLog(tmp_path / "ingestion_log.jsonl"))

    batch = service.ingest(folder, hat="research", load_id="load-folder")

    assert {document.source.source_type.value for document in batch.documents} == {"markdown", "pdf"}
    assert len(batch.source_items) == 2


def test_ingestion_service_ingests_url_with_custom_fetcher(tmp_path: Path) -> None:
    service = IngestionService(IngestionLog(tmp_path / "ingestion_log.jsonl"), url_fetcher=FakeUrlFetcher())

    batch = service.ingest("https://example.com/post", hat="default", load_id="load-url")

    assert batch.documents[0].source.title == "Blog"
    assert batch.documents[0].full_text == "Web text."
