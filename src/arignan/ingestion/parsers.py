from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol

import httpx
from pypdf import PdfReader

from arignan.models import DocumentSection, ParsedDocument, SourceDocument, SourceType


@dataclass(slots=True)
class FetchedUrl:
    url: str
    html: str
    title: str | None = None


class UrlFetcher(Protocol):
    def fetch(self, url: str) -> FetchedUrl:
        """Fetch a URL and return HTML content."""


class HttpUrlFetcher:
    def __init__(self, timeout_seconds: float = 20.0) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch(self, url: str) -> FetchedUrl:
        response = httpx.get(url, follow_redirects=True, timeout=self.timeout_seconds)
        response.raise_for_status()
        return FetchedUrl(url=str(response.url), html=response.text)


class _HtmlSectionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._ignore_depth = 0
        self._in_title = False
        self._in_heading = False
        self._heading_parts: list[str] = []
        self._pending_heading: str | None = None
        self._buffer: list[str] = []
        self._sections: list[DocumentSection] = []
        self.title: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._ignore_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag in {"h1", "h2", "h3"}:
            self._flush_buffer()
            self._in_heading = True
            self._heading_parts = []
        if tag in {"p", "div", "li", "section", "article", "br"}:
            self._buffer.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignore_depth:
            self._ignore_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        if tag in {"h1", "h2", "h3"} and self._in_heading:
            heading = " ".join(part.strip() for part in self._heading_parts if part.strip()).strip()
            self._pending_heading = heading or None
            self._heading_parts = []
            self._in_heading = False
        if tag in {"p", "div", "li", "section", "article"}:
            self._buffer.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignore_depth:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self.title = text if self.title is None else f"{self.title} {text}"
            return
        if self._in_heading:
            self._heading_parts.append(text)
            return
        self._buffer.append(text)

    def result(self) -> tuple[str, list[DocumentSection], str | None]:
        self._flush_buffer()
        if not self._sections:
            text = " ".join(part.strip() for part in self._buffer if part.strip()).strip()
            if text:
                self._sections.append(DocumentSection(text=text))
        full_text = "\n\n".join(section.text for section in self._sections)
        return full_text, self._sections, self.title

    def _flush_buffer(self) -> None:
        text = " ".join(part.strip() for part in self._buffer if part.strip()).strip()
        if text:
            self._sections.append(DocumentSection(text=text, heading=self._pending_heading))
        self._buffer = []
        self._pending_heading = None


class DocumentParser:
    def __init__(self, url_fetcher: UrlFetcher | None = None) -> None:
        self.url_fetcher = url_fetcher or HttpUrlFetcher()

    def parse(self, source: SourceDocument, load_id: str, hat: str) -> ParsedDocument:
        if source.source_type == SourceType.MARKDOWN:
            return self._parse_markdown(source, load_id, hat)
        if source.source_type == SourceType.PDF:
            return self._parse_pdf(source, load_id, hat)
        if source.source_type == SourceType.URL:
            return self._parse_url(source, load_id, hat)
        raise ValueError(f"unsupported source type: {source.source_type}")

    def _parse_markdown(self, source: SourceDocument, load_id: str, hat: str) -> ParsedDocument:
        if source.local_path is None:
            raise ValueError("markdown source must have local_path")
        text = source.local_path.read_text(encoding="utf-8")
        sections = _parse_markdown_sections(text)
        title = source.title or _first_heading(sections) or source.local_path.stem
        return ParsedDocument(
            load_id=load_id,
            hat=hat,
            source=SourceDocument(
                source_type=source.source_type,
                source_uri=source.source_uri,
                local_path=source.local_path,
                title=title,
                metadata={"parser": "markdown"},
            ),
            full_text=text.strip(),
            sections=sections,
            keywords=[],
        )

    def _parse_pdf(self, source: SourceDocument, load_id: str, hat: str) -> ParsedDocument:
        if source.local_path is None:
            raise ValueError("pdf source must have local_path")
        reader = PdfReader(str(source.local_path))
        sections: list[DocumentSection] = []
        page_texts: list[str] = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            page_texts.append(text)
            sections.append(DocumentSection(text=text, page_number=index, heading=f"Page {index}"))
        full_text = "\n\n".join(page_texts).strip()
        if not full_text:
            raise ValueError(f"no extractable text found in pdf: {source.local_path}")
        return ParsedDocument(
            load_id=load_id,
            hat=hat,
            source=SourceDocument(
                source_type=source.source_type,
                source_uri=source.source_uri,
                local_path=source.local_path,
                title=source.title or source.local_path.stem,
                metadata={"parser": "pdf"},
            ),
            full_text=full_text,
            sections=sections,
            keywords=[],
        )

    def _parse_url(self, source: SourceDocument, load_id: str, hat: str) -> ParsedDocument:
        fetched = self.url_fetcher.fetch(source.source_uri)
        parser = _HtmlSectionParser()
        parser.feed(fetched.html)
        full_text, sections, title = parser.result()
        if not full_text:
            raise ValueError(f"no readable text extracted from url: {source.source_uri}")
        return ParsedDocument(
            load_id=load_id,
            hat=hat,
            source=SourceDocument(
                source_type=source.source_type,
                source_uri=fetched.url,
                local_path=None,
                title=title or source.title or fetched.url,
                metadata={"parser": "html"},
            ),
            full_text=full_text,
            sections=sections,
            keywords=[],
        )


def _parse_markdown_sections(text: str) -> list[DocumentSection]:
    lines = text.splitlines()
    sections: list[DocumentSection] = []
    current_heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        section_text = "\n".join(buffer).strip()
        if section_text:
            sections.append(DocumentSection(text=section_text, heading=current_heading))

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            marker, _, rest = stripped.partition(" ")
            if marker and set(marker) == {"#"} and rest:
                flush()
                current_heading = rest.strip()
                buffer = []
                continue
        buffer.append(line)
    flush()

    if sections:
        return sections
    plain_text = text.strip()
    if not plain_text:
        raise ValueError("markdown document is empty")
    return [DocumentSection(text=plain_text)]


def _first_heading(sections: list[DocumentSection]) -> str | None:
    for section in sections:
        if section.heading:
            return section.heading
    return None
