from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from arignan.models import ChunkMetadata, ChunkRecord, DocumentSection, ParsedDocument

SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
CLAUSE_BOUNDARY_PATTERN = re.compile(r"(?<=[;:])\s+|(?<=,)\s+(?=[A-Z0-9\"'(\[])")
NUMERIC_CITATION_PATTERN = re.compile(r"\[(?:\d+(?:\s*[-,]\s*\d+)*)\]")
MARKDOWN_CITATION_PATTERN = re.compile(r"\[@[^\]]+\]|\[\^[^\]]+\]")
AUTHOR_YEAR_CITATION_PATTERN = re.compile(
    r"\((?:[A-Z][A-Za-z'`\-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z'`\-]+|\s+et al\.)?,\s*(?:19|20)\d{2}[a-z]?)(?:;\s*(?:[A-Z][A-Za-z'`\-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z'`\-]+|\s+et al\.)?,\s*(?:19|20)\d{2}[a-z]?))*\)"
)
REFERENCE_HEADING_PATTERN = re.compile(r"^(references|bibliography|works cited|citations?)$", re.IGNORECASE)
REFERENCE_BLOCK_PATTERN = re.compile(
    r"(?:^|\n)\s*(references|bibliography|works cited|citations?)\s*(?:\n|$).*$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(slots=True)
class ChunkingResult:
    document: ParsedDocument
    chunks: list[ChunkRecord]


@dataclass(slots=True)
class _SectionSpan:
    text: str
    heading: str | None
    page_number: int | None
    section_label: str


class Chunker:
    def __init__(self, chunk_size: int = 2800, chunk_overlap: int = 160) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: ParsedDocument) -> list[ChunkRecord]:
        sections = self._section_spans(self._relevant_sections(document))
        chunks: list[ChunkRecord] = []
        for section_index, section in enumerate(sections):
            section_chunks = self._chunk_section_text(section.text)
            for piece_index, chunk_text in enumerate(section_chunks):
                metadata = ChunkMetadata(
                    load_id=document.load_id,
                    hat=document.hat,
                    source_uri=document.source.source_uri,
                    source_path=document.source.local_path,
                    page_number=section.page_number,
                    section=section.section_label or self._fallback_section_label(section_index),
                    heading=section.heading,
                    keywords=list(document.keywords),
                )
                chunks.append(
                    ChunkRecord(
                        chunk_id=self._chunk_id(document, section_index, piece_index, chunk_text),
                        text=chunk_text,
                        metadata=metadata,
                    )
                )
        return chunks

    def chunk_documents(self, documents: list[ParsedDocument]) -> list[ChunkingResult]:
        return [ChunkingResult(document=document, chunks=self.chunk_document(document)) for document in documents]

    def _relevant_sections(self, document: ParsedDocument) -> list[DocumentSection]:
        if not document.sections:
            return [DocumentSection(text=document.full_text)]
        if len(document.sections) == 1 and document.sections[0].heading is None and document.sections[0].page_number is None:
            return [DocumentSection(text=document.full_text)]
        relevant = [section for section in document.sections if not self._is_reference_section(section)]
        return relevant or document.sections

    def _section_spans(self, sections: list[DocumentSection]) -> list[_SectionSpan]:
        if not sections:
            return []
        spans: list[_SectionSpan] = []
        buffered_sections: list[DocumentSection] = []
        buffered_length = 0
        merge_target = max(900, int(self.chunk_size * 0.7))

        def flush() -> None:
            nonlocal buffered_sections, buffered_length
            if not buffered_sections:
                return
            spans.append(self._build_section_span(buffered_sections, len(spans)))
            buffered_sections = []
            buffered_length = 0

        for section in sections:
            section_length = len(self._normalize_text(self._clean_text_for_chunking(section.text)))
            if not buffered_sections:
                buffered_sections = [section]
                buffered_length = section_length
                continue
            if self._should_merge_span(buffered_sections, section, buffered_length, section_length, merge_target):
                buffered_sections.append(section)
                buffered_length += 2 + section_length
                continue
            flush()
            buffered_sections = [section]
            buffered_length = section_length

        flush()
        return spans

    def _chunk_section_text(self, text: str) -> list[str]:
        cleaned = self._clean_text_for_chunking(text)
        normalized = self._normalize_text(cleaned)
        if not normalized:
            return []
        if len(normalized) <= self.chunk_size:
            return [normalized]

        units = self._sentence_units(cleaned)
        if not units:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_length = 0

        for unit in units:
            if len(unit) > self.chunk_size:
                long_units = self._wrap_words(unit)
            else:
                long_units = [unit]

            for piece in long_units:
                extra = len(piece) if not current else len(piece) + 1
                if current and current_length + extra > self.chunk_size:
                    chunks.append(" ".join(current))
                    current = self._tail_overlap(current)
                    current_length = len(" ".join(current))
                    while current and current_length + extra > self.chunk_size:
                        current.pop(0)
                        current_length = len(" ".join(current))
                if current:
                    current_length += len(piece) + 1
                else:
                    current_length = len(piece)
                current.append(piece)

        if current:
            chunks.append(" ".join(current))
        return chunks

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split())

    def _clean_text_for_chunking(self, text: str) -> str:
        cleaned = self._strip_reference_blocks(text)
        cleaned = MARKDOWN_CITATION_PATTERN.sub("", cleaned)
        cleaned = NUMERIC_CITATION_PATTERN.sub("", cleaned)
        cleaned = AUTHOR_YEAR_CITATION_PATTERN.sub("", cleaned)
        cleaned = re.sub(r"\(\s*\)", "", cleaned)
        cleaned = re.sub(r"\s+([,;:.!?])", r"\1", cleaned)
        cleaned = re.sub(r"([,;:])(?=[A-Za-z0-9])", r"\1 ", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _sentence_units(self, text: str) -> list[str]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []
        paragraphs = [self._normalize_text(part) for part in re.split(r"(?:\r?\n){2,}", text) if part.strip()]
        if not paragraphs:
            paragraphs = [normalized]
        units: list[str] = []
        for paragraph in paragraphs:
            sentences = [part.strip() for part in SENTENCE_BOUNDARY_PATTERN.split(paragraph) if part.strip()]
            if not sentences:
                continue
            if len(sentences) == 1 and len(sentences[0]) > self.chunk_size:
                clauses = [part.strip() for part in CLAUSE_BOUNDARY_PATTERN.split(sentences[0]) if part.strip()]
                if len(clauses) > 1:
                    units.extend(clauses)
                    continue
            units.extend(sentences)
        return units or [normalized]

    def _wrap_words(self, text: str) -> list[str]:
        words = text.split(" ")
        pieces: list[str] = []
        current: list[str] = []
        current_length = 0

        for word in words:
            extra = len(word) if not current else len(word) + 1
            if current and current_length + extra > self.chunk_size:
                pieces.append(" ".join(current))
                overlap_words = self._tail_overlap(current)
                current = overlap_words[:] if overlap_words else []
                current_length = len(" ".join(current))
                while current and current_length + extra > self.chunk_size:
                    current.pop(0)
                    current_length = len(" ".join(current))
            if current:
                current_length += len(word) + 1
            else:
                current_length = len(word)
            current.append(word)

        if current:
            pieces.append(" ".join(current))
        return pieces

    def _tail_overlap(self, words: list[str]) -> list[str]:
        overlap_words: list[str] = []
        length = 0
        for word in reversed(words):
            extra = len(word) if not overlap_words else len(word) + 1
            if overlap_words and length + extra > self.chunk_overlap:
                break
            overlap_words.insert(0, word)
            length += extra
        return overlap_words

    def _should_merge_span(
        self,
        buffered_sections: list[DocumentSection],
        next_section: DocumentSection,
        current_length: int,
        next_length: int,
        merge_target: int,
    ) -> bool:
        if current_length >= merge_target:
            return False
        if current_length + 2 + next_length > self.chunk_size:
            return False

        last_section = buffered_sections[-1]
        if self._is_soft_boundary(last_section, next_section):
            return True
        return False

    def _build_section_span(self, sections: list[DocumentSection], span_index: int) -> _SectionSpan:
        text = "\n\n".join(section.text.strip() for section in sections if section.text.strip())
        headings = [
            section.heading.strip()
            for section in sections
            if section.heading and not self._is_page_heading(section.heading)
        ]
        headings = list(dict.fromkeys(headings))
        page_numbers = [section.page_number for section in sections if section.page_number is not None]

        heading = headings[0] if len(headings) == 1 else None
        if len(page_numbers) == 1:
            return _SectionSpan(text=text, heading=heading, page_number=page_numbers[0], section_label=f"Page {page_numbers[0]}")
        if len(page_numbers) > 1:
            return _SectionSpan(text=text, heading=heading, page_number=None, section_label=f"Pages {page_numbers[0]}-{page_numbers[-1]}")
        if len(headings) == 1:
            return _SectionSpan(text=text, heading=headings[0], page_number=None, section_label=headings[0])
        if len(headings) > 1:
            return _SectionSpan(text=text, heading=headings[0], page_number=None, section_label=f"{headings[0]} -> {headings[-1]}")
        return _SectionSpan(text=text, heading=None, page_number=None, section_label=self._fallback_section_label(span_index))

    def _is_soft_boundary(self, current: DocumentSection, following: DocumentSection) -> bool:
        current_heading = (current.heading or "").strip()
        following_heading = (following.heading or "").strip()
        if self._is_page_heading(current_heading) or self._is_page_heading(following_heading):
            return True
        if not current_heading or not following_heading:
            return True
        return current_heading.lower() == following_heading.lower()

    def _is_reference_section(self, section: DocumentSection) -> bool:
        heading = (section.heading or "").strip()
        if heading and REFERENCE_HEADING_PATTERN.fullmatch(heading):
            return True
        cleaned = self._normalize_text(section.text[:240]).lower()
        return bool(cleaned.startswith(("references ", "bibliography ", "works cited ", "citations ")))

    @staticmethod
    def _strip_reference_blocks(text: str) -> str:
        stripped = REFERENCE_BLOCK_PATTERN.sub("\n", text)
        return stripped

    @staticmethod
    def _fallback_section_label(section_index: int) -> str:
        return f"Section {section_index + 1}"

    @staticmethod
    def _is_page_heading(heading: str) -> bool:
        normalized = heading.strip().lower()
        return bool(re.fullmatch(r"page\s+\d+", normalized))

    def _chunk_id(
        self,
        document: ParsedDocument,
        section_index: int,
        piece_index: int,
        chunk_text: str,
    ) -> str:
        digest = hashlib.sha1(
            f"{document.load_id}|{document.source.source_uri}|{section_index}|{piece_index}|{chunk_text}".encode("utf-8")
        ).hexdigest()[:12]
        return f"chunk-{digest}"
