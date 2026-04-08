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


class Chunker:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 80) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: ParsedDocument) -> list[ChunkRecord]:
        sections = self._relevant_sections(document)
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
                    section=self._section_label(section, section_index),
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

    def _section_label(self, section: DocumentSection, section_index: int) -> str:
        if section.heading:
            return section.heading
        if section.page_number is not None:
            return f"Page {section.page_number}"
        return f"Section {section_index + 1}"

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
