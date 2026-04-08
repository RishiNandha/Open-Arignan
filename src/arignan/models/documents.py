from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SourceType(str, Enum):
    URL = "url"
    PDF = "pdf"
    MARKDOWN = "markdown"
    FOLDER = "folder"


@dataclass(slots=True)
class SourceDocument:
    source_type: SourceType
    source_uri: str
    local_path: Path | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_uri:
            raise ValueError("source_uri must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "source_uri": self.source_uri,
            "local_path": str(self.local_path) if self.local_path else None,
            "title": self.title,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceDocument":
        return cls(
            source_type=SourceType(payload["source_type"]),
            source_uri=payload["source_uri"],
            local_path=Path(payload["local_path"]) if payload.get("local_path") else None,
            title=payload.get("title"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class DocumentSection:
    text: str
    heading: str | None = None
    page_number: int | None = None
    char_start: int | None = None
    char_end: int | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("section text must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "heading": self.heading,
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentSection":
        return cls(
            text=payload["text"],
            heading=payload.get("heading"),
            page_number=payload.get("page_number"),
            char_start=payload.get("char_start"),
            char_end=payload.get("char_end"),
        )


@dataclass(slots=True)
class ParsedDocument:
    load_id: str
    hat: str
    source: SourceDocument
    full_text: str
    sections: list[DocumentSection] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.load_id:
            raise ValueError("load_id must not be empty")
        if not self.hat:
            raise ValueError("hat must not be empty")
        if not self.full_text.strip():
            raise ValueError("full_text must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "load_id": self.load_id,
            "hat": self.hat,
            "source": self.source.to_dict(),
            "full_text": self.full_text,
            "sections": [section.to_dict() for section in self.sections],
            "keywords": list(self.keywords),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParsedDocument":
        return cls(
            load_id=payload["load_id"],
            hat=payload["hat"],
            source=SourceDocument.from_dict(payload["source"]),
            full_text=payload["full_text"],
            sections=[DocumentSection.from_dict(item) for item in payload.get("sections", [])],
            keywords=list(payload.get("keywords", [])),
        )


@dataclass(slots=True)
class ChunkMetadata:
    load_id: str
    hat: str
    source_uri: str
    source_path: Path | None = None
    page_number: int | None = None
    section: str | None = None
    heading: str | None = None
    keywords: list[str] = field(default_factory=list)
    topic_folder: str | None = None
    is_map_context: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "load_id": self.load_id,
            "hat": self.hat,
            "source_uri": self.source_uri,
            "source_path": str(self.source_path) if self.source_path else None,
            "page_number": self.page_number,
            "section": self.section,
            "heading": self.heading,
            "keywords": list(self.keywords),
            "topic_folder": self.topic_folder,
            "is_map_context": self.is_map_context,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChunkMetadata":
        return cls(
            load_id=payload["load_id"],
            hat=payload["hat"],
            source_uri=payload["source_uri"],
            source_path=Path(payload["source_path"]) if payload.get("source_path") else None,
            page_number=payload.get("page_number"),
            section=payload.get("section"),
            heading=payload.get("heading"),
            keywords=list(payload.get("keywords", [])),
            topic_folder=payload.get("topic_folder"),
            is_map_context=bool(payload.get("is_map_context", False)),
        )


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty")
        if not self.text.strip():
            raise ValueError("text must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata.to_dict(),
            "embedding": list(self.embedding) if self.embedding is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChunkRecord":
        embedding = payload.get("embedding")
        return cls(
            chunk_id=payload["chunk_id"],
            text=payload["text"],
            metadata=ChunkMetadata.from_dict(payload["metadata"]),
            embedding=list(embedding) if embedding is not None else None,
        )
