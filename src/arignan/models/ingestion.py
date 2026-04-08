from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class LoadOperation(str, Enum):
    INGEST = "ingest"
    DELETE = "delete"


@dataclass(slots=True)
class TopicArtifact:
    hat: str
    topic_folder: str
    source_paths: list[Path]
    markdown_paths: list[Path]
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hat": self.hat,
            "topic_folder": self.topic_folder,
            "source_paths": [str(path) for path in self.source_paths],
            "markdown_paths": [str(path) for path in self.markdown_paths],
            "keywords": list(self.keywords),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TopicArtifact":
        return cls(
            hat=payload["hat"],
            topic_folder=payload["topic_folder"],
            source_paths=[Path(path) for path in payload.get("source_paths", [])],
            markdown_paths=[Path(path) for path in payload.get("markdown_paths", [])],
            keywords=list(payload.get("keywords", [])),
        )


@dataclass(slots=True)
class LoadEvent:
    load_id: str
    operation: LoadOperation
    hat: str
    created_at: str
    source_items: list[str]
    artifact_paths: list[Path] = field(default_factory=list)
    topic_folders: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.load_id:
            raise ValueError("load_id must not be empty")
        if not self.hat:
            raise ValueError("hat must not be empty")
        if not self.created_at:
            raise ValueError("created_at must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "load_id": self.load_id,
            "operation": self.operation.value,
            "hat": self.hat,
            "created_at": self.created_at,
            "source_items": list(self.source_items),
            "artifact_paths": [str(path) for path in self.artifact_paths],
            "topic_folders": list(self.topic_folders),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LoadEvent":
        return cls(
            load_id=payload["load_id"],
            operation=LoadOperation(payload["operation"]),
            hat=payload["hat"],
            created_at=payload["created_at"],
            source_items=list(payload.get("source_items", [])),
            artifact_paths=[Path(path) for path in payload.get("artifact_paths", [])],
            topic_folders=list(payload.get("topic_folders", [])),
            metadata=dict(payload.get("metadata", {})),
        )
