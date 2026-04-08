from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from arignan.models.documents import ChunkMetadata


class RetrievalSource(str, Enum):
    DENSE = "dense"
    LEXICAL = "lexical"
    MAP = "map"


@dataclass(slots=True)
class RetrievalHit:
    chunk_id: str
    text: str
    score: float
    source: RetrievalSource
    metadata: ChunkMetadata
    rank: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "source": self.source.value,
            "metadata": self.metadata.to_dict(),
            "rank": self.rank,
            "extras": self.extras,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetrievalHit":
        return cls(
            chunk_id=payload["chunk_id"],
            text=payload["text"],
            score=float(payload["score"]),
            source=RetrievalSource(payload["source"]),
            metadata=ChunkMetadata.from_dict(payload["metadata"]),
            rank=payload.get("rank"),
            extras=dict(payload.get("extras", {})),
        )
