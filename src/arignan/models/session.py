from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChatTurn:
    role: str
    content: str
    timestamp: str

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("role must not be empty")
        if not self.content:
            raise ValueError("content must not be empty")
        if not self.timestamp:
            raise ValueError("timestamp must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChatTurn":
        return cls(
            role=payload["role"],
            content=payload["content"],
            timestamp=payload["timestamp"],
        )


@dataclass(slots=True)
class SessionState:
    session_id: str
    terminal_pid: int
    hat: str = "auto"
    summary: str | None = None
    turns: list[ChatTurn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.terminal_pid <= 0:
            raise ValueError("terminal_pid must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "terminal_pid": self.terminal_pid,
            "hat": self.hat,
            "summary": self.summary,
            "turns": [turn.to_dict() for turn in self.turns],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionState":
        return cls(
            session_id=payload["session_id"],
            terminal_pid=int(payload["terminal_pid"]),
            hat=payload.get("hat", "auto"),
            summary=payload.get("summary"),
            turns=[ChatTurn.from_dict(item) for item in payload.get("turns", [])],
            metadata=dict(payload.get("metadata", {})),
        )
