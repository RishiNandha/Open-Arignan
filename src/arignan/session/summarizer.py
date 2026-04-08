from __future__ import annotations

from typing import Protocol

from arignan.models import ChatTurn


class SessionSummarizer(Protocol):
    def summarize(self, turns: list[ChatTurn], existing_summary: str | None = None) -> str:
        """Summarize older turns into compact session memory."""


class HeuristicSessionSummarizer:
    def summarize(self, turns: list[ChatTurn], existing_summary: str | None = None) -> str:
        snippets = [f"{turn.role}: {' '.join(turn.content.split())[:80]}" for turn in turns]
        lines: list[str] = []
        if existing_summary:
            lines.append(existing_summary.strip())
        if snippets:
            lines.append("Earlier discussion: " + " | ".join(snippets))
        return "\n".join(line for line in lines if line).strip()
