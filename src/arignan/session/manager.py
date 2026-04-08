from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from arignan.config import SessionConfig
from arignan.models import ChatTurn, SessionState

from .store import SessionStore
from .summarizer import HeuristicSessionSummarizer, SessionSummarizer


class SessionManager:
    def __init__(
        self,
        store: SessionStore,
        config: SessionConfig,
        summarizer: SessionSummarizer | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self.summarizer = summarizer or HeuristicSessionSummarizer()

    def get_or_create(self, terminal_pid: int, hat: str = "auto") -> SessionState:
        existing = self.store.load_active(terminal_pid)
        if existing is not None:
            refreshed = self._apply_idle_timeout(existing)
            self.store.save_active(refreshed)
            return refreshed
        session = SessionState(
            session_id=f"session-{uuid4().hex[:12]}",
            terminal_pid=terminal_pid,
            hat=hat,
            metadata={
                "created_at": self._now().isoformat(),
                "last_activity_at": self._now().isoformat(),
                "kv_cache_reset_at": self._now().isoformat(),
            },
        )
        self.store.save_active(session)
        return session

    def append_turn(self, terminal_pid: int, role: str, content: str, timestamp: str | None = None) -> SessionState:
        session = self.get_or_create(terminal_pid)
        turn = ChatTurn(role=role, content=content, timestamp=timestamp or self._now().isoformat())
        session.turns.append(turn)
        session.metadata["last_activity_at"] = turn.timestamp
        session = self._rollover_if_needed(session)
        self.store.save_active(session)
        return session

    def save_session(self, terminal_pid: int, destination: Path | None = None) -> Path:
        session = self.store.load_active(terminal_pid)
        if session is None or (not session.turns and not session.summary):
            session = self.store.latest_active(require_content=True) or self.get_or_create(terminal_pid)
        return self.store.save_snapshot(session, destination=destination)

    def load_session(self, terminal_pid: int, source: Path) -> SessionState:
        session = self.store.load_snapshot(source, terminal_pid=terminal_pid)
        session.metadata["last_activity_at"] = self._now().isoformat()
        self.store.save_active(session)
        return session

    def reset_session(self, terminal_pid: int, hat: str = "auto") -> SessionState:
        self.store.delete_active(terminal_pid)
        return self.get_or_create(terminal_pid, hat=hat)

    def _rollover_if_needed(self, session: SessionState) -> SessionState:
        total_size = self._estimated_context_size(session)
        if total_size <= self.config.soft_token_limit:
            return session
        if len(session.turns) <= self.config.keep_recent_turns:
            return session

        preserved_turns = session.turns[-self.config.keep_recent_turns :]
        older_turns = session.turns[: -self.config.keep_recent_turns]
        summary = self.summarizer.summarize(older_turns, existing_summary=session.summary)
        new_session = replace(session, summary=summary, turns=list(preserved_turns))
        new_session.metadata["rolled_over_at"] = self._now().isoformat()
        return new_session

    def _apply_idle_timeout(self, session: SessionState) -> SessionState:
        last_activity_raw = session.metadata.get("last_activity_at")
        if not last_activity_raw:
            return session
        last_activity = datetime.fromisoformat(last_activity_raw)
        if self._now() - last_activity <= timedelta(minutes=self.config.idle_timeout_minutes):
            return session
        updated = replace(session)
        updated.metadata["kv_cache_reset_at"] = self._now().isoformat()
        updated.metadata["last_activity_at"] = self._now().isoformat()
        return updated

    def _estimated_context_size(self, session: SessionState) -> int:
        turns_size = sum(len(turn.content) for turn in session.turns)
        summary_size = len(session.summary or "")
        return turns_size + summary_size

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)
