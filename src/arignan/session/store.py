from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

from arignan.models import SessionState


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.active_dir = root / "sessions" / "active"
        self.saved_dir = root / "sessions" / "saved"
        self.active_dir.mkdir(parents=True, exist_ok=True)
        self.saved_dir.mkdir(parents=True, exist_ok=True)

    def active_path(self, terminal_pid: int) -> Path:
        return self.active_dir / f"pid-{terminal_pid}.json"

    def active_session_dir(self, terminal_pid: int) -> Path:
        return self.active_dir / f"pid-{terminal_pid}"

    def active_exception_log_path(self, terminal_pid: int) -> Path:
        return self.active_session_dir(terminal_pid) / "exceptions.log"

    def active_model_call_log_path(self, terminal_pid: int) -> Path:
        return self.active_session_dir(terminal_pid) / "model_calls.log"

    def load_active(self, terminal_pid: int) -> SessionState | None:
        path = self.active_path(terminal_pid)
        if not path.exists():
            return None
        return SessionState.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save_active(self, session: SessionState) -> Path:
        path = self.active_path(session.terminal_pid)
        path.write_text(json.dumps(session.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    def delete_active(self, terminal_pid: int) -> None:
        path = self.active_path(terminal_pid)
        if path.exists():
            path.unlink()
        session_dir = self.active_session_dir(terminal_pid)
        if session_dir.exists():
            shutil.rmtree(session_dir)

    def save_snapshot(self, session: SessionState, destination: Path | None = None) -> Path:
        target = self.resolve_snapshot_destination(destination, session.session_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(session.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    def load_snapshot(self, path: Path, terminal_pid: int | None = None) -> SessionState:
        resolved = self.resolve_snapshot_source(path)
        session = SessionState.from_dict(json.loads(resolved.read_text(encoding="utf-8")))
        if terminal_pid is None:
            return session
        return replace(session, terminal_pid=terminal_pid)

    def latest_active(self, require_content: bool = False) -> SessionState | None:
        sessions: list[SessionState] = []
        for path in sorted(self.active_dir.glob("pid-*.json")):
            session = SessionState.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if require_content and not (session.turns or session.summary):
                continue
            sessions.append(session)
        if not sessions:
            return None
        sessions.sort(key=lambda item: item.metadata.get("last_activity_at", ""), reverse=True)
        return sessions[0]

    def resolve_snapshot_destination(self, destination: Path | None, session_id: str) -> Path:
        if destination is None:
            return self.saved_dir / f"{session_id}.json"
        if destination.is_absolute():
            return destination
        if destination.parent == Path("."):
            name = destination.name
            if "." not in destination.name:
                name = f"{name}.json"
            return self.saved_dir / name
        return self.saved_dir / destination

    def resolve_snapshot_source(self, source: Path) -> Path:
        if source.is_absolute() and source.exists():
            return source
        if source.exists():
            return source
        candidates = [self.saved_dir / source]
        if source.suffix != ".json":
            candidates.append(self.saved_dir / f"{source.name}.json")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return source
