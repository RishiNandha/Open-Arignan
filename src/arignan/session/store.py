from __future__ import annotations

import json
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

    def save_snapshot(self, session: SessionState, destination: Path | None = None) -> Path:
        target = destination or (self.saved_dir / f"{session.session_id}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(session.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    def load_snapshot(self, path: Path, terminal_pid: int | None = None) -> SessionState:
        session = SessionState.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if terminal_pid is None:
            return session
        return replace(session, terminal_pid=terminal_pid)
