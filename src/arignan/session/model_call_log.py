from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from arignan.tracing import ModelCallTrace

from .store import SessionStore


@dataclass(slots=True)
class SessionModelCallLogger:
    store: SessionStore
    terminal_pid: int

    def log_call(self, call: ModelCallTrace) -> Path:
        path = self.store.active_model_call_log_path(self.terminal_pid)
        path.parent.mkdir(parents=True, exist_ok=True)
        session = self.store.load_active(self.terminal_pid)
        session_id = session.session_id if session is not None else None
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "terminal_pid": self.terminal_pid,
            "session_id": session_id,
            **asdict(call),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        return path

