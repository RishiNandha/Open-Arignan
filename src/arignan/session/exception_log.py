from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .store import SessionStore


@dataclass(slots=True)
class SessionExceptionLogger:
    store: SessionStore
    terminal_pid: int

    def log_exception(
        self,
        *,
        component: str,
        task: str,
        exc: BaseException,
        context: dict[str, Any] | None = None,
    ) -> Path:
        path = self.store.active_exception_log_path(self.terminal_pid)
        path.parent.mkdir(parents=True, exist_ok=True)
        session = self.store.load_active(self.terminal_pid)
        session_id = session.session_id if session is not None else None
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "terminal_pid": self.terminal_pid,
            "session_id": session_id,
            "component": component,
            "task": task,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "context": context or {},
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip(),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write("=" * 80 + "\n")
            handle.write(json.dumps(payload, indent=2) + "\n")
        return path
