from __future__ import annotations

import json

from arignan.config import SessionConfig
from arignan.session import SessionManager, SessionStore


def test_session_persistence_overwrites_active_json_after_rollover(app_home) -> None:
    manager = SessionManager(SessionStore(app_home), SessionConfig(soft_token_limit=90, keep_recent_turns=2))

    for index in range(5):
        manager.append_turn(terminal_pid=777, role="user", content=f"Long turn {index} about hybrid retrieval systems")

    active_path = SessionStore(app_home).active_path(777)
    payload = json.loads(active_path.read_text(encoding="utf-8"))

    assert payload["summary"]
    assert len(payload["turns"]) == 2
