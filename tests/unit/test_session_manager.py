from __future__ import annotations

from pathlib import Path

from arignan.config import SessionConfig
from arignan.session import HeuristicSessionSummarizer, SessionManager, SessionStore


def test_session_manager_creates_pid_scoped_session(app_home: Path) -> None:
    manager = SessionManager(SessionStore(app_home), SessionConfig())

    session = manager.get_or_create(terminal_pid=321, hat="default")

    assert session.terminal_pid == 321
    assert session.hat == "default"
    assert session.session_id.startswith("session-")


def test_session_manager_rolls_over_long_history(app_home: Path) -> None:
    manager = SessionManager(
        SessionStore(app_home),
        SessionConfig(soft_token_limit=80, keep_recent_turns=2),
        summarizer=HeuristicSessionSummarizer(),
    )

    for index in range(6):
        manager.append_turn(terminal_pid=654, role="user", content=f"Turn {index} about retrieval and architecture")

    session = manager.get_or_create(654)

    assert session.summary is not None
    assert len(session.turns) == 2
    assert "Earlier discussion" in session.summary


def test_session_manager_save_load_and_reset(app_home: Path) -> None:
    manager = SessionManager(SessionStore(app_home), SessionConfig())
    manager.append_turn(terminal_pid=999, role="user", content="Saved turn")

    saved_path = manager.save_session(999)
    loaded = manager.load_session(terminal_pid=1000, source=saved_path)
    reset = manager.reset_session(1000)

    assert saved_path.exists()
    assert loaded.turns[0].content == "Saved turn"
    assert reset.terminal_pid == 1000
    assert reset.turns == []


def test_session_manager_named_save_uses_saved_dir_and_falls_back_to_latest_active(app_home: Path) -> None:
    manager = SessionManager(SessionStore(app_home), SessionConfig())
    manager.append_turn(terminal_pid=1234, role="user", content="Remember this context")

    saved_path = manager.save_session(terminal_pid=9999, destination=Path("8apr"))

    assert saved_path == app_home / "sessions" / "saved" / "8apr.json"
    assert saved_path.exists()
    assert "Remember this context" in saved_path.read_text(encoding="utf-8")
