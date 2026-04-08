from __future__ import annotations

from arignan.models import SessionState
from arignan.session import SessionExceptionLogger, SessionStore


def test_session_exception_logger_writes_traceback_to_active_session_log(app_home) -> None:
    store = SessionStore(app_home)
    logger = SessionExceptionLogger(store, terminal_pid=555)

    try:
        raise ValueError("debug me")
    except ValueError as exc:
        log_path = logger.log_exception(
            component="cli",
            task="ask command",
            exc=exc,
            context={"question": "What is JEPA?"},
        )

    log_text = log_path.read_text(encoding="utf-8")

    assert log_path == app_home / "sessions" / "active" / "pid-555" / "exceptions.log"
    assert '"component": "cli"' in log_text
    assert '"task": "ask command"' in log_text
    assert '"exception_type": "ValueError"' in log_text
    assert '"question": "What is JEPA?"' in log_text
    assert "debug me" in log_text
    assert "Traceback (most recent call last)" in log_text


def test_delete_active_removes_session_log_artifacts(app_home) -> None:
    store = SessionStore(app_home)
    session = SessionState(session_id="session-1", terminal_pid=777)
    store.save_active(session)
    log_path = store.active_exception_log_path(777)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("traceback", encoding="utf-8")

    store.delete_active(777)

    assert not store.active_path(777).exists()
    assert not store.active_session_dir(777).exists()
