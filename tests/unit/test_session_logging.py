from __future__ import annotations

from arignan.models import SessionState
from arignan.session import SessionExceptionLogger, SessionModelCallLogger, SessionStore
import json

from arignan.tracing import ModelCallTrace, ModelTraceCollector


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


def test_session_model_call_logger_writes_jsonl_entries(app_home) -> None:
    store = SessionStore(app_home)
    store.save_active(SessionState(session_id="session-2", terminal_pid=888))
    logger = SessionModelCallLogger(store, terminal_pid=888)

    log_path = logger.log_call(
        ModelCallTrace(
            component="llm",
            task="answer generation",
            model_name="qwen3:4b-q4_K_M",
            backend="ollama-local",
            status="ok",
            item_count=6,
            detail="default",
        )
    )

    payload = json.loads(log_path.read_text(encoding="utf-8").strip())

    assert log_path == app_home / "sessions" / "active" / "pid-888" / "model_calls.log"
    assert payload["session_id"] == "session-2"
    assert payload["component"] == "llm"
    assert payload["task"] == "answer generation"
    assert payload["model_name"] == "qwen3:4b-q4_K_M"
    assert payload["backend"] == "ollama-local"


def test_model_trace_collector_can_stream_calls_to_session_logger(app_home) -> None:
    store = SessionStore(app_home)
    store.save_active(SessionState(session_id="session-3", terminal_pid=999))
    logger = SessionModelCallLogger(store, terminal_pid=999)
    collector = ModelTraceCollector(on_record=logger.log_call)

    collector.record(
        component="embedder",
        task="dense query embedding",
        model_name="BAAI/bge-base-en-v1.5",
        backend="hashing-embedder",
        item_count=1,
        detail="top_k=14",
    )

    log_path = store.active_model_call_log_path(999)
    payload = json.loads(log_path.read_text(encoding="utf-8").strip())

    assert collector.snapshot()[0].task == "dense query embedding"
    assert payload["component"] == "embedder"
    assert payload["detail"] == "top_k=14"
