from __future__ import annotations

from pathlib import Path

import pytest

from arignan.cli import main


def test_cli_logs_unhandled_exception_to_active_session_log(tmp_path: Path, capsys, monkeypatch) -> None:
    app_home = tmp_path / ".arignan"

    def boom(self, question: str, hat: str = "auto", terminal_pid: int | None = None):
        raise RuntimeError("cli boom")

    monkeypatch.setattr("arignan.cli.ArignanApp.ask", boom)

    with pytest.raises(RuntimeError, match="cli boom"):
        main(["--app-home", str(app_home), "--pid", "4444", "ask", "What is JEPA?"])

    captured = capsys.readouterr()
    log_path = app_home / "sessions" / "active" / "pid-4444" / "exceptions.log"
    log_text = log_path.read_text(encoding="utf-8")

    assert "Full traceback logged to" in captured.err
    assert str(log_path) in captured.err
    assert log_path.exists()
    assert '"component": "cli"' in log_text
    assert '"task": "ask command"' in log_text
    assert "cli boom" in log_text
