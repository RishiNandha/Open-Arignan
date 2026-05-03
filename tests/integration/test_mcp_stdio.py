from __future__ import annotations

import logging
from pathlib import Path

from arignan.cli import McpReporter, McpStderrFormatter, _configure_mcp_stderr_logging


def test_mcp_reporter_logs_progress_to_stderr_and_file(tmp_path: Path, capsys) -> None:
    app_home = tmp_path / ".arignan"
    reporter = McpReporter(app_home)

    reporter.emit("Server started")
    reporter.emit("Listing tools")

    stderr_text = capsys.readouterr().err
    assert "MCP Log Path:" in stderr_text
    assert "[arignan-mcp] Server started" in stderr_text
    assert "[arignan-mcp] Listing tools" in stderr_text

    log_text = (app_home / "sessions" / "mcp.log").read_text()
    assert "Server started" in log_text
    assert "Listing tools" in log_text


def test_mcp_stderr_formatter_maps_known_sdk_messages() -> None:
    formatter = McpStderrFormatter()
    record = logging.LogRecord(
        name="mcp",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Processing request of type InitializeRequest",
        args=(),
        exc_info=None,
    )

    assert formatter.format(record) == "[arignan-mcp] Initialize request received"


def test_configure_mcp_stderr_logging_replaces_arignan_handler() -> None:
    logger = logging.getLogger("mcp")
    original_handlers = list(logger.handlers)

    try:
        _configure_mcp_stderr_logging(stream=None)
        first_handlers = [handler for handler in logger.handlers if getattr(handler, "name", "") == "arignan_mcp_stderr"]
        assert len(first_handlers) == 1

        _configure_mcp_stderr_logging(stream=None)
        second_handlers = [handler for handler in logger.handlers if getattr(handler, "name", "") == "arignan_mcp_stderr"]
        assert len(second_handlers) == 1
        assert second_handlers[0] is not first_handlers[0]
        assert logger.propagate is False
    finally:
        logger.handlers[:] = original_handlers
