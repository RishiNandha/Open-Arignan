from __future__ import annotations

from arignan.cli import build_parser, main


def test_root_help_is_readable_and_avoids_subparser_ellipsis() -> None:
    help_text = build_parser().format_help()

    assert "usage: arignan [--app-home PATH] [--settings PATH] [--pid PID] [-gui] <command> [<args>]" in help_text
    assert "} ..." not in help_text
    assert "commands:" in help_text
    assert "-gui" in help_text
    assert "load" in help_text
    assert "ask" in help_text


def test_gui_flag_dispatches_to_gui_launcher(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_launch_gui(*, app_home, settings_path, terminal_pid):
        captured["app_home"] = app_home
        captured["settings_path"] = settings_path
        captured["terminal_pid"] = terminal_pid
        return 0

    monkeypatch.setattr("arignan.cli.launch_gui", fake_launch_gui)

    result = main(["--app-home", str(tmp_path / ".arignan"), "-gui"])

    assert result == 0
    assert captured["app_home"] == tmp_path / ".arignan"
