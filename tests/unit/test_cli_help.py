from __future__ import annotations

from arignan.cli import build_parser


def test_root_help_is_readable_and_avoids_subparser_ellipsis() -> None:
    help_text = build_parser().format_help()

    assert "usage: arignan [--app-home PATH] [--settings PATH] [--pid PID] <command> [<args>]" in help_text
    assert "} ..." not in help_text
    assert "commands:" in help_text
    assert "load" in help_text
    assert "ask" in help_text
