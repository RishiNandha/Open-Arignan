from __future__ import annotations

from arignan.cli import build_parser, main


def test_root_help_is_readable_and_avoids_subparser_ellipsis() -> None:
    help_text = build_parser().format_help()

    assert "usage: arignan [--app-home PATH] [--settings PATH] [--pid PID] [-gui] <command> [<args>]" in help_text
    assert "} ..." not in help_text
    assert "commands:" in help_text
    assert "-gui" in help_text
    assert "setup" in help_text
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


def test_setup_command_initializes_state_without_constructing_app(monkeypatch, tmp_path, capsys) -> None:
    calls: dict[str, object] = {}

    def fake_initialize_local_state(**kwargs):
        calls["initialize"] = kwargs
        return tmp_path / ".arignan", tmp_path / ".arignan" / "settings.json"

    def fake_download_required_models(app_home, progress=None):
        calls["download"] = app_home
        if progress is not None:
            progress("fake download")
        return app_home / "models"

    def exploding_app(*args, **kwargs):
        raise AssertionError("setup command should not construct ArignanApp")

    monkeypatch.setattr("arignan.setup_flow.initialize_local_state", fake_initialize_local_state)
    monkeypatch.setattr("arignan.setup_flow.download_required_models", fake_download_required_models)
    monkeypatch.setattr("arignan.application.ArignanApp", exploding_app)

    result = main(
        [
            "--app-home",
            str(tmp_path / ".arignan"),
            "setup",
            "--llm-backend",
            "transformers",
            "--llm-model",
            "Qwen3-0.6B",
            "--llm-light-model",
            "Qwen3-0.6B",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert calls["initialize"]["local_llm_backend"] == "transformers"
    assert calls["initialize"]["local_llm_model"] == "Qwen3-0.6B"
    assert calls["initialize"]["local_llm_light_model"] == "Qwen3-0.6B"
    assert calls["initialize"]["refresh_existing"] is False
    assert "Arignan standalone setup complete." in output
