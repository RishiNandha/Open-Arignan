from __future__ import annotations

from pathlib import Path

from arignan.cli import main


def test_cli_load_ask_and_delete_smoke(tmp_path: Path, capsys) -> None:
    app_home = tmp_path / ".arignan"
    source = tmp_path / "notes.md"
    source.write_text("# JEPA Notes\n\nJoint embedding predictive architecture overview.\n", encoding="utf-8")

    assert main(["--app-home", str(app_home), "load", str(source), "--hat", "default"]) == 0
    load_output = capsys.readouterr().out
    load_id = load_output.split("load_id ", maxsplit=1)[1].split(".", maxsplit=1)[0]
    assert "Chunks:" in load_output
    assert "Markdown segments:" in load_output

    assert main(["--app-home", str(app_home), "--pid", "1234", "ask", "What is JEPA?", "--hat", "default"]) == 0
    ask_output = capsys.readouterr().out
    assert "Citations:" in ask_output
    assert "Joint embedding predictive architecture overview." in ask_output
    assert "default/jepa-notes/notes.md:" in ask_output

    assert main(["--app-home", str(app_home), "delete", load_id]) == 0
    delete_output = capsys.readouterr().out
    assert "Deleted loads" in delete_output

    assert main(["--app-home", str(app_home), "list-loads"]) == 0
    log_output = capsys.readouterr().out
    assert "\tdelete\t" in log_output
    assert load_id in log_output


def test_cli_session_save_load_and_reset_smoke(tmp_path: Path, capsys) -> None:
    app_home = tmp_path / ".arignan"
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n\nRetrieval notes.\n", encoding="utf-8")
    main(["--app-home", str(app_home), "load", str(source)])
    capsys.readouterr()
    main(["--app-home", str(app_home), "--pid", "4321", "ask", "retrieval?", "--hat", "default"])
    capsys.readouterr()

    destination = tmp_path / "saved-session.json"
    assert main(["--app-home", str(app_home), "--pid", "4321", "save-session", str(destination)]) == 0
    assert destination.exists()
    capsys.readouterr()

    assert main(["--app-home", str(app_home), "--pid", "5000", "load-session", str(destination)]) == 0
    load_output = capsys.readouterr().out
    assert "Loaded session" in load_output

    assert main(["--app-home", str(app_home), "--pid", "5000", "reset-session"]) == 0
    reset_output = capsys.readouterr().out
    assert "Reset session" in reset_output


def test_cli_debug_modes_print_load_and_retrieval_details(tmp_path: Path, capsys) -> None:
    app_home = tmp_path / ".arignan"
    source = tmp_path / "paper.md"
    source.write_text(
        "# JEPA Paper\n\n"
        "Joint embedding predictive architecture is useful for representation learning.\n\n"
        "## Training\n\n"
        "The method predicts latent targets from context.\n",
        encoding="utf-8",
    )

    assert main(["--app-home", str(app_home), "load", str(source), "--debug"]) == 0
    load_debug = capsys.readouterr().out
    assert "Debug: load details" in load_debug
    assert "Grouping decision:" in load_debug

    assert main(["--app-home", str(app_home), "--pid", "4444", "ask", "What is JEPA?", "--debug"]) == 0
    ask_debug = capsys.readouterr().out
    assert "Debug: ask retrieval" in ask_debug
    assert "Dense hits" in ask_debug
    assert "Reranked hits" in ask_debug
    assert "default/jepa-paper/paper.md:" in ask_debug
