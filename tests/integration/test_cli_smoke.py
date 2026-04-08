from __future__ import annotations

import json
from pathlib import Path

from arignan.cli import main


class FakeLocalGenerator:
    backend_name = "fake-local"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 800,
        temperature: float = 0.1,
    ) -> str:
        if "Return strict JSON only" in system_prompt:
            return json.dumps(
                {
                    "title": "JEPA Notes",
                    "description": "Notes on Joint Embedding Predictive Architecture.",
                    "locator": "overview of JEPA concepts",
                    "keywords": ["JEPA", "Joint Embedding Predictive Architecture", "representation learning"],
                    "summary_markdown": (
                        "# JEPA Notes\n\n"
                        "A compact reference page.\n\n"
                        "## Summary\n"
                        "JEPA is summarized here as a representation-learning approach.\n\n"
                        "## Key Ideas\n"
                        "- Predictive representation learning\n"
                        "- Context-based targets\n"
                        "- Semantic abstraction\n\n"
                        "## Sources\n"
                        "| Source | What To Find | Key Sections | File |\n"
                        "| --- | --- | --- | --- |\n"
                        "| JEPA Notes | Overview of JEPA concepts | JEPA Notes | `notes.md` |\n\n"
                        "## Keywords\n"
                        "JEPA, Joint Embedding Predictive Architecture, representation learning"
                    ),
                }
            )
        if "knowledge-base hat map" in system_prompt:
            return (
                "# Map for Hat: default\n\n"
                "| Topic | Directory | What To Find | Source Files | Keywords |\n"
                "| --- | --- | --- | --- | --- |\n"
                "| JEPA Notes | `summaries/jepa-notes` | overview of JEPA concepts | notes.md | JEPA, representation learning |\n"
            )
        if "global knowledge-base map" in system_prompt:
            return (
                "# Global Map\n\n"
                "| Hat | Map Path | What To Find | High-Level Keywords |\n"
                "| --- | --- | --- | --- |\n"
                "| default | `hats/default/map.md` | overview of JEPA concepts | JEPA, representation learning |\n"
            )
        if self.model_name == "qwen3:0.6b":
            return "Light mode answer: JEPA stands for Joint Embedding Predictive Architecture."
        return "JEPA stands for Joint Embedding Predictive Architecture."


def _patch_local_generator(monkeypatch) -> None:
    monkeypatch.setattr(
        "arignan.application.create_local_text_generator",
        lambda config, progress_sink=None, **kwargs: FakeLocalGenerator(kwargs.get("model_name") or config.local_llm_model),
    )


def test_cli_load_ask_and_delete_smoke(tmp_path: Path, capsys, monkeypatch) -> None:
    _patch_local_generator(monkeypatch)
    app_home = tmp_path / ".arignan"
    source = tmp_path / "notes.md"
    source.write_text("# JEPA Notes\n\nJoint embedding predictive architecture overview.\n", encoding="utf-8")

    assert main(["--app-home", str(app_home), "load", str(source), "--hat", "default"]) == 0
    load_capture = capsys.readouterr()
    load_output = load_capture.out
    load_progress = load_capture.err
    load_id = load_output.split("load_id ", maxsplit=1)[1].split(".", maxsplit=1)[0]
    assert "Chunks:" in load_output
    assert "Markdown segments:" in load_output
    assert "[arignan] Scanning input for load into hat 'default'..." in load_progress
    assert "[arignan] Refreshing map.md for hat 'default'..." in load_progress

    assert main(["--app-home", str(app_home), "--pid", "1234", "ask", "What is JEPA?", "--hat", "default"]) == 0
    ask_capture = capsys.readouterr()
    ask_output = ask_capture.out
    ask_progress = ask_capture.err
    assert "Citations:" in ask_output
    assert "Joint Embedding Predictive Architecture" in ask_output
    assert "default/jepa-notes/notes.md:" in ask_output
    assert "Hat chosen: default" in ask_progress
    assert "Retrieval in progress" in ask_progress
    assert "Reranking" in ask_progress
    assert "Hitting LLM" in ask_progress

    assert main(
        ["--app-home", str(app_home), "--pid", "1234", "ask", "What is JEPA?", "--hat", "default", "--answer-mode", "light"]
    ) == 0
    light_capture = capsys.readouterr()
    assert "Light mode answer: JEPA stands for Joint Embedding Predictive Architecture." in light_capture.out
    assert "Hitting LLM" in light_capture.err

    assert main(
        ["--app-home", str(app_home), "--pid", "1234", "ask", "What is JEPA?", "--hat", "default", "--answer-mode", "none"]
    ) == 0
    none_capture = capsys.readouterr()
    assert "Joint embedding predictive architecture overview." in none_capture.out
    assert "Composing answer" in none_capture.err
    assert "Hitting LLM" not in none_capture.err

    assert main(
        ["--app-home", str(app_home), "--pid", "1234", "ask", "What is JEPA?", "--hat", "default", "--answer-mode", "raw"]
    ) == 0
    raw_capture = capsys.readouterr()
    assert "Top retrieved context:" in raw_capture.out
    assert "default/jepa-notes/notes.md:" in raw_capture.out
    assert "Citations:" not in raw_capture.out
    assert "Composing answer" in raw_capture.err
    assert "Hitting LLM" not in raw_capture.err

    assert main(["--app-home", str(app_home), "delete", load_id]) == 0
    delete_capture = capsys.readouterr()
    delete_output = delete_capture.out
    delete_progress = delete_capture.err
    assert "Deleted loads" in delete_output
    assert "[arignan] Deleting 1 load(s)..." in delete_progress
    assert "[arignan] Recording deletion log..." in delete_progress

    assert main(["--app-home", str(app_home), "list-loads"]) == 0
    log_output = capsys.readouterr().out
    assert "\tdelete\t" in log_output
    assert load_id in log_output


def test_cli_session_save_load_and_reset_smoke(tmp_path: Path, capsys, monkeypatch) -> None:
    _patch_local_generator(monkeypatch)
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


def test_cli_named_save_session_uses_app_home_saved_dir_and_preserves_turns(tmp_path: Path, capsys, monkeypatch) -> None:
    _patch_local_generator(monkeypatch)
    app_home = tmp_path / ".arignan"
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n\nJEPA stands for Joint Embedding Predictive Architecture.\n", encoding="utf-8")
    main(["--app-home", str(app_home), "load", str(source)])
    capsys.readouterr()
    main(["--app-home", str(app_home), "--pid", "4321", "ask", "What does JEPA stand for?"])
    capsys.readouterr()

    assert main(["--app-home", str(app_home), "save-session", "8apr"]) == 0
    save_output = capsys.readouterr().out.strip()
    saved_path = app_home / "sessions" / "saved" / "8apr.json"

    assert Path(save_output) == saved_path
    assert saved_path.exists()
    assert "What does JEPA stand for?" in saved_path.read_text(encoding="utf-8")


def test_cli_debug_modes_print_load_and_retrieval_details(tmp_path: Path, capsys, monkeypatch) -> None:
    _patch_local_generator(monkeypatch)
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
    load_capture = capsys.readouterr()
    load_debug = load_capture.out
    load_progress = load_capture.err
    assert "Debug: load details" in load_debug
    assert "Model calls (" in load_debug
    assert "Grouping decision:" in load_debug
    assert "topic summary markdown" in load_debug
    assert "hat map markdown" in load_debug
    assert "global map markdown" in load_debug
    assert "Calling local LLM for topic summary markdown" in load_progress

    assert main(["--app-home", str(app_home), "--pid", "4444", "ask", "What is JEPA?", "--debug"]) == 0
    ask_capture = capsys.readouterr()
    ask_debug = ask_capture.out
    ask_progress = ask_capture.err
    assert "Debug: ask retrieval" in ask_debug
    assert "Model calls (" in ask_debug
    assert "dense query embedding" in ask_debug
    assert "rerank retrieval candidates" in ask_debug
    assert "answer generation" in ask_debug
    assert "Dense hits" in ask_debug
    assert "Reranked hits" in ask_debug
    assert "default/jepa-paper/paper.md:" in ask_debug
    assert "Hat chosen: default" in ask_progress
    assert "Searching dense index in hat 'default'" in ask_progress
    assert "Fusing retrieval candidates..." in ask_progress


def test_cli_can_delete_entire_hat_after_confirmation(tmp_path: Path, capsys, monkeypatch) -> None:
    _patch_local_generator(monkeypatch)
    app_home = tmp_path / ".arignan"
    source = tmp_path / "notes.md"
    source.write_text("# SNN Notes\n\nSpiking neural networks use discrete spike events.\n", encoding="utf-8")

    assert main(["--app-home", str(app_home), "load", str(source), "--hat", "SNNs"]) == 0
    capsys.readouterr()

    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert main(["--app-home", str(app_home), "delete", "--hat", "SNNs"]) == 0
    delete_output = capsys.readouterr().out

    assert "Deleted hat 'SNNs'." in delete_output
    assert not (app_home / "hats" / "SNNs").exists()

    assert main(["--app-home", str(app_home), "list-loads"]) == 0
    log_output = capsys.readouterr().out
    assert "\tdelete\t" in log_output
    assert "\tSNNs\t" in log_output
    assert "hat:SNNs" in log_output


def test_cli_hat_delete_can_be_cancelled(tmp_path: Path, capsys, monkeypatch) -> None:
    _patch_local_generator(monkeypatch)
    app_home = tmp_path / ".arignan"
    source = tmp_path / "notes.md"
    source.write_text("# SNN Notes\n\nSpiking neural networks use discrete spike events.\n", encoding="utf-8")

    assert main(["--app-home", str(app_home), "load", str(source), "--hat", "SNNs"]) == 0
    capsys.readouterr()

    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert main(["--app-home", str(app_home), "delete", "--hat", "SNNs"]) == 0
    cancel_output = capsys.readouterr().out

    assert "Cancelled hat deletion." in cancel_output
    assert (app_home / "hats" / "SNNs").exists()
