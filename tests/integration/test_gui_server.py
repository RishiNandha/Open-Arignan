from __future__ import annotations

from pathlib import Path
import time

from fastapi.testclient import TestClient

from arignan.application import ArignanApp, AskDebug, AskResult, LoadResult
from arignan.config import load_config
from arignan.gui import create_gui_app


def test_gui_options_and_root_render(tmp_path: Path) -> None:
    app_home = tmp_path / ".arignan"
    app = ArignanApp(load_config(app_home=app_home))
    app.layout.hat("SNNs").ensure()
    client = TestClient(create_gui_app(app))

    root = client.get("/")
    options = client.get("/api/options")

    assert root.status_code == 200
    assert "Open Arignan" in root.text
    assert "react-root" in root.text
    assert "/gui-static/app.jsx" in root.text
    assert "Add More Files To Knowledge Base" not in root.text
    assert options.status_code == 200
    payload = options.json()
    assert "auto" in payload["hats"]
    assert "default" in payload["hats"]
    assert "SNNs" in payload["hats"]
    assert payload["answer_modes"] == ["default", "light", "none", "raw"]


def test_gui_task_endpoints_use_existing_app_flows(tmp_path: Path, monkeypatch) -> None:
    app_home = tmp_path / ".arignan"
    app = ArignanApp(load_config(app_home=app_home))
    captured: dict[str, object] = {}

    def fake_load(input_ref: str, hat: str = "auto") -> LoadResult:
        batch_dir = Path(input_ref)
        uploaded = sorted(path.name for path in batch_dir.iterdir())
        captured["load_input_ref"] = input_ref
        captured["load_hat"] = hat
        captured["uploaded"] = uploaded
        return LoadResult(
            load_id="load-gui",
            hat=hat,
            document_count=len(uploaded),
            topic_folders=["jepa-notes"],
            artifact_paths=[],
            total_chunks=42,
            total_markdown_segments=1,
            failures=[],
            traces=[],
            model_calls=[],
        )

    def fake_ask(question: str, hat: str = "auto", terminal_pid: int | None = None, answer_mode: str = "default") -> AskResult:
        captured["question"] = question
        captured["ask_hat"] = hat
        captured["answer_mode"] = answer_mode
        return AskResult(
            question=question,
            selected_hat=hat,
            answer_mode=answer_mode,
            answer="JEPA stands for Joint Embedding Predictive Architecture.",
            citations=["default/jepa-notes/notes.md: Overview"],
            debug=AskDebug(
                answer_mode=answer_mode,
                expanded_query=question.lower(),
                selected_hat=hat,
                dense_hits=[],
                lexical_hits=[],
                map_hits=[],
                fused_hits=[],
                reranked_hits=[],
                model_calls=[],
            ),
        )

    monkeypatch.setattr(app, "load", fake_load)
    monkeypatch.setattr(app, "ask", fake_ask)
    client = TestClient(create_gui_app(app))

    load_start = client.post(
        "/api/load/start",
        data={"hat": "SNNs"},
        files=[
            ("files", ("paper.pdf", b"%PDF-1.4", "application/pdf")),
            ("files", ("notes.md", b"# Notes\n\nJEPA", "text/markdown")),
        ],
    )
    assert load_start.status_code == 200
    load_payload = _wait_for_task(client, load_start.json()["task_id"])
    assert load_payload["status"] == "done"
    assert load_payload["result"]["load_id"] == "load-gui"
    assert load_payload["result"]["document_count"] == 2
    assert captured["load_hat"] == "SNNs"
    assert captured["uploaded"] == ["001-paper.pdf", "002-notes.md"]
    assert load_payload["result"]["uploaded_files"] == ["paper.pdf", "notes.md"]

    ask_start = client.post(
        "/api/ask/start",
        json={"question": "What is JEPA?", "hat": "SNNs", "answer_mode": "light"},
    )
    assert ask_start.status_code == 200
    ask_payload = _wait_for_task(client, ask_start.json()["task_id"])
    assert ask_payload["status"] == "done"
    assert ask_payload["result"]["answer"] == "JEPA stands for Joint Embedding Predictive Architecture."
    assert ask_payload["result"]["citations"] == ["default/jepa-notes/notes.md: Overview"]
    assert captured["question"] == "What is JEPA?"
    assert captured["ask_hat"] == "SNNs"
    assert captured["answer_mode"] == "light"


def test_gui_load_rejects_unsupported_file_types(tmp_path: Path) -> None:
    app_home = tmp_path / ".arignan"
    app = ArignanApp(load_config(app_home=app_home))
    client = TestClient(create_gui_app(app))

    response = client.post(
        "/api/load",
        data={"hat": "default"},
        files=[("files", ("notes.txt", b"plain text", "text/plain"))],
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def _wait_for_task(client: TestClient, task_id: str) -> dict[str, object]:
    for _ in range(50):
        response = client.get(f"/api/tasks/{task_id}")
        payload = response.json()
        if payload["status"] != "running":
            return payload
        time.sleep(0.02)
    raise AssertionError(f"Task {task_id} did not finish in time.")
