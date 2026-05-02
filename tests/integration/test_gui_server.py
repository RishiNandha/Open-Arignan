from __future__ import annotations

from pathlib import Path
import time

from fastapi.testclient import TestClient

from arignan.application import ArignanApp, AskDebug, AskResult, DeleteHatResult, DeleteResult, LoadResult
from arignan.config import load_config
from arignan.gui import create_gui_app
from arignan.models import LoadEvent, LoadOperation


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
    assert payload["default_rerank_top_k"] == 8


def test_gui_task_endpoints_use_existing_app_flows(tmp_path: Path, monkeypatch) -> None:
    app_home = tmp_path / ".arignan"
    app = ArignanApp(load_config(app_home=app_home))
    captured: dict[str, object] = {}
    fake_events = [
        LoadEvent(
            load_id="load-gui",
            operation=LoadOperation.INGEST,
            hat="SNNs",
            created_at="2026-04-09T12:00:00+00:00",
            source_items=["paper.pdf", "notes.md"],
            topic_folders=["jepa-notes"],
        )
    ]

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

    def fake_ask(
        question: str,
        hat: str = "auto",
        terminal_pid: int | None = None,
        answer_mode: str = "default",
        rerank_top_k: int | None = None,
    ) -> AskResult:
        captured["question"] = question
        captured["ask_hat"] = hat
        captured["answer_mode"] = answer_mode
        captured["rerank_top_k"] = rerank_top_k
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

    def fake_list_ingestions():
        return fake_events

    def fake_list_live_ingestions():
        return fake_events

    def fake_delete(load_ids):
        captured["deleted_load_ids"] = list(load_ids)
        return DeleteResult(deleted_load_ids=list(load_ids), missing_load_ids=[], deleted_topics=["jepa-notes"])

    def fake_delete_hat(hat: str):
        captured["deleted_hat"] = hat
        return DeleteHatResult(hat=hat, existed=True, deleted_load_ids=["load-gui"], deleted_topics=["jepa-notes"])

    monkeypatch.setattr(app, "load", fake_load)
    monkeypatch.setattr(app, "ask", fake_ask)
    monkeypatch.setattr(app, "list_ingestions", fake_list_ingestions)
    monkeypatch.setattr(app, "list_live_ingestions", fake_list_live_ingestions)
    monkeypatch.setattr(app, "delete", fake_delete)
    monkeypatch.setattr(app, "delete_hat", fake_delete_hat)
    client = TestClient(create_gui_app(app))

    library = client.get("/api/library")
    assert library.status_code == 200
    library_payload = library.json()
    assert "default" in library_payload["hats"]
    assert library_payload["loads"][0]["load_id"] == "load-gui"

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
        json={"question": "What is JEPA?", "hat": "SNNs", "answer_mode": "light", "rerank_top_k": 11},
    )
    assert ask_start.status_code == 200
    ask_payload = _wait_for_task(client, ask_start.json()["task_id"])
    assert ask_payload["status"] == "done"
    assert ask_payload["result"]["answer"] == "JEPA stands for Joint Embedding Predictive Architecture."
    assert ask_payload["result"]["citations"] == ["default/jepa-notes/notes.md: Overview"]
    assert captured["question"] == "What is JEPA?"
    assert captured["ask_hat"] == "SNNs"
    assert captured["answer_mode"] == "light"
    assert captured["rerank_top_k"] == 11

    delete_start = client.post("/api/delete/start", json={"load_ids": ["load-gui"]})
    assert delete_start.status_code == 200
    delete_payload = _wait_for_task(client, delete_start.json()["task_id"])
    assert delete_payload["status"] == "done"
    assert delete_payload["result"]["kind"] == "load_delete"
    assert captured["deleted_load_ids"] == ["load-gui"]

    delete_hat_start = client.post("/api/delete/start", json={"hat": "SNNs"})
    assert delete_hat_start.status_code == 200
    delete_hat_payload = _wait_for_task(client, delete_hat_start.json()["task_id"])
    assert delete_hat_payload["status"] == "done"
    assert delete_hat_payload["result"]["kind"] == "hat_delete"
    assert captured["deleted_hat"] == "SNNs"


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


def test_gui_library_filters_stale_loads_for_missing_hat(tmp_path: Path) -> None:
    app_home = tmp_path / ".arignan"
    app = ArignanApp(load_config(app_home=app_home))
    stale_event = LoadEvent(
        load_id="load-stale",
        operation=LoadOperation.INGEST,
        hat="SNNs",
        created_at="2026-04-11T12:00:00+00:00",
        source_items=["paper.pdf"],
        topic_folders=["word2vec"],
    )
    app.ingestion_log.append(stale_event)

    client = TestClient(create_gui_app(app))
    library = client.get("/api/library")

    assert library.status_code == 200
    payload = library.json()
    assert all(item["load_id"] != "load-stale" for item in payload["loads"])


def test_gui_task_failure_returns_session_log_hint(tmp_path: Path, monkeypatch) -> None:
    app_home = tmp_path / ".arignan"
    app = ArignanApp(load_config(app_home=app_home))

    def failing_delete(load_ids):
        raise RuntimeError("delete exploded")

    monkeypatch.setattr(app, "delete", failing_delete)
    client = TestClient(create_gui_app(app))

    delete_start = client.post("/api/delete/start", json={"load_ids": ["load-gui"]})
    assert delete_start.status_code == 200
    payload = _wait_for_task(client, delete_start.json()["task_id"])

    assert payload["status"] == "error"
    assert "Something went wrong while deleting the selected loads." in payload["error"]
    assert "exceptions.log" in payload["error"]


def test_gui_ask_task_exposes_progress_log_and_partial_answer(tmp_path: Path, monkeypatch) -> None:
    app_home = tmp_path / ".arignan"
    app = ArignanApp(load_config(app_home=app_home))

    def fake_ask(question: str, hat: str = "auto", terminal_pid: int | None = None, answer_mode: str = "default", rerank_top_k: int | None = None) -> AskResult:
        app.progress_sink("Running retrieval pipeline...")
        app.progress_sink("Reranking")
        stream_sink = getattr(app.local_text_generator, "stream_sink", None)
        if callable(stream_sink):
            stream_sink("Draft answer ")
            time.sleep(0.05)
            stream_sink("in progress.")
        time.sleep(0.05)
        return AskResult(
            question=question,
            selected_hat=hat,
            answer_mode=answer_mode,
            answer="Draft answer in progress.",
            citations=[],
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

    monkeypatch.setattr(app, "ask", fake_ask)
    client = TestClient(create_gui_app(app))

    ask_start = client.post(
        "/api/ask/start",
        json={"question": "What is JEPA?", "hat": "default", "answer_mode": "default"},
    )
    assert ask_start.status_code == 200
    task_id = ask_start.json()["task_id"]

    running_snapshot = _wait_for_running_task(client, task_id)

    assert running_snapshot["status"] == "running"
    assert "Reranking" in running_snapshot["progress_log"]
    assert running_snapshot["partial_answer"].startswith("Draft answer")

    done_snapshot = _wait_for_task(client, task_id)
    assert done_snapshot["status"] == "done"
    assert done_snapshot["result"]["answer"] == "Draft answer in progress."


def _wait_for_task(client: TestClient, task_id: str) -> dict[str, object]:
    for _ in range(50):
        response = client.get(f"/api/tasks/{task_id}")
        payload = response.json()
        if payload["status"] != "running":
            return payload
        time.sleep(0.02)
    raise AssertionError(f"Task {task_id} did not finish in time.")


def _wait_for_running_task(client: TestClient, task_id: str) -> dict[str, object]:
    for _ in range(50):
        response = client.get(f"/api/tasks/{task_id}")
        payload = response.json()
        if payload["status"] == "running" and payload.get("partial_answer"):
            return payload
        time.sleep(0.02)
    raise AssertionError(f"Task {task_id} did not expose partial answer in time.")
