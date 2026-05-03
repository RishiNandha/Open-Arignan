from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import threading
import time

import anyio
import pytest
from fastapi.testclient import TestClient
from mcp.shared.memory import create_connected_server_and_client_session

from arignan.application import ArignanApp, AskDebug, AskResult, RetrievalResult
from arignan.config import load_config
from arignan.gui import create_gui_app
from arignan.mcp import build_mcp_server
from arignan.models import ChunkMetadata, RetrievalHit, RetrievalSource


class FakeSharedModel:
    def __init__(self, name: str) -> None:
        self.model_name = name
        self.backend_name = f"fake-{name}"
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls: list[str] = []

    @contextmanager
    def use(self, owner: str):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append(owner)
            try:
                yield
            finally:
                self.active -= 1

    def release_device_memory(self) -> bool:
        return True


class FakeLocalGenerator:
    backend_name = "fake-local"

    def __init__(self, model_name: str, progress_sink=None) -> None:
        self.model_name = model_name
        self.progress_sink = progress_sink
        self.stream_sink = None
        self.thinking_sink = None
        self.last_usage = None

    def generate(self, **kwargs):  # pragma: no cover - not needed in this test
        return "unused"


def _make_hit() -> RetrievalHit:
    return RetrievalHit(
        chunk_id="chunk-1",
        text="JEPA is a joint embedding predictive architecture.",
        score=1.0,
        source=RetrievalSource.DENSE,
        metadata=ChunkMetadata(
            load_id="load-1",
            hat="default",
            source_uri="notes.md",
            source_path=Path("notes.md"),
            page_number=1,
            section="Overview",
            heading="Overview",
            topic_folder="jepa-notes",
        ),
        rank=1,
        extras={"rerank_score": 1.0},
    )


def _wait_for_task(client: TestClient, task_id: str) -> dict[str, object]:
    for _ in range(100):
        response = client.get(f"/api/tasks/{task_id}")
        payload = response.json()
        if payload["status"] != "running":
            return payload
        time.sleep(0.02)
    raise AssertionError(f"Task {task_id} did not finish in time.")


@pytest.mark.anyio
async def test_gui_and_first_mcp_retrieve_context_share_fake_models_without_deadlock(tmp_path: Path, monkeypatch) -> None:
    app_home = tmp_path / ".arignan"
    embedder = FakeSharedModel("embedder")
    reranker = FakeSharedModel("reranker")

    monkeypatch.setattr("arignan.application.create_embedder", lambda config, **kwargs: embedder)
    monkeypatch.setattr("arignan.application.create_reranker", lambda config, **kwargs: reranker)
    monkeypatch.setattr(
        "arignan.application.create_local_text_generator",
        lambda config, progress_sink=None, **kwargs: FakeLocalGenerator(kwargs.get("model_name") or config.local_llm_model, progress_sink=progress_sink),
    )
    monkeypatch.setattr("arignan.mcp.server._LazyArignanApp.background_load_retrieval_models", lambda self: None)

    app = ArignanApp(load_config(app_home=app_home))
    gui_started = threading.Event()
    release_gui = threading.Event()

    def fake_ask(
        question: str,
        hat: str = "auto",
        terminal_pid: int | None = None,
        answer_mode: str = "default",
        rerank_top_k: int | None = None,
        answer_context_top_k: int | None = None,
    ) -> AskResult:
        gui_started.set()
        with embedder.use("gui"):
            with reranker.use("gui"):
                release_gui.wait(timeout=1.0)
        return AskResult(
            question=question,
            selected_hat=hat,
            answer_mode=answer_mode,
            answer="GUI answer complete.",
            citations=["default/jepa-notes/notes.md: Page 1, Overview"],
            debug=AskDebug(
                answer_mode=answer_mode,
                expanded_query=question,
                selected_hat=hat,
                dense_hits=[],
                lexical_hits=[],
                map_hits=[],
                fused_hits=[],
                reranked_hits=[],
                model_calls=[],
            ),
        )

    def fake_retrieve_context(
        question: str,
        *,
        hat: str = "auto",
        rerank_top_k: int | None = None,
        answer_context_top_k: int | None = None,
    ) -> RetrievalResult:
        with embedder.use("mcp"):
            with reranker.use("mcp"):
                hit = _make_hit()
                return RetrievalResult(
                    question=question,
                    selected_hat=hat,
                    expanded_query=question,
                    dense_hits=[hit],
                    lexical_hits=[],
                    map_hits=[],
                    fused_hits=[hit],
                    reranked_hits=[hit],
                    answer_hits=[hit],
                )

    monkeypatch.setattr(app, "ask", fake_ask)
    monkeypatch.setattr(app, "retrieve_context", fake_retrieve_context)

    client = TestClient(create_gui_app(app))
    ask_start = client.post(
        "/api/ask/start",
        json={"question": "Explain JEPA", "hat": "default", "answer_mode": "default"},
    )
    assert ask_start.status_code == 200
    task_id = ask_start.json()["task_id"]
    assert gui_started.wait(timeout=1.0)

    server = build_mcp_server(app=None, app_factory=lambda: app, config=app.config)

    async with create_connected_server_and_client_session(server) as session:
        await session.initialize()
        result_holder: dict[str, object] = {}
        mcp_done = anyio.Event()

        async def call_tool() -> None:
            result_holder["result"] = await session.call_tool(
                "retrieve_context",
                {"query": "Find out from the local library what JEPA is.", "hat": "default"},
            )
            mcp_done.set()

        async with anyio.create_task_group() as tg:
            tg.start_soon(call_tool)
            await anyio.sleep(0.05)
            assert not mcp_done.is_set()
            release_gui.set()

    gui_result = _wait_for_task(client, task_id)
    assert gui_result["status"] == "done"
    assert gui_result["result"]["answer"] == "GUI answer complete."

    tool_result = result_holder["result"]
    assert tool_result.structuredContent["contexts"][0]["text"].startswith("JEPA is")
    assert embedder.calls == ["gui", "mcp"]
    assert reranker.calls == ["gui", "mcp"]
    assert embedder.max_active == 1
    assert reranker.max_active == 1
