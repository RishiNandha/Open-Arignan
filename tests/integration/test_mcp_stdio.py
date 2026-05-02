from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path


def test_mcp_stdio_entrypoint_initializes_and_serves_tools(tmp_path: Path) -> None:
    app_home = tmp_path / ".arignan"
    process = _spawn_mcp_process(
        app_home=app_home,
        extra_script="""
import arignan.application as application
from arignan.application import RetrievalResult
from arignan.models import ChunkMetadata, RetrievalHit, RetrievalSource

class DummyGenerator:
    model_name = "qwen3:4b-q4_K_M"
    backend_name = "fake-local"
    progress_sink = None
    stream_sink = None
    thinking_sink = None

    def generate(self, **kwargs):
        return "unused"

class DummyReranker:
    model_name = "mixedbread-ai/mxbai-rerank-base-v1"
    backend_name = "fake-reranker"

    def rerank(self, query, hits, limit, min_score=0.0):
        return hits[:limit]

application.create_embedder = lambda *args, **kwargs: object()
application.create_reranker = lambda *args, **kwargs: DummyReranker()
application.create_local_text_generator = lambda *args, **kwargs: DummyGenerator()

def fake_retrieve_context(self, question, *, hat="auto", rerank_top_k=None, answer_context_top_k=None):
    hit = RetrievalHit(
        chunk_id="chunk-1",
        text="Joint Embedding Predictive Architecture overview.",
        score=0.9,
        source=RetrievalSource.DENSE,
        metadata=ChunkMetadata(
            load_id="load-1",
            hat="default",
            source_uri="notes.md",
            source_path=Path("notes.md"),
            heading="Overview",
            section="Overview",
            topic_folder="jepa-notes",
        ),
    )
    return RetrievalResult(
        question=question,
        selected_hat="default",
        expanded_query=question.lower(),
        dense_hits=[hit],
        lexical_hits=[],
        map_hits=[],
        fused_hits=[hit],
        reranked_hits=[hit],
        answer_hits=[hit],
    )

application.ArignanApp.retrieve_context = fake_retrieve_context
""",
    )
    try:
        initialize = _request(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "1.0.0"},
                },
            },
        )
        assert initialize["result"]["serverInfo"]["name"] == "arignan"

        tools = _request(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        assert any(tool["name"] == "retrieve_context" for tool in tools["result"]["tools"])

        tool_call = _request(
            process,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "retrieve_context", "arguments": {"query": "What is JEPA?"}},
            },
        )
        structured = tool_call["result"]["structuredContent"]
        assert structured["selected_hat"] == "default"
        assert structured["contexts"][0]["citation"] == "default/jepa-notes/notes.md: Overview"
    finally:
        _stop_process(process)


def test_mcp_stdio_initialize_returns_before_app_construction(tmp_path: Path) -> None:
    app_home = tmp_path / ".arignan"
    process = _spawn_mcp_process(
        app_home=app_home,
        extra_script="""
import arignan.application as application

class ExplodingApp:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("ArignanApp should not be constructed during initialize")

application.ArignanApp = ExplodingApp
""",
    )
    try:
        initialize = _request(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "1.0.0"},
                },
            },
        )
        assert initialize["result"]["serverInfo"]["name"] == "arignan"
    finally:
        _stop_process(process)


def test_mcp_stdio_logs_progress_to_stderr(tmp_path: Path) -> None:
    app_home = tmp_path / ".arignan"
    process = _spawn_mcp_process(app_home=app_home, extra_script="")
    try:
        initialize = _request(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "1.0.0"},
                },
            },
        )
        assert initialize["result"]["serverInfo"]["name"] == "arignan"
    finally:
        stderr_text = _stop_process(process)

    assert "[arignan-mcp] Server started" in stderr_text


def _spawn_mcp_process(*, app_home: Path, extra_script: str) -> subprocess.Popen[bytes]:
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from arignan.cli import main\n\n"
        f"{textwrap.dedent(extra_script).strip()}\n\n"
        "raise SystemExit(main(['--mcp', '--app-home', sys.argv[1]]))\n"
    )
    return subprocess.Popen(
        [sys.executable, "-c", script, str(app_home)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
    )


def _request(process: subprocess.Popen[bytes], payload: dict[str, object], *, timeout_seconds: float = 5.0) -> dict[str, object]:
    assert process.stdin is not None
    process.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
    process.stdin.flush()
    return _read_response(process, request_id=payload["id"], timeout_seconds=timeout_seconds)


def _read_response(process: subprocess.Popen[bytes], *, request_id: object, timeout_seconds: float) -> dict[str, object]:
    result: dict[str, object] = {}
    error: list[BaseException] = []

    def _worker() -> None:
        try:
            assert process.stdout is not None
            while True:
                line = process.stdout.readline()
                if line == b"":
                    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
                    raise RuntimeError(f"MCP server closed unexpectedly. stderr:\n{stderr}")
                message = json.loads(line.decode("utf-8"))
                if message.get("id") == request_id:
                    result.update(message)
                    return
        except BaseException as exc:  # pragma: no cover - timeout path handles details
            error.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        raise TimeoutError(f"MCP response timed out after {timeout_seconds} seconds. stderr:\n{stderr}")
    if error:
        raise error[0]
    return result


def _stop_process(process: subprocess.Popen[bytes]) -> str:
    try:
        process.terminate()
    except Exception:
        pass
    try:
        _, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        _, stderr = process.communicate(timeout=5)
    return stderr.decode("utf-8", errors="replace")
