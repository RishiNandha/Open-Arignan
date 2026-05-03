from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.anyio
async def test_mcp_streamable_http_entrypoint_initializes_and_serves_tools(tmp_path: Path) -> None:
    app_home = tmp_path / ".arignan"
    port = _reserve_port()
    process = _spawn_mcp_http_process(app_home=app_home, port=port)
    try:
        base_url = f"http://127.0.0.1:{port}/mcp"
        last_exc: Exception | None = None
        for _ in range(50):
            try:
                async with streamable_http_client(base_url) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        tool_result = await session.call_tool(
                            "retrieve_context",
                            {"query": "What is JEPA?", "hat": "default"},
                        )
                        tool_names = {tool.name for tool in tools.tools}
                        assert "retrieve_context" in tool_names
                        structured = tool_result.structuredContent
                        assert structured["selected_hat"] == "default"
                        assert structured["contexts"][0]["citation"] == "default/jepa-notes/notes.md: Overview"
                        break
            except Exception as exc:  # pragma: no cover - retry path depends on server timing
                last_exc = exc
                await anyio.sleep(0.1)
        else:
            stderr_text = _stop_process(process)
            raise AssertionError(f"HTTP MCP server did not become ready. stderr:\n{stderr_text}") from last_exc
    finally:
        stderr_text = _stop_process(process)

    assert f"[arignan-mcp] Server started on http://127.0.0.1:{port}/mcp" in stderr_text


def _spawn_mcp_http_process(*, app_home: Path, port: int) -> subprocess.Popen[bytes]:
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from arignan.cli import main\n\n"
        f"{textwrap.dedent(_extra_script()).strip()}\n\n"
        "raise SystemExit(main(['--mcp-http', '--mcp-host', '127.0.0.1', '--mcp-port', sys.argv[2], '--app-home', sys.argv[1]]))\n"
    )
    return subprocess.Popen(
        [sys.executable, "-c", script, str(app_home), str(port)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
    )


def _extra_script() -> str:
    return """
import arignan.application as application
import arignan.mcp.server as mcp_server
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
mcp_server._LazyArignanApp.background_load_retrieval_models = lambda self: None

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
"""


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
