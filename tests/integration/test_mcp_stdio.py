from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_mcp_stdio_entrypoint_initializes_and_serves_tools(tmp_path: Path) -> None:
    app_home = tmp_path / ".arignan"
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from types import SimpleNamespace

        import arignan.application as application
        from arignan.application import RetrievalResult
        from arignan.models import ChunkMetadata, RetrievalHit, RetrievalSource
        from arignan.cli import main

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
        sys.exit(main(["--mcp", "--app-home", sys.argv[1]]))
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(app_home)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
    )
    try:
        initialize = _request(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "pytest"}},
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
        process.terminate()
        process.communicate(timeout=10)


def test_mcp_stdio_initialize_returns_before_app_construction(tmp_path: Path) -> None:
    app_home = tmp_path / ".arignan"
    script = textwrap.dedent(
        """
        import sys
        from arignan.cli import main
        import arignan.application as application

        class ExplodingApp:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("ArignanApp should not be constructed during initialize")

        application.ArignanApp = ExplodingApp
        sys.exit(main(["--mcp", "--app-home", sys.argv[1]]))
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(app_home)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
    )
    try:
        initialize = _request(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "pytest"}},
            },
        )
        assert initialize["result"]["serverInfo"]["name"] == "arignan"
    finally:
        process.terminate()
        process.communicate(timeout=10)


def test_mcp_stdio_logs_progress_to_stderr(tmp_path: Path) -> None:
    app_home = tmp_path / ".arignan"
    script = "from arignan.cli import main; raise SystemExit(main(['--mcp', '--app-home', r'%s']))" % str(app_home)
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
    )
    try:
        initialize = _request(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "pytest"}},
            },
        )
        assert initialize["result"]["serverInfo"]["name"] == "arignan"

        tools = _request(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        assert "tools" in tools["result"]
    finally:
        process.terminate()
        _, stderr = process.communicate(timeout=10)

    stderr_text = stderr.decode("utf-8", errors="replace")
    assert "[arignan-mcp] Server started" in stderr_text
    assert "[arignan-mcp] Received headers:" in stderr_text
    assert '"method": "initialize"' in stderr_text
    assert "[arignan-mcp] Initialize request received" in stderr_text
    assert "[arignan-mcp] Listing tools" in stderr_text


def _request(process: subprocess.Popen[bytes], payload: dict[str, object]) -> dict[str, object]:
    assert process.stdin is not None
    body = json.dumps(payload).encode("utf-8")
    process.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    process.stdin.write(body)
    process.stdin.flush()
    return _read_message(process)


def _read_message(process: subprocess.Popen[bytes]) -> dict[str, object]:
    assert process.stdout is not None
    headers: dict[str, str] = {}
    while True:
        line = process.stdout.readline()
        if line == b"":
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            raise RuntimeError(f"MCP server closed unexpectedly. stderr:\n{stderr}")
        text = line.decode("utf-8").strip()
        if not text:
            break
        name, value = text.split(":", maxsplit=1)
        headers[name.strip().lower()] = value.strip()
    length = int(headers["content-length"])
    payload = process.stdout.read(length)
    return json.loads(payload.decode("utf-8"))
