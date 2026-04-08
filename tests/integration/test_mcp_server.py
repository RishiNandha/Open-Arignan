from __future__ import annotations

from pathlib import Path

from arignan.application import ArignanApp
from arignan.config import load_config
from arignan.mcp import ArignanMCPServer


def test_mcp_server_exposes_retrieval_tool_and_global_map_resource(tmp_path: Path) -> None:
    app_home = tmp_path / ".arignan"
    source = tmp_path / "notes.md"
    source.write_text("# JEPA Notes\n\nJoint embedding predictive architecture overview.\n", encoding="utf-8")

    app = ArignanApp(load_config(app_home=app_home))
    app.load(str(source), hat="default")
    server = ArignanMCPServer(app)

    tools = server.list_tools()
    resources = server.list_resources()
    tool_result = server.call_tool("retrieve_context", {"query": "JEPA architecture", "hat": "default"})
    global_map = server.read_resource("arignan://global-map")

    assert tools[0].name == "retrieve_context"
    assert resources[0].uri == "arignan://global-map"
    assert tool_result["contexts"]
    assert "default" in global_map.lower()
