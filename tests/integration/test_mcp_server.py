from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from arignan.application import ArignanApp
from arignan.config import load_config
from arignan.mcp import build_mcp_server


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
        response_format=None,
    ) -> str:
        if "Return strict JSON only" in system_prompt:
            return json.dumps(
                {
                    "title": "JEPA Notes",
                    "description": "Notes on Joint Embedding Predictive Architecture.",
                    "locator": "overview of JEPA architecture",
                    "keywords": ["JEPA", "representation learning"],
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
                        "| JEPA Notes | Overview of JEPA architecture | JEPA Notes | `notes.md` |\n\n"
                        "## Keywords\n"
                        "JEPA, representation learning"
                    ),
                }
            )
        if "knowledge-base hat map" in system_prompt:
            return (
                "# Map for Hat: default\n\n"
                "| Topic | Directory | What To Find | Source Files | Keywords |\n"
                "| --- | --- | --- | --- | --- |\n"
                "| JEPA Notes | `summaries/jepa-notes` | overview of JEPA architecture | notes.md | JEPA, representation learning |\n"
            )
        if "global knowledge-base map" in system_prompt:
            return (
                "# Global Map\n\n"
                "| Hat | Map Path | What To Find | High-Level Keywords |\n"
                "| --- | --- | --- | --- |\n"
                "| default | `hats/default/map.md` | overview of JEPA architecture | JEPA, representation learning |\n"
            )
        return "JEPA stands for Joint Embedding Predictive Architecture."


@pytest.mark.anyio
async def test_mcp_server_exposes_retrieval_tool_and_global_map_resource(tmp_path: Path, monkeypatch) -> None:
    app_home = tmp_path / ".arignan"
    source = tmp_path / "notes.md"
    source.write_text("# JEPA Notes\n\nJoint embedding predictive architecture overview.\n", encoding="utf-8")

    monkeypatch.setattr(
        "arignan.application.create_local_text_generator",
        lambda config, progress_sink=None, **kwargs: FakeLocalGenerator(kwargs.get("model_name") or config.local_llm_model),
    )
    app = ArignanApp(load_config(app_home=app_home))
    app.load(str(source), hat="default")
    server = build_mcp_server(app=app)

    async with create_connected_server_and_client_session(server) as session:
        await session.initialize()
        tools = await session.list_tools()
        resources = await session.list_resources()
        tool_result = await session.call_tool(
            "retrieve_context",
            {"query": "JEPA architecture", "hat": "default", "rerank_top_k": 4, "answer_context_top_k": 1},
        )
        global_map = await session.read_resource("arignan://global-map")

    tool_names = {tool.name for tool in tools.tools}
    assert "retrieve_context" in tool_names
    assert "ask" not in tool_names
    retrieve_tool = next(tool for tool in tools.tools if tool.name == "retrieve_context")
    assert "answer_context_top_k" in retrieve_tool.inputSchema["properties"]
    assert str(resources.resources[0].uri) == "arignan://global-map"
    assert tool_result.structuredContent["contexts"]
    assert len(tool_result.structuredContent["contexts"]) == 1
    assert tool_result.structuredContent["contexts"][0]["citation"].startswith("default/jepa-notes/notes.md:")
    assert global_map.contents[0].text and "default" in global_map.contents[0].text.lower()


@pytest.mark.anyio
async def test_mcp_server_ask_uses_client_backend_by_default_without_hitting_local_llm(
    tmp_path: Path, monkeypatch
) -> None:
    app_home = tmp_path / ".arignan"
    source = tmp_path / "notes.md"
    source.write_text("# JEPA Notes\n\nJoint embedding predictive architecture overview.\n", encoding="utf-8")
    app_home.mkdir(parents=True, exist_ok=True)
    (app_home / "mcp.json").write_text(
        json.dumps({"tools": {"ask": {"enabled": True}}}, indent=2) + "\n",
        encoding="utf-8",
    )

    class FailingGenerator(FakeLocalGenerator):
        def generate(self, **kwargs):
            raise AssertionError("Client-backend MCP ask should not hit the local answer generator")

    monkeypatch.setattr(
        "arignan.application.create_local_text_generator",
        lambda config, progress_sink=None, **kwargs: FailingGenerator(kwargs.get("model_name") or config.local_llm_model),
    )
    app = ArignanApp(load_config(app_home=app_home))
    app.load(str(source), hat="default")
    server = build_mcp_server(app=app, config=app.config)

    async with create_connected_server_and_client_session(server) as session:
        await session.initialize()
        tool_result = await session.call_tool(
            "ask",
            {"query": "What is JEPA?", "hat": "default", "answer_mode": "default", "answer_context_top_k": 1},
        )

    structured = tool_result.structuredContent
    assert structured["llm_backend"] == "client"
    assert structured["route"] == "retrieve"
    assert structured["contexts"]
    assert structured["messages"][-1]["role"] == "user"
    assert structured["answer"] is None


@pytest.mark.anyio
async def test_mcp_server_ask_can_use_local_backend_when_enabled(tmp_path: Path, monkeypatch) -> None:
    app_home = tmp_path / ".arignan"
    source = tmp_path / "notes.md"
    source.write_text("# JEPA Notes\n\nJoint embedding predictive architecture overview.\n", encoding="utf-8")
    app_home.mkdir(parents=True, exist_ok=True)
    (app_home / "mcp.json").write_text(
        json.dumps({"tools": {"ask": {"enabled": True}}}, indent=2) + "\n",
        encoding="utf-8",
    )

    settings = load_config(app_home=app_home)
    settings.mcp_llm_backend = "local"
    monkeypatch.setattr(
        "arignan.application.create_local_text_generator",
        lambda config, progress_sink=None, **kwargs: FakeLocalGenerator(kwargs.get("model_name") or config.local_llm_model),
    )
    app = ArignanApp(settings)
    app.load(str(source), hat="default")
    server = build_mcp_server(app=app, config=app.config)

    async with create_connected_server_and_client_session(server) as session:
        await session.initialize()
        tool_result = await session.call_tool(
            "ask",
            {"query": "What is JEPA?", "hat": "default", "answer_mode": "default"},
        )

    structured = tool_result.structuredContent
    assert structured["llm_backend"] == "local"
    assert structured["answer"] == "JEPA stands for Joint Embedding Predictive Architecture."


@pytest.mark.anyio
async def test_mcp_server_uses_mcp_json_tool_descriptions(tmp_path: Path, monkeypatch) -> None:
    app_home = tmp_path / ".arignan"
    source = tmp_path / "notes.md"
    source.write_text("# JEPA Notes\n\nJoint embedding predictive architecture overview.\n", encoding="utf-8")

    monkeypatch.setattr(
        "arignan.application.create_local_text_generator",
        lambda config, progress_sink=None, **kwargs: FakeLocalGenerator(kwargs.get("model_name") or config.local_llm_model),
    )
    mcp_path = app_home / "mcp.json"
    mcp_path.parent.mkdir(parents=True, exist_ok=True)
    mcp_path.write_text(
        json.dumps(
            {
                "tools": {
                    "retrieve_context": {
                        "description": "Custom retrieve description from mcp.json.",
                    }
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    app = ArignanApp(load_config(app_home=app_home))
    app.load(str(source), hat="default")
    server = build_mcp_server(app=app, config=app.config)

    async with create_connected_server_and_client_session(server) as session:
        await session.initialize()
        tools = await session.list_tools()

    retrieve_tool = next(tool for tool in tools.tools if tool.name == "retrieve_context")
    assert retrieve_tool.description == "Custom retrieve description from mcp.json."


@pytest.mark.anyio
async def test_mcp_server_does_not_register_disabled_tools(tmp_path: Path, monkeypatch) -> None:
    app_home = tmp_path / ".arignan"
    source = tmp_path / "notes.md"
    source.write_text("# JEPA Notes\n\nJoint embedding predictive architecture overview.\n", encoding="utf-8")

    monkeypatch.setattr(
        "arignan.application.create_local_text_generator",
        lambda config, progress_sink=None, **kwargs: FakeLocalGenerator(kwargs.get("model_name") or config.local_llm_model),
    )
    app_home.mkdir(parents=True, exist_ok=True)
    (app_home / "mcp.json").write_text(
        json.dumps({"tools": {"retrieve_context": {"enabled": False}}}, indent=2) + "\n",
        encoding="utf-8",
    )
    app = ArignanApp(load_config(app_home=app_home))
    app.load(str(source), hat="default")
    server = build_mcp_server(app=app, config=app.config)

    async with create_connected_server_and_client_session(server) as session:
        await session.initialize()
        tools = await session.list_tools()

    tool_names = {tool.name for tool in tools.tools}
    assert "retrieve_context" not in tool_names
