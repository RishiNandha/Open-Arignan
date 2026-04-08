from __future__ import annotations

import json
from pathlib import Path

from arignan.application import ArignanApp
from arignan.config import load_config
from arignan.mcp import ArignanMCPServer


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


def test_mcp_server_exposes_retrieval_tool_and_global_map_resource(tmp_path: Path, monkeypatch) -> None:
    app_home = tmp_path / ".arignan"
    source = tmp_path / "notes.md"
    source.write_text("# JEPA Notes\n\nJoint embedding predictive architecture overview.\n", encoding="utf-8")

    monkeypatch.setattr(
        "arignan.application.create_local_text_generator",
        lambda config, progress_sink=None, **kwargs: FakeLocalGenerator(kwargs.get("model_name") or config.local_llm_model),
    )
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
