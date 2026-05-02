from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arignan.application import ArignanApp

@dataclass(slots=True)
class MCPTool:
    name: str
    description: str
    input_schema: dict


@dataclass(slots=True)
class MCPResource:
    uri: str
    name: str
    description: str


class ArignanMCPServer:
    def __init__(self, app: ArignanApp | None = None, *, app_factory: Callable[[], ArignanApp] | None = None) -> None:
        if app is None and app_factory is None:
            raise ValueError("ArignanMCPServer requires either an app instance or an app_factory.")
        self._app = app
        self._app_factory = app_factory

    @property
    def app(self) -> ArignanApp:
        if self._app is None:
            if self._app_factory is None:  # pragma: no cover - constructor guards this
                raise RuntimeError("Arignan MCP app factory is not configured.")
            self._app = self._app_factory()
        return self._app

    def list_tools(self) -> list[MCPTool]:
        return [
            MCPTool(
                name="retrieve_context",
                description="Retrieve local Arignan knowledge context for a query.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "hat": {"type": "string", "default": "auto"},
                        "rerank_top_k": {"type": "integer"},
                        "answer_context_top_k": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            )
        ]

    def call_tool(self, name: str, arguments: dict) -> dict:
        from arignan.application import format_citation

        if name != "retrieve_context":
            raise ValueError(f"unknown MCP tool: {name}")
        query = arguments["query"]
        hat = arguments.get("hat", "auto")
        result = self.app.retrieve_context(
            query,
            hat=hat,
            rerank_top_k=arguments.get("rerank_top_k"),
            answer_context_top_k=arguments.get("answer_context_top_k"),
        )
        return {
            "query": query,
            "selected_hat": result.selected_hat,
            "expanded_query": result.expanded_query,
            "contexts": [
                {
                    "text": hit.text,
                    "source": hit.source.value,
                    "citation": format_citation(hit),
                }
                for hit in result.answer_hits
            ],
        }

    def list_resources(self) -> list[MCPResource]:
        return [
            MCPResource(
                uri="arignan://global-map",
                name="global_map.md",
                description="High-level map across all hats in the local knowledge base.",
            )
        ]

    def read_resource(self, uri: str) -> str:
        if uri != "arignan://global-map":
            raise ValueError(f"unknown MCP resource: {uri}")
        return self.app.layout.global_map_path.read_text(encoding="utf-8")
