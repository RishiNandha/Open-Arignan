from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

if TYPE_CHECKING:
    from arignan.application import ArignanApp


class RetrievedContext(BaseModel):
    text: str
    source: str
    citation: str


class RetrieveContextResult(BaseModel):
    query: str
    selected_hat: str
    expanded_query: str
    contexts: list[RetrievedContext]


@dataclass(slots=True)
class _LazyArignanApp:
    app: ArignanApp | None
    app_factory: Callable[[], ArignanApp] | None
    progress_sink: Callable[[str], None] | None

    def resolve(self) -> ArignanApp:
        if self.app is None:
            if self.app_factory is None:  # pragma: no cover - guarded by builder
                raise RuntimeError("Arignan MCP app factory is not configured.")
            self.progress("Loading Arignan app")
            self.app = self.app_factory()
        return self.app

    def progress(self, message: str) -> None:
        if self.progress_sink is not None:
            self.progress_sink(message)


def build_mcp_server(
    app: ArignanApp | None = None,
    *,
    app_factory: Callable[[], ArignanApp] | None = None,
    progress_sink: Callable[[str], None] | None = None,
) -> FastMCP:
    if app is None and app_factory is None:
        raise ValueError("build_mcp_server requires either an app instance or an app_factory.")

    state = _LazyArignanApp(app=app, app_factory=app_factory, progress_sink=progress_sink)
    mcp = FastMCP("arignan", log_level="INFO")

    @mcp.tool(
        name="retrieve_context",
        description="Retrieve local Arignan knowledge context for a query without calling an answer LLM.",
    )
    def retrieve_context(
        query: str,
        hat: str = "auto",
        rerank_top_k: int | None = None,
        answer_context_top_k: int | None = None,
    ) -> RetrieveContextResult:
        from arignan.application import format_citation

        state.progress(f"Running retrieve_context for query={query!r}")
        result = state.resolve().retrieve_context(
            query,
            hat=hat,
            rerank_top_k=rerank_top_k,
            answer_context_top_k=answer_context_top_k,
        )
        return RetrieveContextResult(
            query=query,
            selected_hat=result.selected_hat,
            expanded_query=result.expanded_query,
            contexts=[
                RetrievedContext(
                    text=hit.text,
                    source=hit.source.value,
                    citation=format_citation(hit),
                )
                for hit in result.answer_hits
            ],
        )

    @mcp.resource(
        "arignan://global-map",
        name="global_map.md",
        description="High-level map across all hats in the local knowledge base.",
        mime_type="text/markdown",
    )
    def global_map() -> str:
        state.progress("Reading global knowledge map")
        return state.resolve().layout.global_map_path.read_text(encoding="utf-8")

    return mcp
