from __future__ import annotations

from dataclasses import dataclass, field
from io import TextIOWrapper
from pathlib import Path
import sys
import threading
import time
from typing import TYPE_CHECKING, Literal
from contextlib import contextmanager

import anyio
import mcp.types as types
from mcp.server.fastmcp import FastMCP
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.shared.message import SessionMessage
from pydantic import BaseModel

from arignan.application import _build_answer_prompt, _build_no_context_answer_prompt, _chat_messages_for_session, format_citation
from arignan.compute import format_torch_cuda_memory
from arignan.config import AppConfig
from arignan.mcp_config import load_mcp_config
from arignan.models import LoadEvent
from arignan.prompts import render_prompt_template

if TYPE_CHECKING:
    from collections.abc import Callable

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


class McpChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class AskToolResult(BaseModel):
    query: str
    selected_hat: str
    llm_backend: Literal["client", "local"]
    route: Literal["retrieve", "chat_context"]
    answer_mode: str
    expanded_query: str | None = None
    answer: str | None = None
    citations: list[str] = []
    contexts: list[RetrievedContext] = []
    system_prompt: str | None = None
    user_prompt: str | None = None
    messages: list[McpChatMessage] = []
    note: str | None = None


class LoadContentResult(BaseModel):
    load_id: str
    hat: str
    document_count: int
    topic_folders: list[str]
    total_chunks: int
    total_markdown_segments: int
    failures: list[str]


class LoadEventResult(BaseModel):
    created_at: str
    operation: str
    load_id: str
    hat: str
    source_items: list[str]


class ListLoadsResult(BaseModel):
    items: list[LoadEventResult]


class DeleteLoadsResult(BaseModel):
    deleted_load_ids: list[str]
    missing_load_ids: list[str]
    deleted_topics: list[str]


class DeleteHatToolResult(BaseModel):
    hat: str
    existed: bool
    deleted_load_ids: list[str]
    deleted_topics: list[str]


@dataclass(slots=True)
class _LazyArignanApp:
    app: ArignanApp | None
    app_factory: Callable[[], ArignanApp] | None
    progress_sink: Callable[[str], None] | None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _retrieval_gate: threading.Semaphore = field(default_factory=lambda: threading.Semaphore(1))
    _active_operations: int = 0
    _operation_counter: int = 0
    _init_in_progress: bool = False
    _init_event: threading.Event = field(default_factory=threading.Event)
    _init_error: BaseException | None = None

    def resolve(self) -> ArignanApp:
        started = time.perf_counter()
        thread_name = threading.current_thread().name
        self.progress(f"[thread={thread_name}] App resolve requested")
        while True:
            wait_event: threading.Event | None = None
            initialize_here = False
            with self._lock:
                if self.app is not None:
                    app = self.app
                    self.progress(f"[thread={thread_name}] Reusing existing Arignan app")
                    break
                if self.app_factory is None:  # pragma: no cover - guarded by builder
                    raise RuntimeError("Arignan MCP app factory is not configured.")
                if self._init_in_progress:
                    wait_event = self._init_event
                    self.progress(f"[thread={thread_name}] Waiting for app init to finish")
                else:
                    self._init_in_progress = True
                    self._init_event = threading.Event()
                    self._init_error = None
                    initialize_here = True
                    wait_event = self._init_event
                    self.progress(f"[thread={thread_name}] App init started")
            if initialize_here:
                try:
                    created_app = self.app_factory()
                except Exception as exc:
                    with self._lock:
                        self._init_in_progress = False
                        self._init_error = exc
                        wait_event.set()
                    self.progress(
                        f"[thread={thread_name}] App init failed after {time.perf_counter() - started:.2f}s: {exc}"
                    )
                    raise
                with self._lock:
                    self.app = created_app
                    self._init_in_progress = False
                    self._init_error = None
                    wait_event.set()
                    app = created_app
                self.progress(
                    f"[thread={thread_name}] Arignan app loaded in {time.perf_counter() - started:.2f}s"
                )
                gpu_snapshot = (
                    format_torch_cuda_memory("GPU allocated after Arignan app load") if "torch" in sys.modules else None
                )
                if gpu_snapshot is not None:
                    self.progress(gpu_snapshot)
                break
            assert wait_event is not None
            wait_event.wait()
            with self._lock:
                if self.app is not None:
                    app = self.app
                    self.progress(f"[thread={thread_name}] App init finished; reusing initialized app")
                    break
                init_error = self._init_error
            if init_error is not None:
                raise init_error
        self.progress(f"[thread={thread_name}] App resolve finished in {time.perf_counter() - started:.2f}s")
        return app

    def background_load_retrieval_models(self) -> None:
        started = time.perf_counter()
        self.progress("Starting background MCP retrieval-model load")
        with self.retrieval_usage("background_retrieval_model_load") as app:
            warm = getattr(app, "warm_retrieval_models", None)
            if callable(warm):
                self.progress("Running background retrieval-model load")
                warm()
        self.progress(f"Background retrieval-model load finished in {time.perf_counter() - started:.2f}s")

    @contextmanager
    def retrieval_usage(self, label: str = "retrieval"):
        started = time.perf_counter()
        thread_name = threading.current_thread().name
        self.progress(f"[thread={thread_name}] {label}: waiting for retrieval gate")
        self._retrieval_gate.acquire()
        gate_acquired_at = time.perf_counter()
        self.progress(
            f"[thread={thread_name}] {label}: acquired retrieval gate in {gate_acquired_at - started:.2f}s"
        )
        with self._lock:
            self._operation_counter += 1
            operation_id = self._operation_counter
            active_before = self._active_operations
            self.progress(
                f"[op={operation_id} thread={thread_name}] {label}: entering retrieval usage "
                f"(active_before={active_before})"
            )
            self._active_operations += 1
        try:
            resolved_app = self.resolve()
            self.progress(
                f"[op={operation_id} thread={thread_name}] {label}: retrieval app ready "
                f"in {time.perf_counter() - started:.2f}s"
            )
            yield resolved_app
        finally:
            try:
                with self._lock:
                    self._active_operations = max(0, self._active_operations - 1)
                    active_after = self._active_operations
                self.progress(
                    f"[op={operation_id} thread={thread_name}] {label}: leaving retrieval usage "
                    f"(active_after={active_after}, duration={time.perf_counter() - started:.2f}s)"
                )
            finally:
                self._retrieval_gate.release()
                self.progress(f"[thread={thread_name}] {label}: released retrieval gate")

    def progress(self, message: str) -> None:
        if self.progress_sink is not None:
            self.progress_sink(message)

    def release_retrieval_models(self, reason: str) -> None:
        app = self.app
        if app is None:
            return
        self.progress(reason)
        released_any = False
        for label, component in (("embedding", getattr(app, "embedder", None)), ("reranking", getattr(app, "reranker", None))):
            release = getattr(component, "release_device_memory", None)
            if callable(release):
                try:
                    if release():
                        self.progress(f"Released {label} model from GPU")
                        released_any = True
                except Exception as exc:  # pragma: no cover - best effort
                    self.progress(f"Non-fatal {label} GPU release error: {exc}")
        if released_any and "torch" in sys.modules:
            gpu_snapshot = format_torch_cuda_memory("GPU after MCP retrieval-model release")
            if gpu_snapshot is not None:
                self.progress(gpu_snapshot)


def build_mcp_server(
    app: ArignanApp | None = None,
    *,
    config: AppConfig | None = None,
    app_factory: Callable[[], ArignanApp] | None = None,
    progress_sink: Callable[[str], None] | None = None,
    host: str | None = None,
    port: int | None = None,
) -> FastMCP:
    if app is None and app_factory is None:
        raise ValueError("build_mcp_server requires either an app instance or an app_factory.")
    effective_config = config or (app.config if app is not None else None)
    if effective_config is None:
        raise ValueError("build_mcp_server requires config when only app_factory is provided.")

    mcp_config = load_mcp_config(effective_config.app_home)
    state = _LazyArignanApp(
        app=app,
        app_factory=app_factory,
        progress_sink=progress_sink,
    )
    mcp = FastMCP(
        mcp_config.server_name,
        instructions=mcp_config.instructions,
        log_level="INFO",
        host=host or "127.0.0.1",
        port=port or 8000,
    )

    if mcp_config.tools["retrieve_context"].enabled:
        @mcp.tool(
            name="retrieve_context",
            description=mcp_config.tools["retrieve_context"].description,
        )
        def retrieve_context(
            query: str,
            hat: str = "auto"
        ) -> RetrieveContextResult:
            state.progress(f"Running retrieve_context for query={query!r}")
            with state.retrieval_usage("retrieve_context") as app_instance:
                result = app_instance.retrieve_context(
                    query,
                    hat=hat
                )
            return RetrieveContextResult(
                query=query,
                selected_hat=result.selected_hat,
                expanded_query=result.expanded_query,
                contexts=_serialize_context_hits(result.answer_hits),
            )

    if mcp_config.tools["ask"].enabled:
        @mcp.tool(
            name="ask",
            description=mcp_config.tools["ask"].description,
        )
        def ask(
            query: str,
            hat: str = "auto",
            answer_mode: Literal["default", "light", "none", "raw"] = "default",
            rerank_top_k: int | None = None,
            answer_context_top_k: int | None = None,
        ) -> AskToolResult:
            with state.retrieval_usage("ask") as app_instance:
                backend = str(getattr(app_instance.config, "mcp_llm_backend", "client")).strip().lower() or "client"
                state.progress(f"Running ask for query={query!r} using {backend} MCP backend")
                if backend == "local":
                    state.release_retrieval_models(
                        "Releasing retrieval models before handing the GPU to the local LLM"
                    )
                    state.progress("Hitting local LLM through ask")
                    result = app_instance.ask(
                        query,
                        hat=hat,
                        answer_mode=answer_mode,
                        rerank_top_k=rerank_top_k,
                        answer_context_top_k=answer_context_top_k,
                    )
                    return AskToolResult(
                        query=query,
                        selected_hat=result.selected_hat,
                        llm_backend="local",
                        route="retrieve",
                        answer_mode=result.answer_mode,
                        answer=result.answer,
                        citations=result.citations,
                    )
                if answer_mode in {"none", "raw"}:
                    state.progress("Answer mode is deterministic; no local answer LLM will be used")
                    result = app_instance.ask(
                        query,
                        hat=hat,
                        answer_mode=answer_mode,
                        rerank_top_k=rerank_top_k,
                        answer_context_top_k=answer_context_top_k,
                    )
                    return AskToolResult(
                        query=query,
                        selected_hat=result.selected_hat,
                        llm_backend="client",
                        route="retrieve",
                        answer_mode=result.answer_mode,
                        answer=result.answer,
                        citations=result.citations,
                    )
                state.progress("Preparing client-side answer package without local answer LLM")
                return _build_client_ask_package(
                    app_instance,
                    query=query,
                    hat=hat,
                    answer_mode=answer_mode,
                    rerank_top_k=rerank_top_k,
                    answer_context_top_k=answer_context_top_k,
                    progress=state.progress,
                )

    if mcp_config.tools["load_content"].enabled:
        @mcp.tool(
            name="load_content",
            description=mcp_config.tools["load_content"].description,
        )
        def load_content(input_ref: str, hat: str = "auto") -> LoadContentResult:
            state.progress(f"Loading content from {input_ref!r}")
            with state.retrieval_usage("load_content") as app_instance:
                result = app_instance.load(input_ref, hat=hat)
            return LoadContentResult(
                load_id=result.load_id,
                hat=result.hat,
                document_count=result.document_count,
                topic_folders=result.topic_folders,
                total_chunks=result.total_chunks,
                total_markdown_segments=result.total_markdown_segments,
                failures=[failure.source_uri for failure in result.failures],
            )

    if mcp_config.tools["list_loads"].enabled:
        @mcp.tool(
            name="list_loads",
            description=mcp_config.tools["list_loads"].description,
        )
        def list_loads() -> ListLoadsResult:
            state.progress("Listing ingestion history")
            items = [_serialize_load_event(event) for event in state.resolve().list_loads()]
            return ListLoadsResult(items=items)

    if mcp_config.tools["delete_loads"].enabled:
        @mcp.tool(
            name="delete_loads",
            description=mcp_config.tools["delete_loads"].description,
        )
        def delete_loads(load_ids: list[str]) -> DeleteLoadsResult:
            state.progress(f"Deleting load IDs: {', '.join(load_ids) if load_ids else '<none>'}")
            with state.retrieval_usage("delete_loads") as app_instance:
                result = app_instance.delete(load_ids)
            return DeleteLoadsResult(
                deleted_load_ids=result.deleted_load_ids,
                missing_load_ids=result.missing_load_ids,
                deleted_topics=result.deleted_topics,
            )

    if mcp_config.tools["delete_hat"].enabled:
        @mcp.tool(
            name="delete_hat",
            description=mcp_config.tools["delete_hat"].description,
        )
        def delete_hat(hat: str) -> DeleteHatToolResult:
            state.progress(f"Deleting hat {hat!r}")
            with state.retrieval_usage("delete_hat") as app_instance:
                result = app_instance.delete_hat(hat)
            return DeleteHatToolResult(
                hat=result.hat,
                existed=result.existed,
                deleted_load_ids=result.deleted_load_ids,
                deleted_topics=result.deleted_topics,
            )

    @mcp.prompt(
        name=mcp_config.prompts["find_from_local_library"].name,
        description=mcp_config.prompts["find_from_local_library"].description,
        title="Find From Local Library",
    )
    def find_from_local_library(user_request: str) -> str:
        state.progress(f"Rendering MCP prompt for local-library retrieval guidance: {user_request!r}")
        return mcp_config.prompts["find_from_local_library"].template.format(user_request=user_request)

    @mcp.resource(
        "arignan://global-map",
        name=mcp_config.resources["global_map"].name,
        description=mcp_config.resources["global_map"].description,
        mime_type="text/markdown",
    )
    def global_map() -> str:
        state.progress("Reading global knowledge map")
        return state.resolve().layout.global_map_path.read_text(encoding="utf-8")

    if app is None and app_factory is not None:
        threading.Thread(
            target=state.background_load_retrieval_models,
            name="background_retrieval_model_load",
            daemon=True,
        ).start()

    return mcp


async def run_mcp_stdio_logged(server: FastMCP, *, progress_sink) -> None:
    stdin = anyio.wrap_file(TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace"))
    stdout = anyio.wrap_file(TextIOWrapper(sys.stdout.buffer, encoding="utf-8"))

    read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
    read_stream_writer: MemoryObjectSendStream[SessionMessage | Exception]
    write_stream: MemoryObjectSendStream[SessionMessage]
    write_stream_reader: MemoryObjectReceiveStream[SessionMessage]

    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    async def stdin_reader() -> None:
        try:
            async with read_stream_writer:
                async for line in stdin:
                    preview = line.strip()
                    if preview:
                        progress_sink(f"Received MCP payload: {preview[:240]}")
                        compact = preview.replace(" ", "")
                        if '"method":"initialize"' in compact:
                            progress_sink("Initialize request received")
                        elif '"method":"ping"' in compact:
                            progress_sink("Ping request received")
                    try:
                        message = types.JSONRPCMessage.model_validate_json(line)
                    except Exception as exc:
                        await read_stream_writer.send(exc)
                        continue
                    await read_stream_writer.send(SessionMessage(message))
        except anyio.ClosedResourceError:  # pragma: no cover
            await anyio.lowlevel.checkpoint()

    async def stdout_writer() -> None:
        try:
            async with write_stream_reader:
                async for session_message in write_stream_reader:
                    json_payload = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
                    await stdout.write(json_payload + "\n")
                    await stdout.flush()
        except anyio.ClosedResourceError:  # pragma: no cover
            await anyio.lowlevel.checkpoint()

    async with anyio.create_task_group() as tg:
        tg.start_soon(stdin_reader)
        tg.start_soon(stdout_writer)
        await server._mcp_server.run(  # pyright: ignore[reportPrivateUsage]
            read_stream,
            write_stream,
            server._mcp_server.create_initialization_options(),  # pyright: ignore[reportPrivateUsage]
        )


def _build_client_ask_package(
    app: ArignanApp,
    *,
    query: str,
    hat: str,
    answer_mode: str,
    rerank_top_k: int | None,
    answer_context_top_k: int | None,
    progress,
) -> AskToolResult:
    pid = app.terminal_pid
    session = app.session_manager.get_or_create(pid, hat=hat)
    selected_hat = app._fallback_selected_hat(hat, session)
    route = "retrieve"
    if answer_mode in {"default", "light"} and session is not None and any(turn.role.lower() == "assistant" for turn in session.turns):
        decision = app._classify_ask_route_with_embeddings(query, selected_hat=selected_hat, session=session)
        route = decision.route
        progress(f"Client-side ask route classified as {route}")
    if route == "chat_context":
        user_prompt = render_prompt_template(
            "conversational_answer_user_template",
            app.prompts.conversational_answer_user_template,
            selected_hat=selected_hat,
            question=query,
        )
        messages = [
            McpChatMessage(role="system", content=app.prompts.conversational_answer_system_prompt),
            *_serialize_chat_messages(_chat_messages_for_session(session, question=query)),
            McpChatMessage(role="user", content=user_prompt),
        ]
        return AskToolResult(
            query=query,
            selected_hat=selected_hat,
            llm_backend="client",
            route="chat_context",
            answer_mode=answer_mode,
            system_prompt=app.prompts.conversational_answer_system_prompt,
            user_prompt=user_prompt,
            messages=messages,
            note="Use the client LLM with recent chat context only; no local retrieval was needed for this turn.",
        )

    retrieval = app.retrieve_context(
        query,
        hat=hat
    )
    answer_hits = retrieval.answer_hits
    if not answer_hits:
        user_prompt = _build_no_context_answer_prompt(
            query,
            selected_hat=retrieval.selected_hat,
            expanded_query=retrieval.expanded_query,
            template=app.prompts.no_context_answer_user_template,
        )
        messages = [
            McpChatMessage(role="system", content=app.prompts.no_context_answer_system_prompt),
            *_serialize_chat_messages(_chat_messages_for_session(session, question=query)),
            McpChatMessage(role="user", content=user_prompt),
        ]
        return AskToolResult(
            query=query,
            selected_hat=retrieval.selected_hat,
            llm_backend="client",
            route="retrieve",
            answer_mode=answer_mode,
            expanded_query=retrieval.expanded_query,
            contexts=[],
            system_prompt=app.prompts.no_context_answer_system_prompt,
            user_prompt=user_prompt,
            messages=messages,
            note="No useful local context was found; the client LLM should answer from recent chat context and its own knowledge.",
        )

    context_limit = app._answer_context_limit(
        answer_mode,
        rerank_top_k=app._effective_rerank_top_k(rerank_top_k),
        answer_context_top_k=answer_context_top_k,
    )
    user_prompt = _build_answer_prompt(
        query,
        answer_hits,
        context_limit=context_limit,
        expanded_query=retrieval.expanded_query,
        selected_hat=retrieval.selected_hat,
        session=session,
        template=app.prompts.answer_user_template,
    )
    system_prompt = app.prompts.answer_system_prompt
    messages = [
        McpChatMessage(role="system", content=system_prompt),
        *_serialize_chat_messages(_chat_messages_for_session(session, question=query)),
        McpChatMessage(role="user", content=user_prompt),
    ]
    return AskToolResult(
        query=query,
        selected_hat=retrieval.selected_hat,
        llm_backend="client",
        route="retrieve",
        answer_mode=answer_mode,
        expanded_query=retrieval.expanded_query,
        contexts=_serialize_context_hits(answer_hits[:context_limit]),
        citations=[format_citation(hit) for hit in answer_hits[: min(len(answer_hits), 5)]],
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        messages=messages,
        note="Use the client LLM with this retrieved local context package; no local answer LLM was invoked.",
    )


def _serialize_context_hits(hits) -> list[RetrievedContext]:
    return [
        RetrievedContext(
            text=hit.text,
            source=hit.source.value,
            citation=format_citation(hit),
        )
        for hit in hits
    ]


def _serialize_chat_messages(messages: list[dict[str, str]]) -> list[McpChatMessage]:
    serialized: list[McpChatMessage] = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role in {"system", "user", "assistant"} and content:
            serialized.append(McpChatMessage(role=role, content=content))
    return serialized


def _serialize_load_event(event: LoadEvent) -> LoadEventResult:
    return LoadEventResult(
        created_at=event.created_at.isoformat(),
        operation=event.operation.value,
        load_id=event.load_id,
        hat=event.hat,
        source_items=list(event.source_items),
    )
