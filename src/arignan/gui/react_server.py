from __future__ import annotations

import shutil
import socket
import tempfile
import threading
import time
import traceback
import uuid
import webbrowser
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from arignan.application import ArignanApp
from arignan.config import load_config
from arignan.models import LoadOperation

SUPPORTED_GUI_UPLOAD_SUFFIXES = {".pdf", ".md", ".markdown"}


class AskPayload(BaseModel):
    question: str
    hat: str = "auto"
    answer_mode: str = "default"
    rerank_top_k: int | None = None
    show_thinking: bool = True


class DeletePayload(BaseModel):
    load_ids: list[str] | None = None
    hat: str | None = None


@dataclass(slots=True)
class GuiTaskState:
    task_id: str
    kind: str
    status: str = "running"
    message: str = "Starting..."
    result: dict[str, object] | None = None
    error: str | None = None
    progress_log: list[str] = field(default_factory=list)
    partial_answer: str = ""
    partial_thinking: str = ""
    thought_started_at: str | None = None
    thought_finished_at: str | None = None
    thought_usage: dict[str, object] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "status": self.status,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "progress_log": list(self.progress_log),
            "partial_answer": self.partial_answer,
            "partial_thinking": self.partial_thinking,
            "thought_started_at": self.thought_started_at,
            "thought_finished_at": self.thought_finished_at,
            "thought_usage": self.thought_usage,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class GuiTaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, GuiTaskState] = {}
        self._lock = threading.Lock()

    def create(self, kind: str, message: str) -> GuiTaskState:
        task = GuiTaskState(task_id=uuid.uuid4().hex, kind=kind, message=message)
        task.progress_log.append(message)
        with self._lock:
            self._tasks[task.task_id] = task
        return task

    def update(self, task_id: str, message: str) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.message = message
            if not task.progress_log or task.progress_log[-1] != message:
                task.progress_log.append(message)
                if len(task.progress_log) > 12:
                    task.progress_log = task.progress_log[-12:]
            task.updated_at = datetime.now(timezone.utc).isoformat()

    def append_partial_answer(self, task_id: str, text: str) -> None:
        if not text:
            return
        with self._lock:
            task = self._tasks[task_id]
            if task.partial_thinking and task.thought_finished_at is None:
                task.thought_finished_at = datetime.now(timezone.utc).isoformat()
            task.partial_answer += text
            if task.message != "Streaming answer":
                task.message = "Streaming answer"
                if not task.progress_log or task.progress_log[-1] != "Streaming answer":
                    task.progress_log.append("Streaming answer")
            task.updated_at = datetime.now(timezone.utc).isoformat()

    def append_partial_thinking(self, task_id: str, text: str) -> None:
        if not text:
            return
        with self._lock:
            task = self._tasks[task_id]
            if task.thought_started_at is None:
                task.thought_started_at = datetime.now(timezone.utc).isoformat()
            task.partial_thinking += text
            if task.message == "Preparing question...":
                task.message = "Thinking"
                if not task.progress_log or task.progress_log[-1] != "Thinking":
                    task.progress_log.append("Thinking")
            task.updated_at = datetime.now(timezone.utc).isoformat()

    def finish(self, task_id: str, result: dict[str, object], *, thought_usage: dict[str, object] | None = None) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.status = "done"
            task.result = result
            task.message = "Done"
            if task.partial_thinking and task.thought_finished_at is None:
                task.thought_finished_at = datetime.now(timezone.utc).isoformat()
            task.thought_usage = thought_usage
            task.updated_at = datetime.now(timezone.utc).isoformat()

    def fail(self, task_id: str, error: str) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.status = "error"
            task.error = error
            task.message = error
            task.updated_at = datetime.now(timezone.utc).isoformat()

    def get(self, task_id: str) -> GuiTaskState | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            return GuiTaskState(**task.to_dict())


def create_gui_app(app: ArignanApp) -> FastAPI:
    gui_app = FastAPI(title="Open Arignan GUI")
    task_store = GuiTaskStore()
    task_lock = threading.Lock()
    frontend_dir = _frontend_dir()
    gui_app.mount("/gui-static", StaticFiles(directory=str(frontend_dir)), name="gui-static")

    @gui_app.get("/", response_class=HTMLResponse)
    async def index() -> FileResponse:
        return FileResponse(frontend_dir / "index.html")

    @gui_app.get("/api/options")
    async def options() -> dict[str, object]:
        hats = ["auto", *sorted(path.name for path in app.layout.hats_dir.iterdir() if path.is_dir())]
        return {
            "app_home": str(app.config.app_home),
            "default_hat": app.config.default_hat,
            "hats": list(dict.fromkeys(hats)),
            "answer_modes": ["default", "light", "none", "raw"],
            "default_rerank_top_k": app.config.retrieval.rerank_top_k,
            "default_show_thinking": True,
        }

    @gui_app.get("/api/library")
    async def library() -> dict[str, object]:
        hats = sorted(path.name for path in app.layout.hats_dir.iterdir() if path.is_dir())
        loads = [
            _serialize_load_event(event)
            for event in sorted(
                app.list_live_ingestions(),
                key=lambda item: item.created_at,
                reverse=True,
            )
        ]
        return {
            "hats": hats,
            "loads": loads,
        }

    @gui_app.get("/api/tasks/{task_id}")
    async def task_status(task_id: str) -> dict[str, object]:
        task = task_store.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found.")
        return task.to_dict()

    @gui_app.post("/api/load")
    async def load_files(
        hat: str = Form("auto"),
        files: list[UploadFile] = File(...),
    ) -> dict[str, object]:
        return await _handle_direct_load(app, hat=hat, files=files)

    @gui_app.post("/api/load/start")
    async def start_load(
        hat: str = Form("auto"),
        files: list[UploadFile] = File(...),
    ) -> dict[str, object]:
        upload_root = app.config.app_home / "gui_uploads"
        upload_root.mkdir(parents=True, exist_ok=True)
        batch_dir = Path(tempfile.mkdtemp(prefix="batch-", dir=str(upload_root)))
        try:
            written_files = await _write_uploaded_files(batch_dir, files)
        except Exception:
            shutil.rmtree(batch_dir, ignore_errors=True)
            raise

        task = task_store.create("load", "Scanning input for load...")

        def _runner() -> None:
            progress_sink = lambda message: task_store.update(task.task_id, _compact_gui_progress("load", message))
            try:
                with _gui_task_context(app, task_lock, progress_sink):
                    result = app.load(str(batch_dir), hat=hat)
                    task_store.finish(task.task_id, _serialize_load_result(result, uploaded_files=written_files))
            except Exception as exc:
                task_store.fail(
                    task.task_id,
                    _task_error_message(
                        app,
                        component="gui",
                        task="load task",
                        exc=exc,
                        context={"hat": hat, "uploaded_files": written_files},
                        user_message="Something went wrong while loading files.",
                    ),
                )
            finally:
                shutil.rmtree(batch_dir, ignore_errors=True)

        threading.Thread(target=_runner, daemon=True).start()
        return {"task_id": task.task_id, "uploaded_files": written_files}

    @gui_app.post("/api/ask")
    async def ask(payload: AskPayload) -> dict[str, object]:
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        result = app.ask(
            question,
            hat=payload.hat,
            answer_mode=payload.answer_mode,
            rerank_top_k=payload.rerank_top_k,
        )
        return _serialize_ask_result(result)

    @gui_app.post("/api/ask/start")
    async def start_ask(payload: AskPayload) -> dict[str, object]:
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        task = task_store.create("ask", "Preparing question...")

        def _runner() -> None:
            progress_sink = lambda message: task_store.update(task.task_id, _compact_gui_progress("ask", message))
            stream_sink = lambda text: task_store.append_partial_answer(task.task_id, text)
            thinking_sink = (lambda text: task_store.append_partial_thinking(task.task_id, text)) if payload.show_thinking else None
            try:
                with _gui_task_context(app, task_lock, progress_sink, stream_sink=stream_sink, thinking_sink=thinking_sink):
                    result = app.ask(
                        question,
                        hat=payload.hat,
                        answer_mode=payload.answer_mode,
                        rerank_top_k=payload.rerank_top_k,
                    )
                    task_store.finish(
                        task.task_id,
                        _serialize_ask_result(result),
                        thought_usage=_serialize_thought_usage(app.local_text_generator),
                    )
            except Exception as exc:
                task_store.fail(
                    task.task_id,
                    _task_error_message(
                        app,
                        component="gui",
                        task="ask task",
                        exc=exc,
                        context={
                            "hat": payload.hat,
                            "answer_mode": payload.answer_mode,
                            "rerank_top_k": payload.rerank_top_k,
                            "show_thinking": payload.show_thinking,
                        },
                        user_message="Something went wrong while answering the question.",
                    ),
                )

        threading.Thread(target=_runner, daemon=True).start()
        return {"task_id": task.task_id}

    @gui_app.post("/api/delete/start")
    async def start_delete(payload: DeletePayload) -> dict[str, object]:
        if payload.hat and payload.load_ids:
            raise HTTPException(status_code=400, detail="Choose either one or more load_ids or a hat, not both.")
        if payload.hat:
            hat = payload.hat.strip()
            if not hat:
                raise HTTPException(status_code=400, detail="Hat cannot be empty.")
            task = task_store.create("delete", f"Deleting hat '{hat}'...")

            def _runner() -> None:
                progress_sink = lambda message: task_store.update(task.task_id, _compact_gui_progress("delete", message))
                try:
                    with _gui_task_context(app, task_lock, progress_sink):
                        result = app.delete_hat(hat)
                        task_store.finish(task.task_id, _serialize_delete_hat_result(result))
                except Exception as exc:
                    task_store.fail(
                        task.task_id,
                        _task_error_message(
                            app,
                            component="gui",
                            task="delete hat task",
                            exc=exc,
                            context={"hat": hat},
                            user_message=f"Something went wrong while deleting hat '{hat}'.",
                        ),
                    )

            threading.Thread(target=_runner, daemon=True).start()
            return {"task_id": task.task_id}

        load_ids = [load_id.strip() for load_id in (payload.load_ids or []) if load_id and load_id.strip()]
        if not load_ids:
            raise HTTPException(status_code=400, detail="No load_ids were provided.")
        task = task_store.create("delete", "Deleting selected loads...")

        def _runner() -> None:
            progress_sink = lambda message: task_store.update(task.task_id, _compact_gui_progress("delete", message))
            try:
                with _gui_task_context(app, task_lock, progress_sink):
                    result = app.delete(load_ids)
                    task_store.finish(task.task_id, _serialize_delete_result(result))
            except Exception as exc:
                task_store.fail(
                    task.task_id,
                    _task_error_message(
                        app,
                        component="gui",
                        task="delete loads task",
                        exc=exc,
                        context={"load_ids": load_ids},
                        user_message="Something went wrong while deleting the selected loads.",
                    ),
                )

        threading.Thread(target=_runner, daemon=True).start()
        return {"task_id": task.task_id}

    return gui_app


def run_gui(
    *,
    app_home: Path | None = None,
    settings_path: Path | None = None,
    terminal_pid: int | None = None,
    open_browser: bool = True,
    host: str = "127.0.0.1",
    port: int | None = None,
) -> int:
    import uvicorn

    resolved_port = port or _find_free_port()
    config = load_config(settings_path=settings_path, app_home=app_home)
    app = ArignanApp(config, progress_sink=_terminal_progress_sink(), terminal_pid=terminal_pid)
    gui_app = create_gui_app(app)
    url = f"http://{host}:{resolved_port}"
    print(f"[arignan] Opening GUI at {url}", flush=True)
    if open_browser:
        _open_browser_later(url)
    uvicorn.run(gui_app, host=host, port=resolved_port, log_level="warning")
    return 0


async def _handle_direct_load(app: ArignanApp, *, hat: str, files: list[UploadFile]) -> dict[str, object]:
    if not files:
        raise HTTPException(status_code=400, detail="No files selected.")
    upload_root = app.config.app_home / "gui_uploads"
    upload_root.mkdir(parents=True, exist_ok=True)
    batch_dir = Path(tempfile.mkdtemp(prefix="batch-", dir=str(upload_root)))
    try:
        written_files = await _write_uploaded_files(batch_dir, files)
        result = app.load(str(batch_dir), hat=hat)
        return _serialize_load_result(result, uploaded_files=written_files)
    finally:
        shutil.rmtree(batch_dir, ignore_errors=True)


async def _write_uploaded_files(batch_dir: Path, files: list[UploadFile]) -> list[str]:
    if not files:
        raise HTTPException(status_code=400, detail="No files selected.")
    written_files: list[str] = []
    unsupported_files: list[str] = []
    for index, upload in enumerate(files, start=1):
        if not upload.filename:
            continue
        original_name = upload.filename.replace("\\", "/").strip("/")
        safe_name = Path(original_name).name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_GUI_UPLOAD_SUFFIXES:
            unsupported_files.append(safe_name)
            continue
        target = batch_dir / f"{index:03d}-{safe_name}"
        target.write_bytes(await upload.read())
        written_files.append(original_name)
    if unsupported_files:
        allowed = ", ".join(sorted(SUPPORTED_GUI_UPLOAD_SUFFIXES))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type for: {', '.join(unsupported_files)}. Use only {allowed}.",
        )
    if not written_files:
        raise HTTPException(status_code=400, detail="No supported files were selected for loading.")
    return written_files


@contextmanager
def _gui_task_context(app: ArignanApp, task_lock: threading.Lock, progress_sink, stream_sink=None, thinking_sink=None):
    with task_lock:
        restore = _bind_task_sinks(
            app,
            progress_sink=_tee_progress(app.progress_sink, progress_sink),
            stream_sink=stream_sink,
            thinking_sink=thinking_sink,
        )
        try:
            yield app
        finally:
            restore()


def _terminal_progress_sink():
    def _emit(message: str) -> None:
        print(f"[arignan] {message}", flush=True)

    return _emit


def _tee_progress(*sinks):
    def _emit(message: str) -> None:
        for sink in sinks:
            if sink is not None:
                sink(message)

    return _emit


def _bind_task_sinks(app: ArignanApp, *, progress_sink, stream_sink=None, thinking_sink=None):
    original_app_sink = app.progress_sink
    original_local_sink = getattr(app.local_text_generator, "progress_sink", None)
    original_light_sink = getattr(app.light_text_generator, "progress_sink", None)
    original_local_stream = getattr(app.local_text_generator, "stream_sink", None)
    original_light_stream = getattr(app.light_text_generator, "stream_sink", None)
    original_local_thinking = getattr(app.local_text_generator, "thinking_sink", None)
    original_light_thinking = getattr(app.light_text_generator, "thinking_sink", None)
    artifact_writer = getattr(app.markdown_repository, "artifact_writer", None)
    original_writer_sink = getattr(artifact_writer, "progress_sink", None) if artifact_writer is not None else None

    app.progress_sink = progress_sink
    if hasattr(app.local_text_generator, "progress_sink"):
        app.local_text_generator.progress_sink = progress_sink
    if hasattr(app.light_text_generator, "progress_sink"):
        app.light_text_generator.progress_sink = progress_sink
    if hasattr(app.local_text_generator, "stream_sink"):
        app.local_text_generator.stream_sink = stream_sink
    if hasattr(app.light_text_generator, "stream_sink"):
        app.light_text_generator.stream_sink = stream_sink
    if hasattr(app.local_text_generator, "thinking_sink"):
        app.local_text_generator.thinking_sink = thinking_sink
    if hasattr(app.light_text_generator, "thinking_sink"):
        app.light_text_generator.thinking_sink = thinking_sink
    if artifact_writer is not None and hasattr(artifact_writer, "progress_sink"):
        artifact_writer.progress_sink = progress_sink

    def _restore() -> None:
        app.progress_sink = original_app_sink
        if hasattr(app.local_text_generator, "progress_sink"):
            app.local_text_generator.progress_sink = original_local_sink
        if hasattr(app.light_text_generator, "progress_sink"):
            app.light_text_generator.progress_sink = original_light_sink
        if hasattr(app.local_text_generator, "stream_sink"):
            app.local_text_generator.stream_sink = original_local_stream
        if hasattr(app.light_text_generator, "stream_sink"):
            app.light_text_generator.stream_sink = original_light_stream
        if hasattr(app.local_text_generator, "thinking_sink"):
            app.local_text_generator.thinking_sink = original_local_thinking
        if hasattr(app.light_text_generator, "thinking_sink"):
            app.light_text_generator.thinking_sink = original_light_thinking
        if artifact_writer is not None and hasattr(artifact_writer, "progress_sink"):
            artifact_writer.progress_sink = original_writer_sink

    return _restore


def _serialize_load_result(result, *, uploaded_files: list[str]) -> dict[str, object]:
    return {
        "load_id": result.load_id,
        "hat": result.hat,
        "document_count": result.document_count,
        "topic_folders": result.topic_folders,
        "total_chunks": result.total_chunks,
        "total_markdown_segments": result.total_markdown_segments,
        "uploaded_files": uploaded_files,
        "failures": [{"source_uri": failure.source_uri, "message": failure.message} for failure in result.failures],
    }


def _serialize_ask_result(result) -> dict[str, object]:
    return {
        "question": result.question,
        "answer": result.answer,
        "citations": result.citations,
        "selected_hat": result.selected_hat,
        "answer_mode": result.answer_mode,
    }


def _serialize_thought_usage(generator) -> dict[str, object] | None:
    usage = getattr(generator, "last_usage", None)
    if not isinstance(usage, dict):
        return None
    return dict(usage)


def _serialize_delete_result(result) -> dict[str, object]:
    return {
        "kind": "load_delete",
        "deleted_load_ids": result.deleted_load_ids,
        "missing_load_ids": result.missing_load_ids,
        "deleted_topics": result.deleted_topics,
        "message": (
            f"Deleted loads: {', '.join(result.deleted_load_ids) or 'none'}. "
            f"Missing: {', '.join(result.missing_load_ids) or 'none'}."
        ),
    }


def _serialize_delete_hat_result(result) -> dict[str, object]:
    return {
        "kind": "hat_delete",
        "hat": result.hat,
        "existed": result.existed,
        "deleted_load_ids": result.deleted_load_ids,
        "deleted_topics": result.deleted_topics,
        "message": (
            f"Deleted hat '{result.hat}'. Loads removed: {len(result.deleted_load_ids)}. "
            f"Topics removed: {len(result.deleted_topics)}."
            if result.existed
            else f"Hat '{result.hat}' was not found."
        ),
    }


def _serialize_load_event(event) -> dict[str, object]:
    return {
        "load_id": event.load_id,
        "hat": event.hat,
        "created_at": event.created_at,
        "source_items": list(event.source_items),
        "topic_folders": list(event.topic_folders),
    }


def _compact_gui_progress(kind: str, message: str) -> str:
    raw = message.strip()
    if not raw:
        return "Working..."
    if kind == "ask":
        mapped = _compact_ask_progress(raw)
        return mapped or raw
    if kind == "delete":
        return _compact_delete_progress(raw)
    return _compact_load_progress(raw)


def _compact_ask_progress(message: str) -> str | None:
    if message.startswith("Hat chosen:"):
        return message
    if any(
        token in message
        for token in (
            "Running retrieval pipeline",
            "Expanding query",
            "Selecting hat",
            "Searching dense index",
            "Searching lexical index",
            "Searching map context",
            "Fusing retrieval candidates",
        )
    ):
        return "Retrieval in progress"
    if "Reranking" in message or "rerank" in message.lower():
        return "Reranking"
    if "Hitting local LLM" in message:
        return "Hitting LLM"
    if "Composing raw retrieval output" in message or "Composing retrieval synthesis answer" in message:
        return "Composing answer"
    if message.startswith("Local LLM unavailable"):
        return message
    return None


def _compact_load_progress(message: str) -> str:
    if "Calling local LLM" in message:
        return "Hitting LLM"
    return message


def _compact_delete_progress(message: str) -> str:
    if "Deleting hat" in message or "Deleting " in message:
        return "Deleting knowledge..."
    if "Removing indexed chunks" in message:
        return "Removing indexed chunks..."
    if "Regenerating topic" in message:
        return "Rewriting topic summaries..."
    if "Refreshing map.md" in message or "Refreshing global_map.md" in message:
        return "Refreshing maps..."
    if "Recording deletion log" in message:
        return "Recording deletion log..."
    return message


def _task_error_message(
    app: ArignanApp,
    *,
    component: str,
    task: str,
    exc: BaseException,
    context: dict[str, object] | None,
    user_message: str,
) -> str:
    return app.format_logged_exception_message(
        component=component,
        task=task,
        exc=exc,
        context=context,
        user_message=user_message,
    )


def _frontend_dir() -> Path:
    return Path(__file__).resolve().parent / "frontend"


def _open_browser_later(url: str) -> None:
    def _worker() -> None:
        time.sleep(0.8)
        try:
            webbrowser.open(url)
        except Exception as exc:
            print(f"[arignan] Failed to open the browser automatically for {url}:", flush=True)
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            return

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
