from __future__ import annotations

import shutil
import socket
import tempfile
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from arignan.application import ArignanApp
from arignan.config import load_config

SUPPORTED_GUI_UPLOAD_SUFFIXES = {".pdf", ".md", ".markdown"}


class AskPayload(BaseModel):
    question: str
    hat: str = "auto"
    answer_mode: str = "default"


@dataclass(slots=True)
class GuiTaskState:
    task_id: str
    kind: str
    status: str = "running"
    message: str = "Starting..."
    result: dict[str, object] | None = None
    error: str | None = None
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
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class GuiTaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, GuiTaskState] = {}
        self._lock = threading.Lock()

    def create(self, kind: str, message: str) -> GuiTaskState:
        task = GuiTaskState(task_id=uuid.uuid4().hex, kind=kind, message=message)
        with self._lock:
            self._tasks[task.task_id] = task
        return task

    def update(self, task_id: str, message: str) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.message = message
            task.updated_at = datetime.now(timezone.utc).isoformat()

    def finish(self, task_id: str, result: dict[str, object]) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.status = "done"
            task.result = result
            task.message = "Done"
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
            task_app = _build_task_app(
                app,
                lambda message: task_store.update(task.task_id, _compact_gui_progress("load", message)),
            )
            try:
                result = task_app.load(str(batch_dir), hat=hat)
                task_store.finish(task.task_id, _serialize_load_result(result, uploaded_files=written_files))
            except Exception as exc:
                task_store.fail(task.task_id, str(exc))
            finally:
                shutil.rmtree(batch_dir, ignore_errors=True)

        threading.Thread(target=_runner, daemon=True).start()
        return {"task_id": task.task_id, "uploaded_files": written_files}

    @gui_app.post("/api/ask")
    async def ask(payload: AskPayload) -> dict[str, object]:
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        result = app.ask(question, hat=payload.hat, answer_mode=payload.answer_mode)
        return _serialize_ask_result(result)

    @gui_app.post("/api/ask/start")
    async def start_ask(payload: AskPayload) -> dict[str, object]:
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        task = task_store.create("ask", "Preparing question...")

        def _runner() -> None:
            task_app = _build_task_app(
                app,
                lambda message: task_store.update(task.task_id, _compact_gui_progress("ask", message)),
            )
            try:
                result = task_app.ask(question, hat=payload.hat, answer_mode=payload.answer_mode)
                task_store.finish(task.task_id, _serialize_ask_result(result))
            except Exception as exc:
                task_store.fail(task.task_id, str(exc))

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
    app = ArignanApp(config, terminal_pid=terminal_pid)
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


def _build_task_app(app: ArignanApp, progress_sink) -> ArignanApp:
    task_app = ArignanApp(app.config, progress_sink=progress_sink, terminal_pid=app.terminal_pid)
    for name in ("load", "ask"):
        if name in app.__dict__:
            setattr(task_app, name, getattr(app, name))
    return task_app


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


def _compact_gui_progress(kind: str, message: str) -> str:
    raw = message.strip()
    if not raw:
        return "Working..."
    if kind == "ask":
        mapped = _compact_ask_progress(raw)
        return mapped or raw
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


def _frontend_dir() -> Path:
    return Path(__file__).resolve().parent / "frontend"


def _open_browser_later(url: str) -> None:
    def _worker() -> None:
        time.sleep(0.8)
        try:
            webbrowser.open(url)
        except Exception:
            return

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
