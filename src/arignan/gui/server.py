from __future__ import annotations

"""Legacy GUI server module.

The active exported GUI entrypoint comes from `arignan.gui.react_server`.
This module is kept only as a historical fallback/reference and should not
drift silently.
"""

import shutil
import socket
import tempfile
import threading
import time
import traceback
import webbrowser
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from arignan.application import ArignanApp
from arignan.config import load_config

SUPPORTED_GUI_UPLOAD_SUFFIXES = {".pdf", ".md", ".markdown"}


class AskPayload(BaseModel):
    question: str
    hat: str = "auto"
    answer_mode: str = "default"


def create_gui_app(app: ArignanApp) -> FastAPI:
    gui_app = FastAPI(title="Open Arignan GUI")

    @gui_app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _render_index_html()

    @gui_app.get("/api/options")
    async def options() -> dict[str, object]:
        hats = ["auto", *sorted(path.name for path in app.layout.hats_dir.iterdir() if path.is_dir())]
        return {
            "app_home": str(app.config.app_home),
            "default_hat": app.config.default_hat,
            "hats": list(dict.fromkeys(hats)),
            "answer_modes": ["default", "light", "none", "raw"],
        }

    @gui_app.post("/api/load")
    async def load_files(
        hat: str = Form("auto"),
        files: list[UploadFile] = File(...),
    ) -> dict[str, object]:
        if not files:
            raise HTTPException(status_code=400, detail="No files selected.")
        upload_root = app.config.app_home / "gui_uploads"
        upload_root.mkdir(parents=True, exist_ok=True)
        batch_dir = Path(tempfile.mkdtemp(prefix="batch-", dir=str(upload_root)))
        try:
            written_files: list[str] = []
            unsupported_files: list[str] = []
            for index, upload in enumerate(files, start=1):
                if not upload.filename:
                    continue
                safe_name = Path(upload.filename).name
                suffix = Path(safe_name).suffix.lower()
                if suffix not in SUPPORTED_GUI_UPLOAD_SUFFIXES:
                    unsupported_files.append(safe_name)
                    continue
                target = batch_dir / f"{index:03d}-{safe_name}"
                target.write_bytes(await upload.read())
                written_files.append(safe_name)
            if unsupported_files:
                allowed = ", ".join(sorted(SUPPORTED_GUI_UPLOAD_SUFFIXES))
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported file type for: {', '.join(unsupported_files)}. "
                        f"Use only {allowed}."
                    ),
                )
            if not written_files:
                raise HTTPException(status_code=400, detail="No supported files were selected for loading.")
            result = app.load(str(batch_dir), hat=hat)
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)
        return {
            "load_id": result.load_id,
            "hat": result.hat,
            "document_count": result.document_count,
            "topic_folders": result.topic_folders,
            "total_chunks": result.total_chunks,
            "total_markdown_segments": result.total_markdown_segments,
            "uploaded_files": written_files,
            "failures": [{"source_uri": failure.source_uri, "message": failure.message} for failure in result.failures],
        }

    @gui_app.post("/api/ask")
    async def ask(payload: AskPayload) -> dict[str, object]:
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        result = app.ask(question, hat=payload.hat, answer_mode=payload.answer_mode)
        return {
            "question": result.question,
            "answer": result.answer,
            "citations": result.citations,
            "selected_hat": result.selected_hat,
            "answer_mode": result.answer_mode,
        }

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


def _render_index_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Open Arignan</title>
  <style>
    :root {
      --bg: #f5f1e8;
      --panel: #fffaf0;
      --panel-strong: #ffffff;
      --ink: #17201f;
      --muted: #66706d;
      --line: #d9cfbf;
      --accent: #156f5c;
      --accent-soft: #d7eee7;
      --shadow: 0 20px 50px rgba(44, 38, 24, 0.12);
      --radius: 24px;
      --code: "IBM Plex Mono", "Cascadia Code", monospace;
      --sans: "Segoe UI", "Aptos", "Helvetica Neue", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(21, 111, 92, 0.08), transparent 28%),
        linear-gradient(180deg, #f7f4ee 0%, var(--bg) 100%);
      min-height: 100vh;
    }
    .shell {
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 40px;
      display: grid;
      gap: 20px;
    }
    .hero, .chat-shell {
      background: rgba(255, 250, 240, 0.88);
      border: 1px solid rgba(217, 207, 191, 0.9);
      backdrop-filter: blur(16px);
      box-shadow: var(--shadow);
      border-radius: var(--radius);
    }
    .hero {
      padding: 22px 24px;
      display: grid;
      gap: 16px;
    }
    .hero-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }
    .brand h1 {
      margin: 0;
      font-size: 1.7rem;
      letter-spacing: -0.03em;
    }
    .brand p {
      margin: 6px 0 0;
      color: var(--muted);
    }
    .toolbar {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    .upload-stack {
      display: grid;
      gap: 6px;
      min-width: 280px;
    }
    .toolbar label {
      display: grid;
      gap: 6px;
      font-size: 0.84rem;
      color: var(--muted);
      font-weight: 600;
    }
    select, button, textarea {
      font: inherit;
    }
    select {
      min-width: 160px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
      padding: 10px 12px;
      color: var(--ink);
    }
    .upload-button {
      min-height: 52px;
      border: none;
      border-radius: 18px;
      background: linear-gradient(135deg, var(--accent), #1f8f77);
      color: white;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 16px 28px rgba(21, 111, 92, 0.25);
      padding: 0 18px;
      display: inline-flex;
      align-items: center;
      gap: 10px;
    }
    .upload-button .plus {
      font-size: 1.55rem;
      line-height: 1;
    }
    .upload-hint {
      color: var(--muted);
      font-size: 0.85rem;
    }
    .upload-button:hover, .send-button:hover {
      filter: brightness(1.03);
      transform: translateY(-1px);
    }
    .status-bar {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 0.92rem;
    }
    .pill {
      border-radius: 999px;
      padding: 6px 12px;
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 700;
    }
    .chat-shell {
      display: grid;
      grid-template-rows: 1fr auto;
      min-height: 68vh;
      overflow: hidden;
    }
    .messages {
      padding: 24px;
      display: grid;
      gap: 16px;
      overflow-y: auto;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.45), rgba(255,250,240,0.55)),
        repeating-linear-gradient(
          180deg,
          transparent,
          transparent 32px,
          rgba(21, 111, 92, 0.025) 32px,
          rgba(21, 111, 92, 0.025) 33px
        );
    }
    .message {
      max-width: min(84%, 820px);
      padding: 16px 18px;
      border-radius: 22px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
      box-shadow: 0 8px 20px rgba(38, 34, 24, 0.07);
    }
    .message.user {
      margin-left: auto;
      border-color: rgba(21, 111, 92, 0.24);
      background: linear-gradient(180deg, #f0faf6, #e6f5ef);
    }
    .message.system {
      max-width: 100%;
      background: transparent;
      border-style: dashed;
      box-shadow: none;
      color: var(--muted);
    }
    .message-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
      font-size: 0.8rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .message-body {
      white-space: pre-wrap;
      line-height: 1.55;
    }
    .citations {
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px dashed var(--line);
      display: grid;
      gap: 6px;
      font-family: var(--code);
      font-size: 0.86rem;
      color: #39534d;
    }
    .composer {
      padding: 18px;
      border-top: 1px solid rgba(217, 207, 191, 0.9);
      display: grid;
      gap: 12px;
      background: rgba(255, 250, 240, 0.95);
    }
    .composer-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: end;
    }
    textarea {
      min-height: 72px;
      max-height: 220px;
      resize: vertical;
      width: 100%;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
      padding: 14px 16px;
      color: var(--ink);
      line-height: 1.5;
    }
    .send-button {
      border: none;
      border-radius: 18px;
      background: linear-gradient(135deg, var(--accent), #1f8f77);
      color: white;
      padding: 0 18px;
      height: 52px;
      font-weight: 700;
      cursor: pointer;
      min-width: 112px;
      box-shadow: 0 14px 26px rgba(21, 111, 92, 0.22);
    }
    .muted {
      color: var(--muted);
      font-size: 0.9rem;
    }
    @media (max-width: 900px) {
      .shell { padding: 16px; }
      .toolbar { width: 100%; }
      .toolbar label { flex: 1 1 160px; }
      .composer-row { grid-template-columns: 1fr; }
      .send-button { width: 100%; }
      .message { max-width: 100%; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="hero-top">
        <div class="brand">
          <h1>Open Arignan</h1>
          <p>Local knowledge base loading and questioning without typing command syntax.</p>
        </div>
        <div class="toolbar">
          <div class="upload-stack">
            <button class="upload-button" id="uploadButton" title="Add more files to knowledge base">
              <span class="plus">+</span>
              <span>Add More Files To Knowledge Base</span>
            </button>
            <div class="upload-hint">Accepted: PDF and Markdown. Files are added to the selected hat below.</div>
          </div>
          <input id="filePicker" type="file" hidden multiple accept=".pdf,.md,.markdown" />
          <label>
            Hat
            <select id="hatSelect"></select>
          </label>
          <label>
            Answer Mode
            <select id="answerModeSelect"></select>
          </label>
        </div>
      </div>
      <div class="status-bar">
        <span class="pill" id="statusPill">Ready</span>
        <span id="statusText">Use “Add More Files To Knowledge Base” to extend your stored knowledge, then ask questions in the chat box below.</span>
      </div>
    </section>

    <section class="chat-shell">
      <div id="messages" class="messages"></div>
      <div class="composer">
        <div class="muted">Ask naturally. Hat and answer mode are taken from the dropdowns above.</div>
        <div class="composer-row">
          <textarea id="questionInput" placeholder="Ask a question about your local knowledge base..."></textarea>
          <button class="send-button" id="sendButton">Ask</button>
        </div>
      </div>
    </section>
  </div>

  <script>
    const messages = document.getElementById("messages");
    const hatSelect = document.getElementById("hatSelect");
    const answerModeSelect = document.getElementById("answerModeSelect");
    const uploadButton = document.getElementById("uploadButton");
    const filePicker = document.getElementById("filePicker");
    const questionInput = document.getElementById("questionInput");
    const sendButton = document.getElementById("sendButton");
    const statusPill = document.getElementById("statusPill");
    const statusText = document.getElementById("statusText");

    function addMessage(role, title, body, citations = []) {
      const item = document.createElement("article");
      item.className = `message ${role}`;
      const head = document.createElement("div");
      head.className = "message-head";
      head.innerHTML = `<span>${title}</span><span>${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>`;
      const content = document.createElement("div");
      content.className = "message-body";
      content.textContent = body;
      item.appendChild(head);
      item.appendChild(content);
      if (citations.length) {
        const citationBox = document.createElement("div");
        citationBox.className = "citations";
        citations.forEach((citation) => {
          const line = document.createElement("div");
          line.textContent = citation;
          citationBox.appendChild(line);
        });
        item.appendChild(citationBox);
      }
      messages.appendChild(item);
      messages.scrollTop = messages.scrollHeight;
    }

    function setStatus(label, text) {
      statusPill.textContent = label;
      statusText.textContent = text;
    }

    async function bootstrap() {
      const response = await fetch("/api/options");
      const payload = await response.json();
      payload.hats.forEach((hat) => {
        const option = document.createElement("option");
        option.value = hat;
        option.textContent = hat;
        hatSelect.appendChild(option);
      });
      hatSelect.value = payload.default_hat || "default";
      payload.answer_modes.forEach((mode) => {
        const option = document.createElement("option");
        option.value = mode;
        option.textContent = mode;
        answerModeSelect.appendChild(option);
      });
      answerModeSelect.value = "default";
      addMessage(
        "system",
        "Ready",
        "Use “Add More Files To Knowledge Base” to extend the knowledge base, then type questions here. The app stays fully local and uses your configured hat and answer mode."
      );
    }

    uploadButton.addEventListener("click", () => filePicker.click());

    filePicker.addEventListener("change", async () => {
      if (!filePicker.files.length) return;
      const formData = new FormData();
      formData.append("hat", hatSelect.value);
      Array.from(filePicker.files).forEach((file) => formData.append("files", file));
      const selectedNames = Array.from(filePicker.files).map((file) => file.name).join(", ");
      setStatus("Loading", `Adding ${filePicker.files.length} file(s) to hat '${hatSelect.value}'...`);
      addMessage("system", "Knowledge base update", `Adding ${filePicker.files.length} file(s) to hat '${hatSelect.value}': ${selectedNames}`);
      try {
        const response = await fetch("/api/load", { method: "POST", body: formData });
        const payload = await parseApiResponse(response);
        if (!response.ok) throw new Error(payload.detail || "Load failed.");
        const topics = payload.topic_folders.length ? payload.topic_folders.join(", ") : "none";
        const failures = payload.failures.length ? ` Failed files: ${payload.failures.length}.` : "";
        const uploaded = (payload.uploaded_files || []).join(", ");
        const summary = `Loaded ${payload.document_count} document(s) into hat '${payload.hat}' as topics: ${topics}. Chunks: ${payload.total_chunks}. Markdown segments: ${payload.total_markdown_segments}.${failures}${uploaded ? ` Uploaded: ${uploaded}.` : ""}`;
        setStatus("Loaded", summary);
        addMessage("system", "Load complete", summary);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setStatus("Error", message);
        addMessage("system", "Load failed", message);
      } finally {
        filePicker.value = "";
      }
    });

    async function askQuestion() {
      const question = questionInput.value.trim();
      if (!question) return;
      const hat = hatSelect.value;
      const answerMode = answerModeSelect.value;
      addMessage("user", "You", question);
      questionInput.value = "";
      setStatus("Thinking", `Asking with hat '${hat}' and answer mode '${answerMode}'...`);
      try {
        const response = await fetch("/api/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, hat, answer_mode: answerMode }),
        });
        const payload = await parseApiResponse(response);
        if (!response.ok) throw new Error(payload.detail || "Ask failed.");
        const selectedHat = payload.selected_hat || hat;
        setStatus("Answered", `Answer ready from hat '${selectedHat}' in mode '${payload.answer_mode}'.`);
        addMessage("assistant", "Arignan", payload.answer, payload.citations || []);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setStatus("Error", message);
        addMessage("system", "Ask failed", message);
      }
    }

    async function parseApiResponse(response) {
      const text = await response.text();
      if (!text) return {};
      try {
        return JSON.parse(text);
      } catch (error) {
        return { detail: text };
      }
    }

    sendButton.addEventListener("click", askQuestion);
    questionInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        askQuestion();
      }
    });

    bootstrap();
  </script>
</body>
</html>
"""
