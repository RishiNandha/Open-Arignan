const { useEffect, useMemo, useRef, useState } = React;

const ANSWER_MODES = ["default", "light", "none", "raw"];

function App() {
  const [options, setOptions] = useState({
    hats: ["auto"],
    default_hat: "default",
    answer_modes: ANSWER_MODES,
  });
  const [hat, setHat] = useState("auto");
  const [answerMode, setAnswerMode] = useState("default");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([
    createMessage({
      role: "assistant",
      title: "Arignan",
      body: "Add more files to the knowledge base, then ask questions here. Long operations will show a compact live status in the same assistant bubble.",
    }),
  ]);
  const [loadDialogOpen, setLoadDialogOpen] = useState(false);
  const [loadHat, setLoadHat] = useState("default");
  const [loadMode, setLoadMode] = useState("files");
  const [selectedUploads, setSelectedUploads] = useState([]);
  const [isAsking, setIsAsking] = useState(false);
  const [isLoadingTask, setIsLoadingTask] = useState(false);
  const messagesRef = useRef(null);
  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);

  useEffect(() => {
    bootstrap();
  }, []);

  useEffect(() => {
    if (!messagesRef.current) return;
    messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
  }, [messages]);

  const sortedHats = useMemo(() => options.hats || ["auto"], [options]);

  async function bootstrap() {
    const payload = await fetchJson("/api/options");
    setOptions(payload);
    setHat(payload.hats?.includes("auto") ? "auto" : payload.default_hat || "default");
    setLoadHat(payload.default_hat || "default");
    setAnswerMode("default");
  }

  function openLoadDialog() {
    setLoadDialogOpen(true);
    setLoadHat(hat === "auto" ? options.default_hat || "default" : hat);
    setLoadMode("files");
    setSelectedUploads([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (folderInputRef.current) folderInputRef.current.value = "";
  }

  function closeLoadDialog() {
    setLoadDialogOpen(false);
    setSelectedUploads([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (folderInputRef.current) folderInputRef.current.value = "";
  }

  function appendMessage(message) {
    setMessages((current) => [...current, message]);
  }

  function patchMessage(id, patch) {
    setMessages((current) =>
      current.map((message) => (message.id === id ? { ...message, ...patch } : message))
    );
  }

  function handleUploadSelection(fileList) {
    const next = Array.from(fileList || []);
    setSelectedUploads(next);
  }

  async function confirmLoad() {
    if (!selectedUploads.length || isLoadingTask) return;
    setIsLoadingTask(true);
    const label = selectedUploads.map((file) => file.webkitRelativePath || file.name).join(", ");
    appendMessage(
      createMessage({
        role: "user",
        title: "You",
        body: `Add more files to knowledge base in hat '${loadHat}': ${label}`,
      })
    );
    const pendingMessage = createMessage({
      role: "assistant",
      title: "Arignan",
      body: "Scanning input for load...",
      pending: true,
    });
    const pendingId = pendingMessage.id;
    appendMessage(pendingMessage);
    closeLoadDialog();

    try {
      const formData = new FormData();
      formData.append("hat", loadHat);
      selectedUploads.forEach((file) => {
        const uploadName = file.webkitRelativePath || file.name;
        formData.append("files", file, uploadName);
      });
      const payload = await fetchJson("/api/load/start", {
        method: "POST",
        body: formData,
      });
      await followTask({
        taskId: payload.task_id,
        pendingId,
        onComplete: (result) => {
          patchMessage(pendingId, {
            pending: false,
            body: formatLoadResult(result),
            citations: [],
          });
        },
      });
    } catch (error) {
      patchMessage(pendingId, {
        pending: false,
        body: normalizeError(error),
        citations: [],
      });
    } finally {
      setIsLoadingTask(false);
    }
  }

  async function askQuestion() {
    const trimmed = question.trim();
    if (!trimmed || isAsking) return;
    setIsAsking(true);
    appendMessage(
      createMessage({
        role: "user",
        title: "You",
        body: trimmed,
      })
    );
    const pendingMessage = createMessage({
      role: "assistant",
      title: "Arignan",
      body: "Preparing question...",
      pending: true,
    });
    const pendingId = pendingMessage.id;
    appendMessage(pendingMessage);
    setQuestion("");

    try {
      const payload = await fetchJson("/api/ask/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed, hat, answer_mode: answerMode }),
      });
      await followTask({
        taskId: payload.task_id,
        pendingId,
        onComplete: (result) => {
          patchMessage(pendingId, {
            pending: false,
            body: result.answer,
            citations: result.citations || [],
          });
        },
      });
    } catch (error) {
      patchMessage(pendingId, {
        pending: false,
        body: normalizeError(error),
        citations: [],
      });
    } finally {
      setIsAsking(false);
    }
  }

  async function followTask({ taskId, pendingId, onComplete }) {
    let keepPolling = true;
    while (keepPolling) {
      const snapshot = await fetchJson(`/api/tasks/${taskId}`);
      if (snapshot.message) {
        patchMessage(pendingId, { body: snapshot.message, pending: snapshot.status === "running" });
      }
      if (snapshot.status === "done") {
        onComplete(snapshot.result || {});
        keepPolling = false;
        return;
      }
      if (snapshot.status === "error") {
        patchMessage(pendingId, {
          pending: false,
          body: snapshot.error || snapshot.message || "The task failed.",
          citations: [],
        });
        keepPolling = false;
        return;
      }
      await delay(450);
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1 className="brand-title">Open Arignan</h1>
          <p className="brand-subtitle">Local knowledge base loading and questioning with a chat-style UI.</p>
        </div>
        <button type="button" className="add-button" onClick={openLoadDialog}>
          <span className="plus">+</span>
          <span>Add More Files To Knowledge Base</span>
        </button>
      </header>

      <section className="chat-panel">
        <div className="messages" ref={messagesRef}>
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
        </div>
      </section>

      <section className="composer-panel">
        <div className="composer-toolbar">
          <label className="control-label">
            Hat
            <select className="select-control" value={hat} onChange={(event) => setHat(event.target.value)}>
              {sortedHats.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
          <label className="control-label">
            Answer Mode
            <select
              className="select-control"
              value={answerMode}
              onChange={(event) => setAnswerMode(event.target.value)}
            >
              {(options.answer_modes || ANSWER_MODES).map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="question-row">
          <textarea
            className="question-input"
            placeholder="Ask a question about your local knowledge base..."
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                askQuestion();
              }
            }}
          />
          <button type="button" className="send-button" onClick={askQuestion} disabled={isAsking}>
            Ask
          </button>
        </div>
      </section>

      {loadDialogOpen && (
        <LoadDialog
          hats={sortedHats.filter((value) => value !== "auto")}
          loadHat={loadHat}
          setLoadHat={setLoadHat}
          loadMode={loadMode}
          setLoadMode={setLoadMode}
          selectedUploads={selectedUploads}
          onCancel={closeLoadDialog}
          onConfirm={confirmLoad}
          onSelectFiles={() => fileInputRef.current?.click()}
          onSelectFolder={() => folderInputRef.current?.click()}
          onUploadSelection={handleUploadSelection}
          fileInputRef={fileInputRef}
          folderInputRef={folderInputRef}
          busy={isLoadingTask}
        />
      )}
    </div>
  );
}

function MessageBubble({ message }) {
  return (
    <div className={`message-row ${message.role}`}>
      <article className="message-bubble">
        <div className="message-meta">
          <span>{message.title}</span>
          <span>{message.timeLabel}</span>
        </div>
        <div className={`message-body ${message.pending ? "pending" : ""}`}>
          {message.pending ? (
            <span className="pending-line">
              <span className="spinner" />
              <span>{message.body}</span>
            </span>
          ) : (
            message.body
          )}
        </div>
        {!message.pending && Boolean(message.citations?.length) && (
          <div className="citations">
            {message.citations.map((citation, index) => (
              <div key={`${citation}-${index}`}>{citation}</div>
            ))}
          </div>
        )}
      </article>
    </div>
  );
}

function LoadDialog({
  hats,
  loadHat,
  setLoadHat,
  loadMode,
  setLoadMode,
  selectedUploads,
  onCancel,
  onConfirm,
  onSelectFiles,
  onSelectFolder,
  onUploadSelection,
  fileInputRef,
  folderInputRef,
  busy,
}) {
  return (
    <div className="modal-scrim" role="dialog" aria-modal="true">
      <section className="modal-card">
        <div className="modal-head">
          <h2>Add More Files To Knowledge Base</h2>
          <p>Choose whether you want to add a few files or a whole folder, pick the target hat, and then confirm the load.</p>
        </div>

        <div className="modal-grid">
          <label className="control-label">
            Hat
            <select className="select-control" value={loadHat} onChange={(event) => setLoadHat(event.target.value)}>
              {hats.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>

          <div className="choice-grid">
            <button
              type="button"
              className={`choice-card ${loadMode === "files" ? "active" : ""}`}
              onClick={() => setLoadMode("files")}
            >
              <span className="choice-card-title">Pick files</span>
              <span className="choice-card-copy">Use this when you want to add a few PDFs or markdown files.</span>
            </button>
            <button
              type="button"
              className={`choice-card ${loadMode === "folder" ? "active" : ""}`}
              onClick={() => setLoadMode("folder")}
            >
              <span className="choice-card-title">Pick folder</span>
              <span className="choice-card-copy">Use this when you want to load a whole folder in one go.</span>
            </button>
          </div>

          <div className="picker-actions">
            <button type="button" className="picker-button" onClick={loadMode === "files" ? onSelectFiles : onSelectFolder}>
              {loadMode === "files" ? "Choose files" : "Choose folder"}
            </button>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            hidden
            multiple
            accept=".pdf,.md,.markdown"
            onChange={(event) => onUploadSelection(event.target.files)}
          />
          <input
            ref={folderInputRef}
            type="file"
            hidden
            multiple
            webkitdirectory="true"
            onChange={(event) => onUploadSelection(event.target.files)}
          />

          <div className="selection-box">
            {selectedUploads.length ? (
              <>
                <div>{selectedUploads.length} item(s) selected</div>
                <ol className="selection-list">
                  {selectedUploads.slice(0, 8).map((file) => (
                    <li key={file.webkitRelativePath || file.name}>{file.webkitRelativePath || file.name}</li>
                  ))}
                </ol>
                {selectedUploads.length > 8 && <div className="selection-hint">Only the first 8 items are shown here.</div>}
              </>
            ) : (
              <div className="selection-hint">Nothing selected yet. Choose files or a folder first, then click Add.</div>
            )}
          </div>
        </div>

        <div className="modal-actions">
          <button type="button" className="modal-action" onClick={onCancel} disabled={busy}>
            Cancel
          </button>
          <button type="button" className="modal-action primary" onClick={onConfirm} disabled={!selectedUploads.length || busy}>
            Add
          </button>
        </div>
      </section>
    </div>
  );
}

function createMessage({ role, title, body, citations = [], pending = false }) {
  return {
    id: `${role}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    role,
    title,
    body,
    citations,
    pending,
    timeLabel: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
  };
}

async function fetchJson(url, init = {}) {
  const response = await fetch(url, init);
  const text = await response.text();
  const payload = text ? safeJsonParse(text) : {};
  if (!response.ok) {
    throw new Error(payload.detail || text || "Request failed.");
  }
  return payload;
}

function safeJsonParse(text) {
  try {
    return JSON.parse(text);
  } catch (error) {
    return { detail: text };
  }
}

function formatLoadResult(result) {
  const topics = result.topic_folders?.length ? result.topic_folders.join(", ") : "none";
  const uploaded = result.uploaded_files?.length ? result.uploaded_files.join(", ") : "";
  const failedCount = result.failures?.length || 0;
  let summary = `Loaded ${result.document_count} document(s) into hat '${result.hat}' with load_id ${result.load_id}. Topics: ${topics}. Chunks: ${result.total_chunks}. Markdown segments: ${result.total_markdown_segments}. Failed files: ${failedCount}.`;
  if (uploaded) {
    summary += ` Uploaded: ${uploaded}.`;
  }
  if (failedCount) {
    const failed = result.failures.map((item) => item.source_uri).join(", ");
    summary += ` Failed: ${failed}.`;
  }
  return summary;
}

function normalizeError(error) {
  if (error instanceof Error) return error.message;
  return String(error);
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

ReactDOM.createRoot(document.getElementById("react-root")).render(<App />);
