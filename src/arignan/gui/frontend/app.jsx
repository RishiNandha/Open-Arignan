const { useEffect, useLayoutEffect, useMemo, useRef, useState } = React;

const ANSWER_MODES = ["default", "light", "none", "raw"];

function App() {
  const [options, setOptions] = useState({
    hats: ["auto"],
    default_hat: "default",
    answer_modes: ANSWER_MODES,
    default_rerank_top_k: 8,
    default_answer_context_top_k: 8,
    default_show_thinking: true,
  });
  const [library, setLibrary] = useState({
    hats: [],
    loads: [],
  });
  const [hat, setHat] = useState("auto");
  const [answerMode, setAnswerMode] = useState("default");
  const [rerankTopK, setRerankTopK] = useState(8);
  const [answerContextTopK, setAnswerContextTopK] = useState(8);
  const [showThinking, setShowThinking] = useState(true);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([
    createMessage({
      role: "assistant",
      title: "Arignan",
      body: "Add more files to the knowledge base, then ask questions here. Long operations will show a compact live status in the same assistant bubble.",
    }),
  ]);
  const [loadDialogOpen, setLoadDialogOpen] = useState(false);
  const [manageDialogOpen, setManageDialogOpen] = useState(false);
  const [loadHat, setLoadHat] = useState("default");
  const [newHatName, setNewHatName] = useState("");
  const [loadMode, setLoadMode] = useState("files");
  const [selectedUploads, setSelectedUploads] = useState([]);
  const [isAsking, setIsAsking] = useState(false);
  const [isLoadingTask, setIsLoadingTask] = useState(false);
  const [isDeletingTask, setIsDeletingTask] = useState(false);
  const messagesRef = useRef(null);
  const shouldFollowMessagesRef = useRef(true);
  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);

  useEffect(() => {
    bootstrap();
  }, []);

  useLayoutEffect(() => {
    if (!messagesRef.current) return;
    if (!shouldFollowMessagesRef.current) return;
    messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
  }, [messages]);

  const sortedHats = useMemo(() => options.hats || ["auto"], [options]);

  async function bootstrap() {
    const payload = await fetchJson("/api/options");
    setOptions(payload);
    setHat(payload.hats?.includes("auto") ? "auto" : payload.default_hat || "default");
    setLoadHat(payload.default_hat || "default");
    setAnswerMode("default");
    setRerankTopK(payload.default_rerank_top_k || 8);
    setAnswerContextTopK(payload.default_answer_context_top_k || 8);
    setShowThinking(payload.default_show_thinking !== false);
    await refreshLibrary();
  }

  async function refreshLibrary() {
    const payload = await fetchJson("/api/library");
    setLibrary(payload);
  }

  async function openFileTarget(target) {
    try {
      await fetchJson(`/api/open-file/${target}`, { method: "POST" });
    } catch (error) {
      window.alert(normalizeError(error));
    }
  }

  function openLoadDialog() {
    setLoadDialogOpen(true);
    setLoadHat(hat === "auto" ? options.default_hat || "default" : hat);
    setNewHatName("");
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

  function openManageDialog() {
    refreshLibrary();
    setManageDialogOpen(true);
  }

  function closeManageDialog() {
    setManageDialogOpen(false);
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
    const effectiveHat = (newHatName || "").trim() || loadHat;
    const label = selectedUploads.map((file) => file.webkitRelativePath || file.name).join(", ");
    appendMessage(
      createMessage({
        role: "user",
        title: "You",
        body: `Add more files to knowledge base in hat '${effectiveHat}': ${label}`,
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
      formData.append("hat", effectiveHat);
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
          bootstrap();
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

  async function deleteLoad(load) {
    if (isDeletingTask) return;
    const confirmed = window.confirm(`Delete load '${load.load_id}' from hat '${load.hat}'?`);
    if (!confirmed) return;
    setIsDeletingTask(true);
    appendMessage(
      createMessage({
        role: "user",
        title: "You",
        body: `Delete load '${load.load_id}' from hat '${load.hat}'`,
      })
    );
    const pendingMessage = createMessage({
      role: "assistant",
      title: "Arignan",
      body: "Deleting selected loads...",
      pending: true,
    });
    const pendingId = pendingMessage.id;
    appendMessage(pendingMessage);
    try {
      const payload = await fetchJson("/api/delete/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ load_ids: [load.load_id] }),
      });
      await followTask({
        taskId: payload.task_id,
        pendingId,
        onComplete: (result) => {
          patchMessage(pendingId, {
            pending: false,
            body: result.message,
            citations: [],
          });
          bootstrap();
        },
      });
    } catch (error) {
      patchMessage(pendingId, {
        pending: false,
        body: normalizeError(error),
        citations: [],
      });
    } finally {
      setIsDeletingTask(false);
    }
  }

  async function deleteHat(hatName) {
    if (isDeletingTask) return;
    const confirmed = window.confirm(`Delete entire hat '${hatName}'? This removes all indexes, summaries, and source copies for that hat.`);
    if (!confirmed) return;
    setIsDeletingTask(true);
    appendMessage(
      createMessage({
        role: "user",
        title: "You",
        body: `Delete hat '${hatName}'`,
      })
    );
    const pendingMessage = createMessage({
      role: "assistant",
      title: "Arignan",
      body: `Deleting hat '${hatName}'...`,
      pending: true,
    });
    const pendingId = pendingMessage.id;
    appendMessage(pendingMessage);
    try {
      const payload = await fetchJson("/api/delete/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hat: hatName }),
      });
      await followTask({
        taskId: payload.task_id,
        pendingId,
        onComplete: (result) => {
          patchMessage(pendingId, {
            pending: false,
            body: result.message,
            citations: [],
          });
          bootstrap();
        },
      });
    } catch (error) {
      patchMessage(pendingId, {
        pending: false,
        body: normalizeError(error),
        citations: [],
      });
    } finally {
      setIsDeletingTask(false);
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
        body: JSON.stringify({
          question: trimmed,
          hat,
          answer_mode: answerMode,
          rerank_top_k: normalizeRerankTopK(rerankTopK, options.default_rerank_top_k || 8),
          answer_context_top_k: normalizeRerankTopK(
            answerContextTopK,
            options.default_answer_context_top_k || 8
          ),
          show_thinking: showThinking,
        }),
      });
      await followTask({
        taskId: payload.task_id,
        pendingId,
        onComplete: (snapshot) => {
          const result = snapshot.result || {};
          patchMessage(pendingId, {
            pending: false,
            body: result.answer,
            citations: result.citations || [],
            partialThinking: snapshot.partial_thinking || "",
            thoughtStartedAt: snapshot.thought_started_at || null,
            thoughtFinishedAt: snapshot.thought_finished_at || null,
            thoughtUsage: snapshot.thought_usage || null,
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
      if (snapshot.message || snapshot.partial_answer || snapshot.partial_thinking || snapshot.progress_log) {
        patchMessage(pendingId, {
          body: snapshot.message || "Working...",
          pending: snapshot.status === "running",
          progressLog: snapshot.progress_log || [],
          partialAnswer: snapshot.partial_answer || "",
          partialThinking: snapshot.partial_thinking || "",
          thoughtStartedAt: snapshot.thought_started_at || null,
          thoughtFinishedAt: snapshot.thought_finished_at || null,
          thoughtUsage: snapshot.thought_usage || null,
        });
      }
      if (snapshot.status === "done") {
        onComplete(snapshot);
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
        <div className="brand-block">
          <h1 className="brand-title">Open Arignan</h1>
          <div className="brand-tools">
            <button type="button" className="ghost-button header-tool-button" onClick={() => openFileTarget("logs")}>
              Open Logs
            </button>
            <button
              type="button"
              className="ghost-button header-tool-button"
              onClick={() => openFileTarget("settings")}
            >
              Open Settings
            </button>
            <button
              type="button"
              className="ghost-button header-tool-button"
              onClick={() => openFileTarget("prompts")}
            >
              Open Prompts
            </button>
          </div>
        </div>
        <div className="header-actions">
          <button type="button" className="ghost-button" onClick={openManageDialog}>
            Manage Knowledge Base
          </button>
          <button type="button" className="add-button" onClick={openLoadDialog}>
            <span className="plus">+</span>
            <span>Add More Files To Knowledge Base</span>
          </button>
        </div>
      </header>

      <section className="chat-panel">
        <div className="messages" ref={messagesRef} onScroll={handleMessagesScroll}>
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
          <label className="control-label compact-control">
            Rerank Candidates
            <input
              className="text-control"
              type="number"
              min="1"
              step="1"
              value={rerankTopK}
              onChange={(event) => setRerankTopK(event.target.value)}
            />
          </label>
          <label className="control-label compact-control">
            Final Context
            <input
              className="text-control"
              type="number"
              min="1"
              step="1"
              value={answerContextTopK}
              onChange={(event) => setAnswerContextTopK(event.target.value)}
            />
          </label>
          <button
            type="button"
            className={`toggle-control${showThinking ? " is-active" : ""}`}
            aria-pressed={showThinking}
            onClick={() => setShowThinking((value) => !value)}
          >
            <span className="toggle-pill" aria-hidden="true">
              <span className="toggle-thumb" />
            </span>
            <span>Show Thinking</span>
          </button>
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
          newHatName={newHatName}
          setNewHatName={setNewHatName}
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

      {manageDialogOpen && (
        <ManageDialog
          hats={library.hats || []}
          loads={library.loads || []}
          onClose={closeManageDialog}
          onDeleteLoad={deleteLoad}
          onDeleteHat={deleteHat}
          busy={isDeletingTask}
        />
      )}
    </div>
  );

  function handleMessagesScroll() {
    const container = messagesRef.current;
    if (!container) return;
    shouldFollowMessagesRef.current = isNearBottom(container);
  }
}

function MessageBubble({ message }) {
  const [thinkingOpen, setThinkingOpen] = useState(false);
  const thinkingSummary = summarizeThinking(message);

  return (
    <div className={`message-row ${message.role}`}>
      <article className="message-bubble">
        <div className="message-meta">
          <span>{message.title}</span>
          <span>{message.timeLabel}</span>
        </div>
        <div className={`message-body ${message.pending ? "pending" : ""}`}>
          {message.pending ? (
            <div className="pending-stack">
              <span className="pending-line">
                <span className="spinner" />
                <span>{message.body}</span>
              </span>
              {Boolean(message.progressLog?.length) && (
                <div className="pending-progress">
                  {message.progressLog.slice(-5).map((line, index) => (
                    <div key={`${line}-${index}`} className="pending-progress-line">
                      {line}
                    </div>
                  ))}
                </div>
              )}
              {message.partialThinking ? (
                <div className="thinking-stream">
                  <div className="thinking-label">Thinking…</div>
                  <div className="thinking-content">{message.partialThinking}</div>
                </div>
              ) : null}
              {message.partialAnswer ? <div className="streaming-answer">{message.partialAnswer}</div> : null}
            </div>
          ) : (
            <>
              {message.body}
              {message.partialThinking ? (
                <div className="thinking-panel">
                  <button type="button" className="thinking-toggle" onClick={() => setThinkingOpen((open) => !open)}>
                    {thinkingOpen ? "Hide" : "Show"} {thinkingSummary}
                  </button>
                  {thinkingOpen ? <div className="thinking-content">{message.partialThinking}</div> : null}
                </div>
              ) : null}
            </>
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
  newHatName,
  setNewHatName,
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
            Existing Hat
            <select className="select-control" value={loadHat} onChange={(event) => setLoadHat(event.target.value)}>
              {hats.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>

          <label className="control-label">
            Or Create New Hat
            <input
              className="text-control"
              type="text"
              placeholder="Type a new hat name..."
              value={newHatName}
              onChange={(event) => setNewHatName(event.target.value)}
            />
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

function ManageDialog({ hats, loads, onClose, onDeleteLoad, onDeleteHat, busy }) {
  return (
    <div className="modal-scrim" role="dialog" aria-modal="true">
      <section className="modal-card manage-modal">
        <div className="modal-head">
          <h2>Manage Knowledge Base</h2>
          <p>Delete older loads or remove an entire hat from the local knowledge base.</p>
        </div>

        <div className="manage-grid">
          <section className="manage-section">
            <div className="manage-title-row">
              <h3>Loads</h3>
              <span>{loads.length}</span>
            </div>
            <div className="manage-list">
              {loads.length ? (
                loads.map((load) => (
                  <article key={load.load_id} className="manage-item">
                    <div className="manage-item-main">
                      <div className="manage-item-title">{load.load_id}</div>
                      <div className="manage-item-meta">
                        Hat: {load.hat} | Topics: {(load.topic_folders || []).join(", ") || "n/a"}
                      </div>
                      <div className="manage-item-copy">
                        {(load.source_items || []).slice(0, 3).join(", ") || "No source details"}
                      </div>
                    </div>
                    <button type="button" className="danger-button" disabled={busy} onClick={() => onDeleteLoad(load)}>
                      Delete
                    </button>
                  </article>
                ))
              ) : (
                <div className="selection-hint">No ingestions found yet.</div>
              )}
            </div>
          </section>

          <section className="manage-section">
            <div className="manage-title-row">
              <h3>Hats</h3>
              <span>{hats.length}</span>
            </div>
            <div className="manage-list">
              {hats.length ? (
                hats.map((hat) => (
                  <article key={hat} className="manage-item">
                    <div className="manage-item-main">
                      <div className="manage-item-title">{hat}</div>
                      <div className="manage-item-copy">Delete this only if you want to remove the entire hat.</div>
                    </div>
                    <button type="button" className="danger-button" disabled={busy} onClick={() => onDeleteHat(hat)}>
                      Delete Hat
                    </button>
                  </article>
                ))
              ) : (
                <div className="selection-hint">No hats found yet.</div>
              )}
            </div>
          </section>
        </div>

        <div className="modal-actions">
          <button type="button" className="modal-action" onClick={onClose} disabled={busy}>
            Close
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
    progressLog: pending ? [body] : [],
    partialAnswer: "",
    partialThinking: "",
    thoughtStartedAt: null,
    thoughtFinishedAt: null,
    thoughtUsage: null,
    timeLabel: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
  };
}

function summarizeThinking(message) {
  const seconds = thinkingDurationSeconds(message);
  if (seconds !== null) {
    return `thought for ${seconds.toFixed(seconds >= 10 ? 0 : 1)}s`;
  }
  return "thinking trace";
}

function thinkingDurationSeconds(message) {
  if (message.thoughtStartedAt && message.thoughtFinishedAt) {
    const started = Date.parse(message.thoughtStartedAt);
    const finished = Date.parse(message.thoughtFinishedAt);
    if (Number.isFinite(started) && Number.isFinite(finished) && finished >= started) {
      return (finished - started) / 1000;
    }
  }
  const totalDuration = message.thoughtUsage?.total_duration;
  if (typeof totalDuration === "number" && totalDuration > 0) {
    return totalDuration / 1_000_000_000;
  }
  return null;
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

function isNearBottom(container) {
  const remaining = container.scrollHeight - container.scrollTop - container.clientHeight;
  return remaining <= 48;
}

function normalizeRerankTopK(value, fallback) {
  const parsed = Number.parseInt(String(value), 10);
  if (Number.isNaN(parsed) || parsed < 1) return fallback;
  return parsed;
}

ReactDOM.createRoot(document.getElementById("react-root")).render(<App />);
