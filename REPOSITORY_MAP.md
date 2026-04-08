# Repository Map for LLMs and AI Agents

This file is a fast orientation guide for agents that need to patch the repository safely.

## What This Repo Is

`open-arignan` is a local-first knowledge-base and retrieval system with:

- CLI commands for loading content, asking questions, deleting ingestions, and managing sessions
- hat-based storage namespaces under one app-home
- ingestion for markdown, PDFs, folders, and URLs
- hybrid retrieval across dense, lexical, and markdown/map artifacts
- local-LLM-authored `summary.md`, `map.md`, and `global_map.md`
- deterministic local stand-ins for answer generation, embedding, reranking, and session summarization

Important current reality:

- setup downloads model artifacts and runtime uses the local text model for markdown artifacts
- `ask` answer generation is still deterministic
- the default local text model is `Qwen/Qwen3-1.7B`
- dense retrieval prefers local Qdrant storage when available and falls back to JSON storage otherwise
- long-running CLI operations now print progress messages to `stderr`
- swallowed LLM failures now write full tracebacks to a session-local `exceptions.log`

## Top-Level Layout

```text
.
|-- README.md
|-- REPOSITORY_MAP.md
|-- setup.py
|-- pyproject.toml
|-- src/arignan/
|-- tests/
|-- docs/
|-- .setuptools/        # packaging scratch/output
`-- __pycache__/        # generated cache
```

Read these first:

- `README.md`: source-of-truth architecture intent
- `src/arignan/application.py`: main orchestration layer
- `src/arignan/cli.py`: CLI surface and user-visible flow
- `src/arignan/config.py`: defaults and settings behavior
- `src/arignan/setup_flow.py`: user bootstrap flow

## Main Execution Paths

### User Setup

Command:

```text
python setup.py
```

Flow:

1. `setup.py`
2. `src/arignan/setup_flow.py:run_setup`
3. package install
4. app-home initialization
5. model download
6. `bin/` launcher creation

Packaging note:

- `setup.py` also dispatches `egg_info`, `bdist_wheel`, and similar packaging commands back to setuptools. Do not break that split behavior when editing setup.

### CLI

Command:

```text
arignan ...
```

Flow:

1. `src/arignan/cli.py`
2. `src/arignan/config.py:load_config`
3. `src/arignan/application.py:ArignanApp`
4. subsystem-specific modules

Commands currently implemented:

- `load`
- `ask`
- `delete`
- `list-loads`
- `save-session`
- `load-session`
- `reset-session`

CLI behavior worth remembering:

- progress messages print to `stderr` with an `[arignan] ...` prefix
- `load --debug` and `ask --debug` print model-call traces and internal details
- uncaught CLI exceptions are logged to the active session log before being re-raised

### MCP

Flow:

1. `src/arignan/mcp/server.py`
2. `ArignanApp`
3. retrieval pipeline  reranker

Implemented MCP surface:

- tool: `retrieve_context`
- resource: `arignan://global-map`

## Source Map by Area

### Configuration, Paths, and Model Registry

- `src/arignan/config.py`
  - owns `AppConfig`
  - writes and loads `settings.json`
  - enforces that `embedding_model` is fixed and cannot be overridden in settings
- `src/arignan/paths.py`
  - resolves app home and settings paths
  - supports explicit `--app-home`, `ARIGNAN_HOME`, persisted app-home pointer, then fallback `~/.arignan`
- `src/arignan/model_registry.py`
  - shared model alias resolution
  - model-id sanitization
  - model storage directory derivation
  - this is the shared source used by setup and runtime loading

### Storage and Schemas

- `src/arignan/storage/layout.py`
  - creates and validates the on-disk structure
  - `auto` is a runtime selector only and cannot be a persisted hat name
- `src/arignan/models/`
  - canonical dataclasses for documents, chunks, ingestion events, retrieval hits, and sessions

### Ingestion

- `src/arignan/ingestion/discovery.py`
  - resolves inputs from URLs, markdown, PDFs, and folders
- `src/arignan/ingestion/parsers.py`
  - normalizes source content into `ParsedDocument`
- `src/arignan/ingestion/log.py`
  - append/read support for `ingestion_log.jsonl`
- `src/arignan/ingestion/service.py`
  - ingestion batch orchestration and `load_id` creation

### Indexing

- `src/arignan/indexing/chunking.py`
  - heading-aware chunking with overlap
  - current defaults favor larger chunks and fewer fragments
- `src/arignan/indexing/embedding.py`
  - `HashingEmbedder` for deterministic local behavior
  - sentence-transformer boundary is present for future real embedding runtime
- `src/arignan/indexing/dense.py`
  - `DenseIndexer`
  - `LocalDenseIndex`
  - prefers Qdrant local mode if available
  - falls back to JSON-backed dense storage if Qdrant import is unavailable
- `src/arignan/indexing/lexical.py`
  - BM25-style lexical index

### Grouping and Markdown Artifacts

- `src/arignan/grouping/planner.py`
  - decides standalone vs merge vs segment
  - deterministic heuristic replacement for an LLM grouping policy
- `src/arignan/markdown/rendering.py`
  - shared deterministic rendering helpers
  - shared keyword extraction / text cleanup / markdown table helpers
  - active deterministic source for exported markdown helpers
- `src/arignan/markdown/generator.py`
  - repository/storage layer for topic folders
  - writes topic manifests
  - writes topic markdown trees
  - regenerates hat/global maps from manifests
  - now supports batching by skipping map refreshes until the caller requests them
- `src/arignan/markdown/writer.py`
  - artifact rendering boundary
  - heuristic fallback rendering
  - local-LLM-backed artifact generation
  - progress reporting for LLM calls
  - session-local traceback logging for swallowed LLM failures
- `src/arignan/llm/runtime.py`
  - loads the configured local text model from `<app_home>/models`
  - provides the transformers-based local generation adapter

Topic folder invariant:

```text
<app_home>/hats/<hat>/summaries/<topic_folder>/
|-- original_files/
|-- markdown_tree/
`-- .topic_manifest.json
```

### Retrieval and Reranking

- `src/arignan/retrieval/pipeline.py`
  - query expansion
  - hat selection
  - dense search
  - lexical search
  - markdown/map retrieval
  - reciprocal rank fusion
  - progress reporting for multi-step retrieval
- `src/arignan/retrieval/reranking.py`
  - heuristic reranker
  - cross-encoder boundary for future runtime integration

### Sessions and Exception Logs

- `src/arignan/session/store.py`
  - persisted active/saved session JSON
  - per-PID active session directories for logs
  - active exception log path helper
- `src/arignan/session/manager.py`
  - PID-scoped session lifecycle
  - rollover logic
  - idle timeout metadata handling
- `src/arignan/session/summarizer.py`
  - current deterministic summarizer
- `src/arignan/session/exception_log.py`
  - appends full exception tracebacks to active-session log files

Active session log invariant:

```text
<app_home>/sessions/active/
|-- pid-<pid>.json
`-- pid-<pid>/exceptions.log
```

### App Orchestration

- `src/arignan/application.py`
  - highest-value file for most behavior changes
  - wires ingestion, indexing, markdown generation, retrieval, reranking, deletion, and sessions
  - centralizes dense and lexical indexer construction
  - batches map regeneration during `load` and grouped-topic regeneration to avoid redundant LLM calls
  - owns user-facing progress emission for multi-step operations

## Behavior That Is Intentionally Simplified

These are common places where an agent may assume there is a real model/runtime when there is not yet one:

- `src/arignan/application.py:synthesize_answer`
  - answers are assembled deterministically from retrieval hits
- `src/arignan/indexing/embedding.py:HashingEmbedder`
  - used for deterministic embedding behavior in tests and local flows
- `src/arignan/retrieval/reranking.py:HeuristicReranker`
  - current default reranker in the app
- `src/arignan/session/summarizer.py`
  - rollover summary is deterministic, not LLM-authored

If you upgrade one of these areas to a real runtime, patch tests and docs at the same time.

## Common Patch Tasks

### Change CLI behavior or user-facing progress

Touch:

- `src/arignan/cli.py`
- `src/arignan/application.py`
- relevant integration tests in `tests/integration/test_cli_smoke.py`

### Change setup/bootstrap behavior

Touch:

- `setup.py`
- `src/arignan/setup_flow.py`
- `src/arignan/model_registry.py`
- `tests/unit/test_setup_flow.py`
- `tests/unit/test_setup_py_dispatch.py`

### Change app-home storage layout

Touch:

- `src/arignan/storage/layout.py`
- `src/arignan/markdown/generator.py`
- `src/arignan/session/store.py`
- `src/arignan/application.py`
- storage/markdown/session integration tests

Be careful:

- topic manifests are used by map regeneration and delete/regeneration logic
- active session logs now live beside active session JSON

### Change ingestion or parsing

Touch:

- `src/arignan/ingestion/discovery.py`
- `src/arignan/ingestion/parsers.py`
- `src/arignan/ingestion/service.py`
- ingestion and parser tests

### Change chunking or retrieval quality

Touch:

- `src/arignan/indexing/chunking.py`
- `src/arignan/retrieval/pipeline.py`
- `src/arignan/retrieval/reranking.py`
- retrieval/reranking integration tests

### Change grouping or markdown generation

Touch:

- `src/arignan/grouping/planner.py`
- `src/arignan/markdown/rendering.py`
- `src/arignan/markdown/generator.py`
- `src/arignan/markdown/writer.py`
- `tests/integration/test_grouping_pipeline.py`
- `tests/integration/test_markdown_repository.py`
- `tests/integration/test_end_to_end_flow.py`
- `tests/unit/test_markdown_writer.py`

### Change session semantics or exception logging

Touch:

- `src/arignan/session/manager.py`
- `src/arignan/session/store.py`
- `src/arignan/session/exception_log.py`
- `src/arignan/session/summarizer.py`
- `tests/unit/test_session_manager.py`
- `tests/unit/test_session_logging.py`
- `tests/integration/test_session_persistence.py`

## Key Invariants

- `local_llm_model` is configurable; `embedding_model` is not
- app home defaults to `~/.arignan`
- `--hat` defaults to `auto`
- persisted hat names cannot be `auto`
- grouped topic state is tracked through `.topic_manifest.json`
- deleting a grouped load should regenerate surviving grouped markdown, not blindly delete the topic
- `map.md` and `global_map.md` are regenerated from manifests
- setup downloads models into `<app_home>/models`
- active session exceptions are logged to `<app_home>/sessions/active/pid-<pid>/exceptions.log`
- `load` and grouped delete/regeneration should avoid redundant map/global-map refreshes inside inner loops

## Test Map

Test layout:

- `tests/unit/`: fast module-level behavior
- `tests/integration/`: storage, CLI, retrieval, markdown, session, MCP, and end-to-end flows
- `tests/fixtures/`: reusable markdown and grouped-topic inputs
- `tests/fixtures/pdf_fixture.py`: programmatic PDF fixture helper

Highest-signal tests when patching:

- `tests/integration/test_end_to_end_flow.py`
- `tests/integration/test_cli_smoke.py`
- `tests/integration/test_markdown_repository.py`
- `tests/integration/test_retrieval_pipeline.py`
- `tests/unit/test_markdown_writer.py`
- `tests/unit/test_session_logging.py`
- `tests/unit/test_setup_flow.py`

Main test command:

```text
python -m pytest
```

## Agent Tips

- Start from `ArignanApp` if you need to understand user-visible behavior
- Start from `StorageLayout`, `MarkdownRepository`, and `SessionStore` if your patch changes filesystem shape
- Start from `model_registry.py` if your patch touches setup/runtime model resolution
- Start from `markdown/rendering.py` if your patch is deterministic markdown cleanup or keyword extraction
- Treat docs in `docs/` as implementation-state notes, not the primary runtime surface
- Ignore `.setuptools/` and `__pycache__/` unless you are debugging packaging or caches
- If a change touches setup, retrieval, storage layout, or sessions, verify both unit and integration coverage
