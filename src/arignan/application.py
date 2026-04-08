from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from arignan.config import AppConfig
from arignan.grouping import GroupingDecision, GroupingPlan, GroupingPlanner, estimate_markdown_length
from arignan.indexing import Chunker, DenseIndexer, HashingEmbedder, LexicalIndex, LexicalIndexer, LocalDenseIndex, tokenize
from arignan.ingestion import IngestionLog, IngestionService
from arignan.llm import TransformersTextGenerator
from arignan.markdown import MarkdownRepository
from arignan.markdown.writer import HeuristicArtifactWriter, LLMArtifactWriter
from arignan.models import ChunkRecord, LoadEvent, LoadOperation, ParsedDocument, RetrievalHit
from arignan.retrieval import HeuristicReranker, RetrievalPipeline
from arignan.session import SessionExceptionLogger, SessionManager, SessionStore
from arignan.storage import StorageLayout
from arignan.tracing import ModelCallTrace, ModelTraceCollector


@dataclass(slots=True)
class LoadDocumentTrace:
    source_uri: str
    title: str
    topic_folder: str
    grouping_decision: str
    chunk_count: int
    markdown_segment_count: int
    rationale: list[str]
    segment_titles: list[str]


@dataclass(slots=True)
class LoadResult:
    load_id: str
    hat: str
    document_count: int
    topic_folders: list[str]
    artifact_paths: list[Path]
    total_chunks: int
    total_markdown_segments: int
    traces: list[LoadDocumentTrace]
    model_calls: list[ModelCallTrace]


@dataclass(slots=True)
class AskDebug:
    expanded_query: str
    selected_hat: str
    dense_hits: list[RetrievalHit]
    lexical_hits: list[RetrievalHit]
    map_hits: list[RetrievalHit]
    fused_hits: list[RetrievalHit]
    reranked_hits: list[RetrievalHit]
    model_calls: list[ModelCallTrace]


@dataclass(slots=True)
class AskResult:
    question: str
    selected_hat: str
    answer: str
    citations: list[str]
    debug: AskDebug


@dataclass(slots=True)
class DeleteResult:
    deleted_load_ids: list[str]
    missing_load_ids: list[str]
    deleted_topics: list[str]


@dataclass(slots=True)
class DeleteHatResult:
    hat: str
    existed: bool
    deleted_load_ids: list[str]
    deleted_topics: list[str]


class ArignanApp:
    def __init__(
        self,
        config: AppConfig,
        progress_sink: Callable[[str], None] | None = None,
        terminal_pid: int | None = None,
    ) -> None:
        self.config = config
        self.progress_sink = progress_sink
        self.terminal_pid = terminal_pid or os.getppid() or os.getpid()
        self.layout = StorageLayout.from_home(config.app_home).ensure()
        self.ingestion_log = IngestionLog(self.layout.ingestion_log_path)
        self.ingestion_service = IngestionService(self.ingestion_log)
        self.grouping_planner = GroupingPlanner(max_md_length=config.markdown.max_md_length)
        self.embedder = HashingEmbedder()
        self.chunker = Chunker(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
        )
        self.reranker = HeuristicReranker()
        self.session_manager = SessionManager(SessionStore(config.app_home), config.session)
        self.exception_logger = SessionExceptionLogger(self.session_manager.store, self.terminal_pid)
        self.trace_collector = ModelTraceCollector()
        artifact_writer = LLMArtifactWriter(
            generator=TransformersTextGenerator(config),
            fallback=HeuristicArtifactWriter(),
            trace_sink=self.trace_collector,
            progress_sink=self.progress_sink,
            exception_logger=self.exception_logger,
        )
        self.markdown_repository = MarkdownRepository(artifact_writer=artifact_writer)

    def load(self, input_ref: str, hat: str = "auto") -> LoadResult:
        self.trace_collector.clear()
        target_hat = self.config.default_hat if hat == "auto" else hat
        self._emit_progress(f"Scanning input for load into hat '{target_hat}'...")
        self.layout.hat(target_hat).ensure()
        batch = self.ingestion_service.ingest(input_ref, hat=target_hat, log_event=False)
        self._emit_progress(f"Discovered {len(batch.documents)} document(s).")
        artifact_paths: list[Path] = []
        topic_folders: list[str] = []
        total_chunks = 0
        total_markdown_segments = 0
        traces: list[LoadDocumentTrace] = []

        for index, document in enumerate(batch.documents, start=1):
            label = document.source.title or Path(document.source.source_uri).name
            self._emit_progress(f"[{index}/{len(batch.documents)}] Checking related material for '{label}'...")
            related_hits = self._related_hits_for_document(document)
            self._emit_progress(f"[{index}/{len(batch.documents)}] Planning grouping for '{label}'...")
            plan = self.grouping_planner.plan(document, related_hits=related_hits)
            existing_docs = self._existing_topic_documents(target_hat, plan.topic_folder) if plan.decision is GroupingDecision.MERGE else []
            documents_for_topic = existing_docs + [document]
            normalized_plan = self._normalize_plan(plan, documents_for_topic)

            self._emit_progress(f"[{index}/{len(batch.documents)}] Chunking and indexing '{label}'...")
            chunks = self.chunker.chunk_document(document)
            chunks = self._assign_topic_folder(chunks, normalized_plan.topic_folder)
            self._dense_indexer(target_hat).index_chunks(chunks)
            self._lexical_indexer(target_hat).index_chunks(chunks)
            total_chunks += len(chunks)

            self._emit_progress(f"[{index}/{len(batch.documents)}] Writing topic '{normalized_plan.topic_folder}'...")
            artifact = self.markdown_repository.write_topic(
                self.layout,
                hat=target_hat,
                documents=documents_for_topic,
                plan=normalized_plan,
                refresh_maps=False,
            )
            artifact_paths.extend(artifact.markdown_paths)
            total_markdown_segments += len(artifact.markdown_paths)
            if artifact.topic_folder not in topic_folders:
                topic_folders.append(artifact.topic_folder)
            traces.append(
                LoadDocumentTrace(
                    source_uri=document.source.source_uri,
                    title=document.source.title or Path(document.source.source_uri).name,
                    topic_folder=artifact.topic_folder,
                    grouping_decision=normalized_plan.decision.value,
                    chunk_count=len(chunks),
                    markdown_segment_count=len(artifact.markdown_paths),
                    rationale=list(normalized_plan.rationale),
                    segment_titles=[segment.title for segment in normalized_plan.segments],
                )
            )

        if batch.documents:
            self._emit_progress(f"Refreshing map.md for hat '{target_hat}'...")
            self.markdown_repository.update_hat_map(self.layout, target_hat)
            self._emit_progress("Refreshing global_map.md...")
            self.markdown_repository.update_global_map(self.layout)

        self._emit_progress("Recording ingestion log...")
        self.ingestion_log.append(
            LoadEvent(
                load_id=batch.load_id,
                operation=LoadOperation.INGEST,
                hat=target_hat,
                created_at=datetime.now(timezone.utc).isoformat(),
                source_items=batch.source_items,
                artifact_paths=artifact_paths,
                topic_folders=topic_folders,
                metadata={"input_ref": batch.input_ref},
            )
        )
        return LoadResult(
            load_id=batch.load_id,
            hat=target_hat,
            document_count=len(batch.documents),
            topic_folders=topic_folders,
            artifact_paths=artifact_paths,
            total_chunks=total_chunks,
            total_markdown_segments=total_markdown_segments,
            traces=traces,
            model_calls=self.trace_collector.snapshot(),
        )

    def ask(self, question: str, hat: str = "auto", terminal_pid: int | None = None) -> AskResult:
        self.trace_collector.clear()
        pid = terminal_pid or self.terminal_pid
        self.session_manager.append_turn(pid, role="user", content=question)
        self._emit_progress("Running retrieval pipeline...")
        bundle = RetrievalPipeline(
            self.layout,
            embedder=self.embedder,
            dense_limit=self.config.retrieval.dense_top_k,
            lexical_limit=self.config.retrieval.lexical_top_k,
            map_limit=self.config.retrieval.map_top_k,
            fused_limit=self.config.retrieval.fused_top_k,
            trace_sink=self.trace_collector,
            progress_sink=self.progress_sink,
        ).retrieve(question, hat=hat)
        self._emit_progress("Reranking retrieved candidates...")
        self.trace_collector.record(
            component="reranker",
            task="rerank retrieval candidates",
            model_name=self.reranker.model_name,
            backend=getattr(self.reranker, "backend_name", type(self.reranker).__name__),
            item_count=len(bundle.fused_hits),
            detail=f"top_k={self.config.retrieval.rerank_top_k}",
        )
        reranked = self.reranker.rerank(
            bundle.expanded_query,
            bundle.fused_hits,
            limit=self.config.retrieval.rerank_top_k,
            min_score=0.05,
        )
        answer_hits = _content_hits(reranked)
        self._emit_progress("Composing final answer...")
        answer = synthesize_answer(question, answer_hits, expanded_query=bundle.expanded_query)
        self.session_manager.append_turn(pid, role="assistant", content=answer)
        citations = _unique_citations(answer_hits, limit=3)
        debug = AskDebug(
            expanded_query=bundle.expanded_query,
            selected_hat=bundle.selected_hat,
            dense_hits=bundle.dense_hits,
            lexical_hits=bundle.lexical_hits,
            map_hits=bundle.map_hits,
            fused_hits=bundle.fused_hits,
            reranked_hits=reranked,
            model_calls=self.trace_collector.snapshot(),
        )
        return AskResult(question=question, selected_hat=bundle.selected_hat, answer=answer, citations=citations, debug=debug)

    def list_events(self) -> list[LoadEvent]:
        return self.ingestion_log.read_all()

    def list_ingestions(self) -> list[LoadEvent]:
        return [event for event in self.ingestion_log.read_all() if event.operation is LoadOperation.INGEST]

    def list_loads(self) -> list[LoadEvent]:
        return self.list_events()

    def delete(self, load_ids: list[str]) -> DeleteResult:
        self._emit_progress(f"Deleting {len(load_ids)} load(s)...")
        all_events = self.ingestion_log.read_all()
        ingest_events = {
            event.load_id: event
            for event in all_events
            if event.operation is LoadOperation.INGEST
        }
        deleted_topics: list[str] = []
        missing = [load_id for load_id in load_ids if load_id not in ingest_events]
        to_delete = [load_id for load_id in load_ids if load_id in ingest_events]

        affected_hats = {ingest_events[load_id].hat for load_id in to_delete}
        for hat in affected_hats:
            self._emit_progress(f"Removing indexed chunks from hat '{hat}'...")
            dense = self._dense_indexer(hat, trace=False)
            lexical = self._lexical_indexer(hat)
            for load_id in to_delete:
                dense.delete_load(load_id)
                lexical.delete_load(load_id)

            for manifest_path in sorted(self.layout.hat(hat).summaries_dir.glob("*/.topic_manifest.json")):
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                documents = [ParsedDocument.from_dict(item) for item in payload.get("documents", [])]
                remaining = [document for document in documents if document.load_id not in to_delete]
                topic_dir = manifest_path.parent
                if len(remaining) == len(documents):
                    continue
                if not remaining:
                    shutil.rmtree(topic_dir)
                    deleted_topics.append(payload["topic_folder"])
                    continue
                self._emit_progress(f"Regenerating topic '{payload['topic_folder']}' in hat '{hat}'...")
                plan = GroupingPlan(
                    decision=GroupingDecision.MERGE if len(remaining) > 1 else GroupingDecision.STANDALONE,
                    topic_folder=payload["topic_folder"],
                    estimated_length=sum(estimate_markdown_length(document.full_text) for document in remaining),
                )
                self.markdown_repository.regenerate_topic(
                    self.layout,
                    hat=hat,
                    documents=remaining,
                    plan=plan,
                    refresh_maps=False,
                )
                deleted_topics.append(payload["topic_folder"])

            self._emit_progress(f"Refreshing map.md for hat '{hat}'...")
            self.markdown_repository.update_hat_map(self.layout, hat)

        if affected_hats:
            self._emit_progress("Refreshing global_map.md...")
            self.markdown_repository.update_global_map(self.layout)
        if to_delete:
            self._emit_progress("Recording deletion log...")
            self.ingestion_log.append(
                LoadEvent(
                    load_id=f"delete-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                    operation=LoadOperation.DELETE,
                    hat="multiple" if len(affected_hats) > 1 else next(iter(affected_hats), self.config.default_hat),
                    created_at=datetime.now(timezone.utc).isoformat(),
                    source_items=to_delete,
                    artifact_paths=[],
                    topic_folders=deleted_topics,
                    metadata={"deleted_load_ids": to_delete},
                )
            )
        return DeleteResult(deleted_load_ids=to_delete, missing_load_ids=missing, deleted_topics=deleted_topics)

    def delete_hat(self, hat: str) -> DeleteHatResult:
        hat_layout = self.layout.hat(hat)
        existed = hat_layout.root.exists()
        deleted_topics: list[str] = []
        deleted_load_ids = sorted(
            event.load_id
            for event in self.ingestion_log.read_all()
            if event.operation is LoadOperation.INGEST and event.hat == hat
        )

        if existed:
            self._emit_progress(f"Deleting hat '{hat}' from storage...")
            if hat_layout.summaries_dir.exists():
                deleted_topics = sorted(path.name for path in hat_layout.summaries_dir.iterdir() if path.is_dir())
            shutil.rmtree(hat_layout.root)
            self._emit_progress("Refreshing global_map.md...")
            self.markdown_repository.update_global_map(self.layout)
            self._emit_progress("Recording deletion log...")
            self.ingestion_log.append(
                LoadEvent(
                    load_id=f"delete-hat-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                    operation=LoadOperation.DELETE,
                    hat=hat,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    source_items=[f"hat:{hat}"],
                    artifact_paths=[],
                    topic_folders=deleted_topics,
                    metadata={"deleted_hat": hat, "deleted_load_ids": deleted_load_ids, "mode": "hat"},
                )
            )

        return DeleteHatResult(
            hat=hat,
            existed=existed,
            deleted_load_ids=deleted_load_ids,
            deleted_topics=deleted_topics,
        )

    def save_session(self, terminal_pid: int | None = None, destination: Path | None = None) -> Path:
        pid = terminal_pid or self.terminal_pid
        return self.session_manager.save_session(pid, destination=destination)

    def load_session(self, source: Path, terminal_pid: int | None = None):
        pid = terminal_pid or self.terminal_pid
        return self.session_manager.load_session(pid, source)

    def reset_session(self, terminal_pid: int | None = None):
        pid = terminal_pid or self.terminal_pid
        return self.session_manager.reset_session(pid)

    def _related_hits_for_document(self, document: ParsedDocument) -> list[RetrievalHit]:
        query = f"{document.source.title or ''} {document.full_text[:200]}".strip()
        dense = self._dense_indexer(document.hat)
        lexical = self._lexical_indexer(document.hat)
        return dense.search(query, 3) + lexical.search(query, 3)

    def _existing_topic_documents(self, hat: str, topic_folder: str) -> list[ParsedDocument]:
        manifest_path = self.layout.hat(hat).summaries_dir / topic_folder / ".topic_manifest.json"
        if not manifest_path.exists():
            return []
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [ParsedDocument.from_dict(item) for item in payload.get("documents", [])]

    def _normalize_plan(self, plan: GroupingPlan, documents: list[ParsedDocument]) -> GroupingPlan:
        if plan.decision is GroupingDecision.SEGMENT:
            return plan
        decision = GroupingDecision.MERGE if len(documents) > 1 else GroupingDecision.STANDALONE
        return GroupingPlan(
            decision=decision,
            topic_folder=plan.topic_folder,
            estimated_length=sum(estimate_markdown_length(document.full_text) for document in documents),
            merge_target_topic=plan.merge_target_topic,
            related_chunk_ids=plan.related_chunk_ids,
            rationale=plan.rationale,
        )

    @staticmethod
    def _assign_topic_folder(chunks: list[ChunkRecord], topic_folder: str) -> list[ChunkRecord]:
        for chunk in chunks:
            chunk.metadata.topic_folder = topic_folder
        return chunks

    def _dense_indexer(self, hat: str, *, trace: bool = True) -> DenseIndexer:
        trace_sink = self.trace_collector if trace else None
        return DenseIndexer(
            self.embedder,
            LocalDenseIndex(self.layout.hat(hat).vector_index_dir),
            trace_sink=trace_sink,
        )

    def _lexical_indexer(self, hat: str) -> LexicalIndexer:
        return LexicalIndexer(LexicalIndex(self.layout.hat(hat).bm25_index_dir))

    def _emit_progress(self, message: str) -> None:
        if self.progress_sink is not None:
            self.progress_sink(message)

    def log_exception(
        self,
        *,
        component: str,
        task: str,
        exc: BaseException,
        context: dict[str, object] | None = None,
    ) -> Path:
        return self.exception_logger.log_exception(
            component=component,
            task=task,
            exc=exc,
            context=context,
        )


def synthesize_answer(question: str, hits: list[RetrievalHit], expanded_query: str | None = None) -> str:
    if not hits:
        return "No relevant local knowledge was found for that question."
    return _synthesize_answer(question, hits, expanded_query=expanded_query or question)


def _synthesize_answer(question: str, hits: list[RetrievalHit], expanded_query: str) -> str:
    key_points = _best_supporting_sentences(expanded_query, hits, limit=4)
    if not key_points:
        fallback = _truncate_text(hits[0].text, 220)
        return _clean_sentence(fallback)

    answer_sentences = _compose_answer_sentences(question, key_points)
    if not answer_sentences:
        return _clean_sentence(key_points[0])
    return " ".join(answer_sentences)


def _best_supporting_sentences(query: str, hits: list[RetrievalHit], limit: int = 4) -> list[str]:
    query_terms = set(tokenize(query))
    scored: list[tuple[float, str]] = []
    seen: set[str] = set()
    for hit in hits[:6]:
        rerank_score = float(hit.extras.get("rerank_score", hit.score))
        for sentence in _candidate_sentences(hit):
            normalized = " ".join(sentence.split()).strip()
            if len(normalized) < 24:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            overlap = len(query_terms & set(tokenize(normalized)))
            score = (overlap * 2.5) + rerank_score + float(hit.score)
            if hit.metadata.heading:
                score += 0.2
            scored.append((score, normalized))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [_clean_sentence(sentence) for _, sentence in scored[:limit]]


def _candidate_sentences(hit: RetrievalHit) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", hit.text) if part.strip()]
    if parts:
        return parts
    if hit.metadata.heading and hit.metadata.heading != hit.text:
        return [hit.metadata.heading.strip()]
    return []


def _compose_answer_sentences(question: str, sentences: list[str], max_sentences: int = 3) -> list[str]:
    chosen: list[str] = []
    seen_terms: set[str] = set()
    for sentence in sentences:
        normalized = _clean_sentence(sentence)
        if not normalized:
            continue
        sentence_terms = set(tokenize(normalized))
        if chosen and sentence_terms and sentence_terms <= seen_terms:
            continue
        chosen.append(normalized)
        seen_terms.update(sentence_terms)
        if len(chosen) >= max_sentences:
            break
    if not chosen:
        return []

    return chosen[:max_sentences]


def _clean_sentence(text: str) -> str:
    cleaned = " ".join(text.split()).strip()
    if cleaned.endswith(":"):
        cleaned = cleaned[:-1].rstrip()
    cleaned = re.sub(r"^[^A-Za-z0-9(]+", "", cleaned)
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _truncate_text(text: str, max_length: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


def format_citation(hit: RetrievalHit) -> str:
    metadata = hit.metadata
    hat = metadata.hat or "default"
    topic_folder = metadata.topic_folder or _fallback_topic_folder(metadata)
    filename = _citation_filename(metadata)
    location = _citation_location(metadata, filename)
    return f"{hat}/{topic_folder}/{filename}: {location}"


def _unique_citations(hits: list[RetrievalHit], limit: int) -> list[str]:
    citations: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        citation = format_citation(hit)
        if citation in seen:
            continue
        seen.add(citation)
        citations.append(citation)
        if len(citations) >= limit:
            break
    return citations


def _content_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    non_map_hits = [hit for hit in hits if not hit.metadata.is_map_context]
    return non_map_hits or hits


def _fallback_topic_folder(metadata) -> str:
    source_name = _citation_filename(metadata)
    stem = Path(source_name).stem
    return stem or "maps"


def _citation_filename(metadata) -> str:
    if metadata.source_path:
        return metadata.source_path.name
    source_uri = metadata.source_uri
    if not source_uri:
        return "unknown"
    if "://" in source_uri:
        parsed = urlparse(source_uri)
        return Path(parsed.path).name or parsed.netloc or source_uri
    return Path(source_uri).name or source_uri


def _citation_location(metadata, filename: str) -> str:
    locations: list[str] = []
    if metadata.page_number is not None:
        locations.append(f"Page {metadata.page_number}")
    heading = (metadata.heading or "").strip()
    section = (metadata.section or "").strip()
    for candidate in [heading, section]:
        if not candidate:
            continue
        if candidate in locations:
            continue
        if metadata.page_number is not None and candidate.lower() == f"page {metadata.page_number}".lower():
            continue
        locations.append(candidate)
    if locations:
        return ", ".join(locations)
    return Path(filename).stem or filename
