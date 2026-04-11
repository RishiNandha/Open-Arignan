from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal
from pathlib import Path
from urllib.parse import urlparse

from arignan.config import AppConfig
from arignan.grouping import (
    GroupingDecision,
    GroupingPlan,
    GroupingPlanner,
    derive_topic_folder,
    estimate_markdown_length,
)
from arignan.indexing import Chunker, DenseIndexer, LexicalIndex, LexicalIndexer, LocalDenseIndex, create_embedder, tokenize
from arignan.ingestion import IngestionFailure, IngestionLog, IngestionService
from arignan.llm import LocalTextGenerator, create_local_text_generator
from arignan.markdown import MarkdownRepository, derive_keywords
from arignan.markdown.writer import HeuristicArtifactWriter, LLMArtifactWriter
from arignan.models import ChunkRecord, LoadEvent, LoadOperation, ParsedDocument, RetrievalHit, SessionState, SourceDocument
from arignan.retrieval import RetrievalPipeline, create_reranker, describe_question
from arignan.session import SessionExceptionLogger, SessionManager, SessionModelCallLogger, SessionStore
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
class TopicGroupingRecord:
    topic_folder: str
    title: str
    locator: str
    description: str
    keywords: list[str]
    summary_excerpt: str
    source_count: int
    estimated_length: int
    current_load: bool


@dataclass(slots=True)
class GroupingRecommendation:
    members: list[str]
    target_topic_folder: str
    confidence: float
    rationale: str


@dataclass(slots=True)
class LoadResult:
    load_id: str
    hat: str
    document_count: int
    topic_folders: list[str]
    artifact_paths: list[Path]
    total_chunks: int
    total_markdown_segments: int
    failures: list[IngestionFailure]
    traces: list[LoadDocumentTrace]
    model_calls: list[ModelCallTrace]


@dataclass(slots=True)
class AskDebug:
    answer_mode: str
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
    answer_mode: str
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
        self.session_manager = SessionManager(SessionStore(config.app_home), config.session)
        self.exception_logger = SessionExceptionLogger(self.session_manager.store, self.terminal_pid)
        self.model_call_logger = SessionModelCallLogger(self.session_manager.store, self.terminal_pid)
        self.trace_collector = ModelTraceCollector(on_record=self.model_call_logger.log_call)
        self.embedder = create_embedder(
            config,
            progress_sink=self.progress_sink,
            exception_logger=self.exception_logger,
        )
        self.chunker = Chunker(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
        )
        self.reranker = create_reranker(
            config,
            progress_sink=self.progress_sink,
            exception_logger=self.exception_logger,
        )
        self.local_text_generator = create_local_text_generator(config, progress_sink=self.progress_sink)
        self.light_text_generator = create_local_text_generator(
            config,
            progress_sink=self.progress_sink,
            model_name=config.local_llm_light_model,
        )
        self.heuristic_artifact_writer = HeuristicArtifactWriter()
        self.provisional_markdown_repository = MarkdownRepository(artifact_writer=self.heuristic_artifact_writer)
        artifact_writer = LLMArtifactWriter(
            generator=self.local_text_generator,
            fallback=self.heuristic_artifact_writer,
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
        batch = self.ingestion_service.ingest(
            input_ref,
            hat=target_hat,
            log_event=False,
            on_parse_error=lambda source, exc: self._handle_load_parse_error(
                source=source,
                exc=exc,
                hat=target_hat,
            ),
            on_progress=self._emit_progress,
        )
        self._emit_progress(
            f"Discovered {len(batch.documents)} loadable document(s); failed sources: {len(batch.failures)}."
        )
        artifact_paths: list[Path] = []
        topic_folders: list[str] = []
        total_chunks = 0
        total_markdown_segments = 0
        traces: list[LoadDocumentTrace] = []

        for index, document in enumerate(batch.documents, start=1):
            label = document.source.title or Path(document.source.source_uri).name
            self._emit_progress(f"[{index}/{len(batch.documents)}] Planning provisional topic for '{label}'...")
            plan = self.grouping_planner.plan(
                document,
                related_hits=[],
                merge_candidates=[],
                llm_merge_hint=None,
            )
            documents_for_topic = [document]
            final_plan = self._normalize_plan(plan, documents_for_topic)

            self._emit_progress(f"[{index}/{len(batch.documents)}] Chunking and indexing '{label}'...")
            chunks = self.chunker.chunk_document(document)
            chunks = self._assign_topic_folder(chunks, final_plan.topic_folder)
            self._dense_indexer(target_hat).index_chunks(chunks)
            self._lexical_indexer(target_hat).index_chunks(chunks)
            total_chunks += len(chunks)

            self._emit_progress(f"[{index}/{len(batch.documents)}] Writing topic '{final_plan.topic_folder}'...")
            artifact = self.provisional_markdown_repository.write_topic(
                self.layout,
                hat=target_hat,
                documents=documents_for_topic,
                plan=final_plan,
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
                    grouping_decision=final_plan.decision.value,
                    chunk_count=len(chunks),
                    markdown_segment_count=len(artifact.markdown_paths),
                    rationale=list(final_plan.rationale),
                    segment_titles=[segment.title for segment in final_plan.segments],
                )
            )
            self._emit_progress(
                f"[{index}/{len(batch.documents)}] Finished loading '{Path(document.source.source_uri).name}' into topic '{artifact.topic_folder}'."
            )

        if batch.documents:
            self._emit_progress("Reviewing completed topic summaries for regrouping...")
            topic_folders, artifact_paths, total_markdown_segments, traces = self._post_load_regroup(
                hat=target_hat,
                load_id=batch.load_id,
                topic_folders=topic_folders,
                traces=traces,
            )
            self._emit_progress(f"Refreshing map.md for hat '{target_hat}'...")
            self.markdown_repository.update_hat_map(self.layout, target_hat)
            self._emit_progress("Refreshing global_map.md...")
            self.markdown_repository.update_global_map(self.layout)

        if batch.documents:
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
            failures=batch.failures,
            traces=traces,
            model_calls=self.trace_collector.snapshot(),
        )

    def ask(
        self,
        question: str,
        hat: str = "auto",
        terminal_pid: int | None = None,
        answer_mode: Literal["default", "light", "none", "raw"] = "default",
    ) -> AskResult:
        self.trace_collector.clear()
        pid = terminal_pid or self.terminal_pid
        session = self.session_manager.append_turn(pid, role="user", content=question)
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
        rerank_min_score = 0.05 if getattr(self.reranker, "backend_name", "") == "heuristic-reranker" else float("-inf")
        reranked = self.reranker.rerank(
            bundle.expanded_query,
            bundle.fused_hits,
            limit=self.config.retrieval.rerank_top_k,
            min_score=rerank_min_score,
        )
        answer_hits = _content_hits(reranked)
        if not answer_hits:
            fallback_hits = _content_hits(bundle.fused_hits)
            if fallback_hits:
                self._emit_progress("Reranker found no strong hits; using top retrieved context instead...")
                answer_hits = fallback_hits
        answer, citations = compose_answer(
            question,
            answer_hits,
            answer_mode=answer_mode,
            context_limit=self._answer_context_limit(answer_mode),
            expanded_query=bundle.expanded_query,
            selected_hat=bundle.selected_hat,
            default_generator=self.local_text_generator,
            light_generator=self.light_text_generator,
            trace_sink=self.trace_collector,
            exception_logger=self.exception_logger,
            progress_sink=self.progress_sink,
            session=session,
        )
        self.session_manager.append_turn(pid, role="assistant", content=answer)
        debug = AskDebug(
            answer_mode=answer_mode,
            expanded_query=bundle.expanded_query,
            selected_hat=bundle.selected_hat,
            dense_hits=bundle.dense_hits,
            lexical_hits=bundle.lexical_hits,
            map_hits=bundle.map_hits,
            fused_hits=bundle.fused_hits,
            reranked_hits=reranked,
            model_calls=self.trace_collector.snapshot(),
        )
        return AskResult(
            question=question,
            selected_hat=bundle.selected_hat,
            answer_mode=answer_mode,
            answer=answer,
            citations=citations,
            debug=debug,
        )

    def list_events(self) -> list[LoadEvent]:
        return self.ingestion_log.read_all()

    def list_ingestions(self) -> list[LoadEvent]:
        return [event for event in self.ingestion_log.read_all() if event.operation is LoadOperation.INGEST]

    def list_live_ingestions(self) -> list[LoadEvent]:
        live_events: list[LoadEvent] = []
        for event in self.list_ingestions():
            if self._ingestion_event_exists(event):
                live_events.append(event)
        return live_events

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
            hat_layout = self.layout.hat(hat)
            if not hat_layout.root.exists():
                deleted_topics.extend(
                    topic_folder
                    for load_id in to_delete
                    if ingest_events[load_id].hat == hat
                    for topic_folder in ingest_events[load_id].topic_folders
                )
                continue
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
        keywords = derive_keywords([document], limit=6)
        headings = [
            section.heading.strip()
            for section in document.sections
            if section.heading and not re.fullmatch(r"page\s+\d+", section.heading.strip(), flags=re.IGNORECASE)
        ][:6]
        query = " ".join(
            part
            for part in [
                document.source.title or "",
                " ".join(keywords),
                " ".join(headings),
                document.full_text[:500],
            ]
            if part.strip()
        ).strip()
        dense = self._dense_indexer(document.hat)
        lexical = self._lexical_indexer(document.hat)
        return dense.search(query, 5) + lexical.search(query, 5)

    def _existing_topic_documents(self, hat: str, topic_folder: str) -> list[ParsedDocument]:
        manifest_path = self.layout.hat(hat).summaries_dir / topic_folder / ".topic_manifest.json"
        if not manifest_path.exists():
            return []
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [ParsedDocument.from_dict(item) for item in payload.get("documents", [])]

    def _collect_grouping_topics(self, hat: str, load_id: str) -> list[TopicGroupingRecord]:
        topics: list[TopicGroupingRecord] = []
        for manifest_path in sorted(self.layout.hat(hat).summaries_dir.glob("*/.topic_manifest.json")):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            documents = [ParsedDocument.from_dict(item) for item in payload.get("documents", [])]
            title = str(payload.get("title") or payload["topic_folder"].replace("-", " ")).strip()
            locator = str(payload.get("locator") or "").strip()
            description = str(payload.get("description") or "").strip()
            keywords = [str(item).strip() for item in payload.get("keywords", []) if str(item).strip()]
            topic_dir = manifest_path.parent
            summary_excerpt = _read_topic_summary_excerpt(topic_dir / "summary.md")
            if not summary_excerpt:
                summary_excerpt = _read_topic_summary_excerpt(topic_dir / "topic_index.md")
            estimated_length = sum(estimate_markdown_length(document.full_text) for document in documents)
            topics.append(
                TopicGroupingRecord(
                    topic_folder=payload["topic_folder"],
                    title=title,
                    locator=locator,
                    description=description,
                    keywords=keywords,
                    summary_excerpt=summary_excerpt,
                    source_count=len(documents),
                    estimated_length=estimated_length,
                    current_load=any(document.load_id == load_id for document in documents),
                )
            )
        return topics

    def _grouping_recommendations(
        self,
        hat: str,
        load_id: str,
        topics: list[TopicGroupingRecord],
    ) -> list[GroupingRecommendation]:
        if len(topics) < 2:
            return []
        if not any(topic.current_load for topic in topics):
            return []

        self._emit_progress(
            f"Reviewing {len(topics)} topic summaries in hat '{hat}' with the local LLM for possible groups..."
        )
        prompt = _build_grouping_review_prompt(hat, topics)
        try:
            raw = self.local_text_generator.generate(
                system_prompt=GROUPING_REVIEW_SYSTEM_PROMPT,
                user_prompt=prompt,
                max_new_tokens=320,
                temperature=0.0,
                response_format=GROUPING_REVIEW_RESPONSE_FORMAT,
            )
            recommendations = _parse_grouping_review(raw, topics)
        except Exception as exc:
            log_path = self.log_exception(
                component="llm",
                task="batch grouping review",
                exc=exc,
                context={"hat": hat, "topic_count": len(topics), "load_id": load_id},
            )
            self.trace_collector.record(
                component="llm",
                task="batch grouping review",
                model_name=self.local_text_generator.model_name,
                backend=self.local_text_generator.backend_name,
                status="fallback",
                item_count=len(topics),
                detail=f"{hat} | exception | {log_path.resolve()}",
            )
            return []

        if recommendations:
            preview = "; ".join(
                f"{', '.join(item.members)} -> {item.target_topic_folder} ({item.confidence:.2f})"
                for item in recommendations[:3]
            )
            self._emit_progress(f"Batch grouping review suggested {len(recommendations)} merge candidate(s): {preview}")
            detail = f"{hat} | {preview}"
        else:
            self._emit_progress("Batch grouping review did not suggest any merge candidates.")
            detail = f"{hat} | 0 recommendation(s)"
        self.trace_collector.record(
            component="llm",
            task="batch grouping review",
            model_name=self.local_text_generator.model_name,
            backend=self.local_text_generator.backend_name,
            status="ok" if recommendations else "fallback",
            item_count=len(topics),
            detail=detail,
        )
        return recommendations

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

    def _post_load_regroup(
        self,
        *,
        hat: str,
        load_id: str,
        topic_folders: list[str],
        traces: list[LoadDocumentTrace],
    ) -> tuple[list[str], list[Path], int, list[LoadDocumentTrace]]:
        if not topic_folders:
            return topic_folders, [], 0, traces

        trace_by_source = {trace.source_uri: trace for trace in traces}
        topics = self._collect_grouping_topics(hat, load_id)
        recommendations = self._grouping_recommendations(hat, load_id, topics)
        applied_topics: set[str] = set()
        topics_by_folder = {topic.topic_folder: topic for topic in topics}

        for recommendation in recommendations:
            member_topics = [topic for topic in recommendation.members if topic in topics_by_folder]
            if len(member_topics) < 2:
                continue
            if recommendation.target_topic_folder not in member_topics:
                continue
            if any(topic in applied_topics for topic in member_topics):
                continue
            if recommendation.confidence < GROUPING_REVIEW_MIN_CONFIDENCE:
                continue
            if not any(topics_by_folder[topic].current_load for topic in member_topics):
                continue

            manifests = [self._read_topic_manifest(hat, topic) for topic in member_topics]
            if any(item is None for item in manifests):
                continue
            payloads_and_docs = [item for item in manifests if item is not None]
            if any(payload.get("decision") == GroupingDecision.SEGMENT.value for payload, _ in payloads_and_docs):
                continue

            merged_documents = _merge_documents(
                [],
                [document for _, documents in payloads_and_docs for document in documents],
            )
            estimated_length = sum(estimate_markdown_length(document.full_text) for document in merged_documents)
            if estimated_length > self.config.markdown.max_md_length:
                continue

            final_plan = GroupingPlan(
                decision=GroupingDecision.MERGE,
                topic_folder=recommendation.target_topic_folder,
                estimated_length=estimated_length,
                merge_target_topic=recommendation.target_topic_folder,
                rationale=[
                    (
                        f"Post-load grouping review merged {', '.join(sorted(member_topics))} into "
                        f"'{recommendation.target_topic_folder}' ({recommendation.confidence:.2f})."
                    ),
                    recommendation.rationale,
                ],
            )
            self._emit_progress(
                "Applying grouped merge "
                f"{', '.join(member_topics)} -> '{recommendation.target_topic_folder}' "
                f"(confidence {recommendation.confidence:.2f})..."
            )
            for document in merged_documents:
                if document.load_id != load_id:
                    continue
                trace = trace_by_source.get(document.source.source_uri)
                if trace is None:
                    continue
                trace.topic_folder = recommendation.target_topic_folder
                trace.grouping_decision = GroupingDecision.MERGE.value
                trace.rationale = list(dict.fromkeys([*trace.rationale, *final_plan.rationale]))

            self.provisional_markdown_repository.regenerate_topic(
                self.layout,
                hat=hat,
                documents=merged_documents,
                plan=final_plan,
                refresh_maps=False,
            )
            for topic_folder in member_topics:
                if topic_folder == recommendation.target_topic_folder:
                    continue
                current_topic_dir = self.layout.hat(hat).summaries_dir / topic_folder
                if current_topic_dir.exists():
                    shutil.rmtree(current_topic_dir)
            applied_topics.update(member_topics)

        self._emit_progress("Finalizing wiki summaries for current load...")
        self._finalize_load_topics_with_llm(hat, load_id)
        self._reindex_load_topics(hat, load_id)
        final_topic_folders, artifact_paths, total_markdown_segments = self._final_load_artifacts(hat, load_id, traces)
        return final_topic_folders, artifact_paths, total_markdown_segments, traces

    def _final_load_artifacts(
        self,
        hat: str,
        load_id: str,
        traces: list[LoadDocumentTrace],
    ) -> tuple[list[str], list[Path], int]:
        trace_by_source = {trace.source_uri: trace for trace in traces}
        topic_folders: list[str] = []
        artifact_paths: list[Path] = []
        total_markdown_segments = 0
        for manifest_path in sorted(self.layout.hat(hat).summaries_dir.glob("*/.topic_manifest.json")):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            documents = [ParsedDocument.from_dict(item) for item in payload.get("documents", [])]
            selected = [document for document in documents if document.load_id == load_id]
            if not selected:
                continue
            topic_folder = payload["topic_folder"]
            if topic_folder not in topic_folders:
                topic_folders.append(topic_folder)
            markdown_paths = [Path(path) for path in payload.get("markdown_paths", [])]
            artifact_paths.extend(markdown_paths)
            total_markdown_segments += len(markdown_paths)
            for document in selected:
                trace = trace_by_source.get(document.source.source_uri)
                if trace is None:
                    continue
                trace.topic_folder = topic_folder
                trace.markdown_segment_count = len(markdown_paths)
        return topic_folders, artifact_paths, total_markdown_segments

    def _read_topic_manifest(self, hat: str, topic_folder: str) -> tuple[dict[str, object], list[ParsedDocument]] | None:
        manifest_path = self.layout.hat(hat).summaries_dir / topic_folder / ".topic_manifest.json"
        if not manifest_path.exists():
            return None
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        documents = [ParsedDocument.from_dict(item) for item in payload.get("documents", [])]
        return payload, documents

    def _reindex_load_topics(self, hat: str, load_id: str) -> None:
        dense = self._dense_indexer(hat, trace=False)
        lexical = self._lexical_indexer(hat)
        dense.delete_load(load_id)
        lexical.delete_load(load_id)
        chunks: list[ChunkRecord] = []
        for manifest_path in sorted(self.layout.hat(hat).summaries_dir.glob("*/.topic_manifest.json")):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            topic_folder = payload["topic_folder"]
            documents = [ParsedDocument.from_dict(item) for item in payload.get("documents", [])]
            for document in documents:
                if document.load_id != load_id:
                    continue
                document_chunks = self._assign_topic_folder(self.chunker.chunk_document(document), topic_folder)
                chunks.extend(document_chunks)
        if chunks:
            dense.index_chunks(chunks)
            lexical.index_chunks(chunks)

    def _finalize_load_topics_with_llm(self, hat: str, load_id: str) -> None:
        for manifest_path in sorted(self.layout.hat(hat).summaries_dir.glob("*/.topic_manifest.json")):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            documents = [ParsedDocument.from_dict(item) for item in payload.get("documents", [])]
            if not any(document.load_id == load_id for document in documents):
                continue
            decision_value = str(payload.get("decision") or GroupingDecision.STANDALONE.value)
            try:
                decision = GroupingDecision(decision_value)
            except ValueError:
                decision = GroupingDecision.STANDALONE
            plan = GroupingPlan(
                decision=decision,
                topic_folder=payload["topic_folder"],
                estimated_length=int(payload.get("estimated_length") or 0),
                merge_target_topic=payload.get("merge_target_topic"),
                related_chunk_ids=list(payload.get("related_chunk_ids", [])),
                rationale=list(payload.get("rationale", [])),
            )
            self.markdown_repository.regenerate_topic(
                self.layout,
                hat=hat,
                documents=documents,
                plan=plan,
                refresh_maps=False,
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

    def _answer_context_limit(self, answer_mode: Literal["default", "light", "none", "raw"]) -> int:
        retrieval = self.config.retrieval
        if answer_mode == "light":
            return retrieval.answer_context_top_k_light
        if answer_mode == "none":
            return retrieval.answer_context_top_k_none
        if answer_mode == "raw":
            return retrieval.answer_context_top_k_raw
        return retrieval.answer_context_top_k_default

    def _handle_load_parse_error(
        self,
        *,
        source: SourceDocument,
        exc: BaseException,
        hat: str,
    ) -> None:
        label = source.local_path.name if source.local_path is not None else source.source_uri
        log_path = self.log_exception(
            component="ingestion",
            task="parse source during load",
            exc=exc,
            context={
                "hat": hat,
                "source_uri": source.source_uri,
                "source_type": source.source_type.value,
            },
        )
        self._emit_progress(
            f"Failed to parse '{label}'; continuing with remaining sources. Log: {log_path.resolve()}"
        )

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

    def format_logged_exception_message(
        self,
        *,
        component: str,
        task: str,
        exc: BaseException,
        context: dict[str, object] | None = None,
        user_message: str = "Something went wrong.",
    ) -> str:
        log_path = self.log_exception(component=component, task=task, exc=exc, context=context)
        return f"{user_message} See {log_path.resolve()}"

    def _ingestion_event_exists(self, event: LoadEvent) -> bool:
        hat_layout = self.layout.hat(event.hat)
        if not hat_layout.root.exists():
            return False
        if not event.topic_folders:
            return True
        for topic_folder in event.topic_folders:
            if (hat_layout.summaries_dir / topic_folder / ".topic_manifest.json").exists():
                return True
        return False


def synthesize_answer(question: str, hits: list[RetrievalHit], expanded_query: str | None = None) -> str:
    if not hits:
        return "No relevant local knowledge was found for that question."
    return _synthesize_answer(question, hits, expanded_query=expanded_query or question)


def compose_answer(
    question: str,
    hits: list[RetrievalHit],
    *,
    answer_mode: Literal["default", "light", "none", "raw"],
    context_limit: int,
    expanded_query: str,
    selected_hat: str,
    default_generator: LocalTextGenerator,
    light_generator: LocalTextGenerator,
    trace_sink: ModelTraceCollector | None = None,
    exception_logger: SessionExceptionLogger | None = None,
    progress_sink: Callable[[str], None] | None = None,
    session: SessionState | None = None,
) -> tuple[str, list[str]]:
    if answer_mode == "raw":
        if progress_sink is not None:
            progress_sink("Composing raw retrieval output...")
        return render_raw_hits(hits, limit=context_limit), []
    if answer_mode == "none":
        if progress_sink is not None:
            progress_sink("Composing retrieval synthesis answer...")
        return synthesize_answer(question, hits, expanded_query=expanded_query), _unique_citations(hits, limit=3)

    generator = default_generator if answer_mode == "default" else light_generator
    answer = generate_answer(
        question,
        hits,
        context_limit=context_limit,
        expanded_query=expanded_query,
        selected_hat=selected_hat,
        generator=generator,
        max_new_tokens=480 if answer_mode == "default" else 320,
        trace_sink=trace_sink,
        exception_logger=exception_logger,
        progress_sink=progress_sink,
        session=session,
    )
    return answer, _unique_citations(hits, limit=3)


ANSWER_SYSTEM_PROMPT = """You answer questions for a private local knowledge base.
Use only the provided retrieved context and prior session context.
If the retrieved context is insufficient, say that the local knowledge base does not contain enough information.
Write a direct, technically accurate answer in concise natural language.
Do not mention retrieval, chunks, prompts, or hidden system behavior.
Do not add a citations, sources, or references section; citations are handled separately outside your answer.
Follow this answering discipline:
- Answer the question in the first sentence whenever possible.
- Prefer a short explanatory paragraph, optionally followed by a second short paragraph for nuance.
- If the context supports a definition or expansion, state it directly before elaborating.
- If multiple retrieved passages disagree, briefly describe the uncertainty instead of guessing."""


def generate_answer(
    question: str,
    hits: list[RetrievalHit],
    *,
    context_limit: int,
    expanded_query: str,
    selected_hat: str,
    generator: LocalTextGenerator,
    max_new_tokens: int = 480,
    trace_sink: ModelTraceCollector | None = None,
    exception_logger: SessionExceptionLogger | None = None,
    progress_sink: Callable[[str], None] | None = None,
    session: SessionState | None = None,
) -> str:
    if not hits:
        _record_answer_trace(
            trace_sink,
            generator=generator,
            status="skipped",
            item_count=0,
            detail=f"{selected_hat} (no context)",
        )
        return "No relevant local knowledge was found for that question."

    if progress_sink is not None:
        progress_sink(f"Hitting local LLM for answer generation ({generator.model_name})...")
    prompt = _build_answer_prompt(
        question,
        hits,
        context_limit=context_limit,
        expanded_query=expanded_query,
        selected_hat=selected_hat,
        session=session,
    )
    try:
        raw = generator.generate(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
        )
        answer = _normalize_generated_answer(raw)
    except Exception as exc:
        log_path = _log_answer_exception(
            exception_logger,
            exc=exc,
            selected_hat=selected_hat,
            hit_count=len(hits),
        )
        if progress_sink is not None:
            progress_sink(_answer_fallback_message(log_path))
        _record_answer_trace(
            trace_sink,
            generator=generator,
            status="fallback",
            item_count=len(hits),
            detail=f"{selected_hat} (exception)",
        )
        return synthesize_answer(question, hits, expanded_query=expanded_query)

    if not answer:
        _record_answer_trace(
            trace_sink,
            generator=generator,
            status="fallback",
            item_count=len(hits),
            detail=f"{selected_hat} (empty output)",
        )
        return synthesize_answer(question, hits, expanded_query=expanded_query)

    _record_answer_trace(
        trace_sink,
        generator=generator,
        status="ok",
        item_count=len(hits),
        detail=selected_hat,
    )
    return answer


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


def render_raw_hits(hits: list[RetrievalHit], limit: int = 8) -> str:
    if not hits:
        return "No relevant local knowledge was found for that question."
    lines = ["Top retrieved context:"]
    for index, hit in enumerate(hits[:limit], start=1):
        score = float(hit.extras.get("rerank_score", hit.extras.get("rrf_score", hit.score)))
        snippet = _truncate_text(hit.text, 520)
        lines.append(f"{index}. [{score:.3f}] {format_citation(hit)}")
        lines.append(f"   {snippet}")
        if index < min(len(hits), limit):
            lines.append("")
    return "\n".join(lines)


def _build_answer_prompt(
    question: str,
    hits: list[RetrievalHit],
    *,
    context_limit: int,
    expanded_query: str,
    selected_hat: str,
    session: SessionState | None,
) -> str:
    question_intent, focus_topic, answer_brief = describe_question(question)
    lines = [
        "<task>",
        "Answer the user's question using only the retrieved context.",
        "Produce a concise, well-structured final answer for an end user.",
        "</task>",
        "",
        "<answer_requirements>",
        "- First sentence should answer the question directly when possible.",
        "- Organize the answer into 1 to 2 short paragraphs.",
        "- Prefer synthesis over quotation.",
        "- Use the strongest evidence first.",
        "- If the answer is incomplete from the context, say so plainly.",
        "- Do not mention sources, citations, retrieval, passages, or the prompt.",
        "</answer_requirements>",
        "",
        "<query>",
        f"Hat: {selected_hat}",
        f"Question: {question}",
        f"Expanded query: {expanded_query}",
        "</query>",
        "",
        "<question_brief>",
        f"Intent: {question_intent}",
        f"Focus topic: {focus_topic}",
        f"Preferred answer shape: {answer_brief}",
        "</question_brief>",
        "",
        "<example>",
        "Question: What does TTFS stand for?",
        "Retrieved context: TTFS is short for Time To First Spike and is used in spiking neural network discussions.",
        "Good answer: TTFS stands for Time To First Spike. In this context it refers to a spiking-neural-network timing scheme that represents information through the latency of the first spike.",
        "</example>",
    ]

    summary = (session.summary or "").strip() if session is not None else ""
    if summary:
        lines.extend(
            [
                "",
                "<session_summary>",
                summary,
                "</session_summary>",
            ]
        )

    recent_turns = _recent_turns_for_prompt(session, question=question)
    if recent_turns:
        lines.append("")
        lines.append("<recent_dialogue>")
        for turn in recent_turns:
            role = "User" if turn.role.lower() == "user" else "Assistant"
            lines.append(f"{role}: {turn.content.strip()}")
        lines.append("</recent_dialogue>")

    lines.extend(
        [
            "",
            "<retrieved_passages>",
        ]
    )
    for index, hit in enumerate(hits[:context_limit], start=1):
        score = hit.extras.get("rerank_score", hit.extras.get("rrf_score", hit.score))
        lines.extend(
            [
                f"<passage rank=\"{index}\" score=\"{float(score):.3f}\" citation=\"{format_citation(hit)}\">",
                _truncate_text(hit.text, 1200),
                "</passage>",
            ]
        )
    lines.extend(
        [
            "</retrieved_passages>",
            "",
            "<final_instruction>",
            "Write only the final answer for the user.",
            "</final_instruction>",
        ]
    )
    return "\n".join(lines).rstrip()


def _recent_turns_for_prompt(session: SessionState | None, *, question: str) -> list:
    if session is None or not session.turns:
        return []
    turns = list(session.turns)
    if turns and turns[-1].role.lower() == "user" and turns[-1].content == question:
        turns = turns[:-1]
    return turns[-4:]


def _normalize_generated_answer(text: str) -> str:
    normalized = text.strip()
    fenced = re.match(r"```(?:markdown|text)?\s*(.*?)\s*```$", normalized, re.DOTALL)
    if fenced:
        normalized = fenced.group(1).strip()
    normalized = re.sub(r"^(?:answer|response)\s*:\s*", "", normalized, flags=re.IGNORECASE)
    cleaned_lines: list[str] = []
    for raw_line in normalized.splitlines():
        stripped = raw_line.strip()
        if re.match(r"^(?:citations?|sources?|references?)\s*:?$", stripped, flags=re.IGNORECASE):
            break
        if re.match(r"^(?:citations?|sources?|references?)\s*:", stripped, flags=re.IGNORECASE):
            break
        cleaned_lines.append(raw_line.rstrip())
    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _record_answer_trace(
    trace_sink: ModelTraceCollector | None,
    *,
    generator: LocalTextGenerator,
    status: str,
    item_count: int | None,
    detail: str | None,
) -> None:
    if trace_sink is None:
        return
    trace_sink.record(
        component="llm",
        task="answer generation",
        model_name=getattr(generator, "model_name", type(generator).__name__),
        backend=getattr(generator, "backend_name", type(generator).__name__),
        status=status,
        item_count=item_count,
        detail=detail,
    )


def _log_answer_exception(
    exception_logger: SessionExceptionLogger | None,
    *,
    exc: BaseException,
    selected_hat: str,
    hit_count: int,
) -> Path | None:
    if exception_logger is None:
        return None
    return exception_logger.log_exception(
        component="llm",
        task="answer generation",
        exc=exc,
        context={"hat": selected_hat, "hit_count": hit_count},
    )


def _answer_fallback_message(log_path: Path | None) -> str:
    message = "Local LLM unavailable for answer generation; using retrieval synthesis fallback."
    if log_path is None:
        return message
    return f"{message} Log: {log_path.resolve()}"


GROUPING_REVIEW_MIN_CONFIDENCE = 0.68


GROUPING_REVIEW_SYSTEM_PROMPT = """You review topic pages inside one local research-wiki hat.
Each topic already has a compiled wiki-style summary.
Your task is to suggest topic groups only when the topics clearly belong in one shared wiki page.
Focus on topics marked as part of the current load, but you may merge them into older topics in the same hat.
Be selective, but not timid:
- Do not merge just because topics are from the same broad field.
- Prefer merge when topics are different notes, papers, or subtopics around the same named method family, embedding family, algorithm, training objective, or conceptual cluster.
- Prefer merge when the topics share concrete terms such as the same model name, method name, objective, or technical vocabulary.
- Prefer merge when one topic is clearly a subtopic, implementation note, training note, or explanatory note for another topic.
- Skip merges that would make the target page unfocused.
Return strict JSON only with this shape:
{
  "recommendations": [
    {
      "members": ["topic-a", "topic-b"],
      "target_topic_folder": "topic-b",
      "confidence": 0.0,
      "rationale": "short reason"
    }
  ]
}
If no merge is warranted, return {"recommendations": []}."""


GROUPING_REVIEW_RESPONSE_FORMAT = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "members": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                    },
                    "target_topic_folder": {"type": "string"},
                    "confidence": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["members", "target_topic_folder", "confidence", "rationale"],
            },
        }
    },
    "required": ["recommendations"],
}


def _build_grouping_review_prompt(hat: str, topics: list[TopicGroupingRecord]) -> str:
    lines = [
        f"Review the topic list for hat '{hat}'.",
        "Suggest groups only when multiple topics should become one shared wiki page.",
        "Return high-signal merge recommendations with confidence scores.",
        "",
        "<topic_list>",
    ]
    for index, topic in enumerate(topics, start=1):
        lines.extend(
            [
                f"<topic rank=\"{index}\" topic_folder=\"{topic.topic_folder}\">",
                f"Current load: {'yes' if topic.current_load else 'no'}",
                f"Title: {topic.title or topic.topic_folder}",
                f"Locator: {topic.locator or 'n/a'}",
                f"Description: {topic.description or 'n/a'}",
                f"Keywords: {', '.join(topic.keywords) or 'none'}",
                f"Source count: {topic.source_count}",
                f"Estimated length: {topic.estimated_length}",
                f"Summary excerpt: {topic.summary_excerpt or 'n/a'}",
                "</topic>",
            ]
        )
    lines.extend(
        [
            "</topic_list>",
            "",
            "<pair_hints>",
        ]
    )
    pair_hints = _candidate_group_hints(topics)
    if pair_hints:
        lines.extend(pair_hints)
    else:
        lines.append("No strong overlap hints detected from titles and keywords.")
    lines.extend(
        [
            "</pair_hints>",
            "",
            "Only recommend merges that would improve the wiki as a cleaner, richer lookup surface.",
        ]
    )
    return "\n".join(lines)


def _parse_grouping_review(raw: str, topics: list[TopicGroupingRecord]) -> list[GroupingRecommendation]:
    normalized = raw.strip()
    fenced = re.match(r"```(?:json)?\s*(.*?)\s*```$", normalized, re.DOTALL)
    if fenced:
        normalized = fenced.group(1).strip()
    payload = json.loads(normalized)
    valid_topics = {topic.topic_folder for topic in topics}
    recommendations: list[GroupingRecommendation] = []
    for item in payload.get("recommendations", []):
        if not isinstance(item, dict):
            continue
        members = [str(member).strip() for member in item.get("members", []) if str(member).strip()]
        members = [member for member in members if member in valid_topics]
        members = list(dict.fromkeys(members))
        target_topic_folder = str(item.get("target_topic_folder", "")).strip()
        if len(members) < 2 or target_topic_folder not in members:
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        rationale = str(item.get("rationale", "")).strip() or "No rationale provided."
        recommendations.append(
            GroupingRecommendation(
                members=members,
                target_topic_folder=target_topic_folder,
                confidence=confidence,
                rationale=rationale,
            )
        )
    recommendations.sort(key=lambda item: item.confidence, reverse=True)
    return recommendations


def _read_topic_summary_excerpt(summary_path: Path) -> str:
    if not summary_path.exists():
        return ""
    try:
        raw = summary_path.read_text(encoding="utf-8")
    except OSError:
        return ""


GROUPING_HINT_STOPWORDS = {
    "a",
    "an",
    "and",
    "architecture",
    "approach",
    "based",
    "for",
    "from",
    "ideas",
    "implementation",
    "in",
    "intro",
    "introduction",
    "learning",
    "method",
    "methods",
    "model",
    "models",
    "note",
    "notes",
    "of",
    "on",
    "overview",
    "paper",
    "representation",
    "research",
    "the",
    "to",
    "training",
}


def _candidate_group_hints(topics: list[TopicGroupingRecord], limit: int = 8) -> list[str]:
    scored: list[tuple[int, str]] = []
    for index, left in enumerate(topics):
        left_terms = _grouping_terms(left)
        for right in topics[index + 1 :]:
            if not (left.current_load or right.current_load):
                continue
            right_terms = _grouping_terms(right)
            shared = sorted(left_terms & right_terms)
            if len(shared) < 2:
                continue
            score = len(shared) + (1 if left.current_load and right.current_load else 0)
            scored.append(
                (
                    score,
                    f"{left.topic_folder} <-> {right.topic_folder}: shared terms {', '.join(shared[:6])}",
                )
            )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [hint for _, hint in scored[:limit]]


def _grouping_terms(topic: TopicGroupingRecord) -> set[str]:
    text = " ".join(
        part
        for part in [
            topic.topic_folder.replace("-", " "),
            topic.title,
            topic.locator,
            topic.description,
            " ".join(topic.keywords),
        ]
        if part
    )
    return {
        token
        for token in tokenize(text)
        if len(token) > 2 and token not in GROUPING_HINT_STOPWORDS
    }
    return _truncate_text(_flatten_markdown_for_grouping(raw), 700)


def _flatten_markdown_for_grouping(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return ""
    normalized = re.sub(r"^#{1,6}\s*", "", normalized, flags=re.MULTILINE)
    normalized = re.sub(r"^\|\s*---.*$", "", normalized, flags=re.MULTILINE)
    normalized = re.sub(r"^\|", "", normalized, flags=re.MULTILINE)
    normalized = re.sub(r"\|", " ", normalized)
    normalized = re.sub(r"^\s*-\s+", "", normalized, flags=re.MULTILINE)
    normalized = re.sub(r"`", "", normalized)
    return " ".join(normalized.split())


def _merge_documents(existing: list[ParsedDocument], incoming: list[ParsedDocument]) -> list[ParsedDocument]:
    merged: list[ParsedDocument] = []
    seen: set[tuple[str, str]] = set()
    for document in [*existing, *incoming]:
        key = (document.load_id, document.source.source_uri)
        if key in seen:
            continue
        seen.add(key)
        merged.append(document)
    return merged


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
