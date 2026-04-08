from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from arignan.models import LoadEvent, LoadOperation, ParsedDocument, SourceDocument

from .discovery import discover_sources
from .log import IngestionLog
from .parsers import DocumentParser, PdfOcrRequired, UrlFetcher


@dataclass(slots=True)
class IngestionBatch:
    load_id: str
    hat: str
    input_ref: str
    source_items: list[str]
    documents: list[ParsedDocument]
    failures: list["IngestionFailure"]


@dataclass(slots=True)
class IngestionFailure:
    source_uri: str
    message: str


ParseErrorHandler = Callable[[SourceDocument, BaseException], None]
ProgressHandler = Callable[[str], None]


def generate_load_id(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    return f"load-{current.strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"


class IngestionService:
    def __init__(self, ingestion_log: IngestionLog, url_fetcher: UrlFetcher | None = None) -> None:
        self.ingestion_log = ingestion_log
        self.parser = DocumentParser(url_fetcher=url_fetcher)

    def ingest(
        self,
        input_ref: str | Path,
        hat: str,
        load_id: str | None = None,
        log_event: bool = True,
        on_parse_error: ParseErrorHandler | None = None,
        on_progress: ProgressHandler | None = None,
    ) -> IngestionBatch:
        resolved_load_id = load_id or generate_load_id()
        sources = discover_sources(input_ref)
        if on_progress is not None:
            on_progress(f"Found {len(sources)} supported source(s) to load.")
        documents: list[ParsedDocument] = []
        failures: list[IngestionFailure] = []
        source_items: list[str] = []
        deferred_ocr_sources: list[tuple[int, SourceDocument]] = []
        for index, source in enumerate(sources, start=1):
            label = source.local_path.name if source.local_path is not None else source.source_uri
            if on_progress is not None:
                on_progress(f"[{index}/{len(sources)}] Parsing '{label}'...")
            try:
                parsed = self.parser.parse(
                    source,
                    load_id=resolved_load_id,
                    hat=hat,
                    allow_ocr=False,
                )
            except PdfOcrRequired:
                deferred_ocr_sources.append((index, source))
                if on_progress is not None:
                    on_progress(
                        f"[{index}/{len(sources)}] No embedded text found in '{label}'; deferring OCR until after other files."
                    )
                continue
            except Exception as exc:
                failures.append(IngestionFailure(source_uri=source.source_uri, message=str(exc)))
                if on_parse_error is not None:
                    on_parse_error(source, exc)
                continue
            documents.append(parsed)
            source_items.append(source.source_uri)
        if deferred_ocr_sources and on_progress is not None:
            on_progress(f"{len(deferred_ocr_sources)} source(s) need OCR and may take longer.")
        for ocr_index, (source_index, source) in enumerate(deferred_ocr_sources, start=1):
            label = source.local_path.name if source.local_path is not None else source.source_uri
            if on_progress is not None:
                on_progress(
                    f"[OCR {ocr_index}/{len(deferred_ocr_sources)} | source {source_index}/{len(sources)}] "
                    f"No embedded text found in '{label}'; trying OCR now, this may take a while..."
                )
            try:
                parsed = self.parser.parse(
                    source,
                    load_id=resolved_load_id,
                    hat=hat,
                    allow_ocr=True,
                )
            except Exception as exc:
                failures.append(IngestionFailure(source_uri=source.source_uri, message=str(exc)))
                if on_parse_error is not None:
                    on_parse_error(source, exc)
                continue
            documents.append(parsed)
            source_items.append(source.source_uri)
        batch = IngestionBatch(
            load_id=resolved_load_id,
            hat=hat,
            input_ref=str(input_ref),
            source_items=source_items,
            documents=documents,
            failures=failures,
        )
        if log_event and batch.source_items:
            self.ingestion_log.append(
                LoadEvent(
                    load_id=batch.load_id,
                    operation=LoadOperation.INGEST,
                    hat=hat,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    source_items=batch.source_items,
                    artifact_paths=[],
                    topic_folders=[],
                    metadata={"input_ref": batch.input_ref},
                )
            )
        return batch
