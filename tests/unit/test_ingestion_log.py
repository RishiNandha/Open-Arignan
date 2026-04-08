from __future__ import annotations

from pathlib import Path

from arignan.ingestion import IngestionLog
from arignan.models import LoadEvent, LoadOperation


def test_ingestion_log_appends_and_reads_events(tmp_path: Path) -> None:
    log = IngestionLog(tmp_path / "ingestion_log.jsonl")
    event = LoadEvent(
        load_id="load-1",
        operation=LoadOperation.INGEST,
        hat="default",
        created_at="2026-04-08T02:30:00Z",
        source_items=["notes.md"],
    )

    log.append(event)

    assert log.read_all() == [event]
    assert log.find_by_load_id("load-1") == [event]
