from __future__ import annotations

from pathlib import Path

from arignan.models import (
    ChatTurn,
    ChunkMetadata,
    ChunkRecord,
    DocumentSection,
    LoadEvent,
    LoadOperation,
    ParsedDocument,
    RetrievalHit,
    RetrievalSource,
    SessionState,
    SourceDocument,
    SourceType,
    TopicArtifact,
)


def test_parsed_document_round_trip() -> None:
    document = ParsedDocument(
        load_id="load-1",
        hat="default",
        source=SourceDocument(
            source_type=SourceType.MARKDOWN,
            source_uri="notes.md",
            local_path=Path("notes.md"),
            title="Notes",
            metadata={"author": "Ada"},
        ),
        full_text="Heading\n\nBody",
        sections=[DocumentSection(text="Body", heading="Heading")],
        keywords=["rag", "vector-db"],
    )

    restored = ParsedDocument.from_dict(document.to_dict())

    assert restored == document


def test_chunk_record_round_trip() -> None:
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        text="Context body",
        metadata=ChunkMetadata(
            load_id="load-1",
            hat="default",
            source_uri="paper.pdf",
            source_path=Path("paper.pdf"),
            page_number=8,
            section="RF Calibration",
            heading="Calibration",
            keywords=["calibre xrc"],
            topic_folder="rfic",
        ),
        embedding=[0.1, 0.2, 0.3],
    )

    restored = ChunkRecord.from_dict(chunk.to_dict())

    assert restored == chunk


def test_ingestion_and_artifact_round_trip() -> None:
    artifact = TopicArtifact(
        hat="default",
        topic_folder="jepa",
        source_paths=[Path("paper1.pdf"), Path("paper2.pdf")],
        markdown_paths=[Path("summary.md")],
        keywords=["jepa"],
    )
    event = LoadEvent(
        load_id="load-42",
        operation=LoadOperation.INGEST,
        hat="default",
        created_at="2026-04-08T02:15:00Z",
        source_items=["paper1.pdf", "paper2.pdf"],
        artifact_paths=[Path("hats/default/summaries/jepa/summary.md")],
        topic_folders=[artifact.topic_folder],
        metadata={"artifacts": [artifact.to_dict()]},
    )

    restored = LoadEvent.from_dict(event.to_dict())

    assert restored == event


def test_retrieval_hit_round_trip() -> None:
    hit = RetrievalHit(
        chunk_id="chunk-1",
        text="retrieved text",
        score=0.91,
        source=RetrievalSource.DENSE,
        metadata=ChunkMetadata(
            load_id="load-1",
            hat="default",
            source_uri="file.md",
        ),
        rank=1,
        extras={"rrf_score": 42.0},
    )

    restored = RetrievalHit.from_dict(hit.to_dict())

    assert restored == hit


def test_session_state_round_trip() -> None:
    session = SessionState(
        session_id="session-1",
        terminal_pid=100,
        hat="default",
        summary="Prior discussion summary",
        turns=[
            ChatTurn(role="user", content="What is JEPA?", timestamp="2026-04-08T02:00:00Z"),
            ChatTurn(role="assistant", content="It is a predictive architecture.", timestamp="2026-04-08T02:00:05Z"),
        ],
        metadata={"saved": True},
    )

    restored = SessionState.from_dict(session.to_dict())

    assert restored == session
