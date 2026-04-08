from __future__ import annotations

from pathlib import Path

from arignan.grouping import GroupingDecision, GroupingPlanner
from arignan.models import ChunkMetadata, DocumentSection, ParsedDocument, RetrievalHit, RetrievalSource, SourceDocument, SourceType


def test_grouping_planner_returns_standalone_for_small_unrelated_doc() -> None:
    document = ParsedDocument(
        load_id="load-1",
        hat="default",
        source=SourceDocument(
            source_type=SourceType.MARKDOWN,
            source_uri="jepa-notes.md",
            local_path=Path("jepa-notes.md"),
            title="JEPA Notes",
        ),
        full_text="Short notes on JEPA.",
        sections=[DocumentSection(text="Short notes on JEPA.", heading="JEPA Notes")],
    )

    plan = GroupingPlanner(max_md_length=1000).plan(document, related_hits=[])

    assert plan.decision is GroupingDecision.STANDALONE
    assert plan.topic_folder == "jepa-notes"


def test_grouping_planner_merges_with_existing_topic_when_related_and_small() -> None:
    document = ParsedDocument(
        load_id="load-2",
        hat="default",
        source=SourceDocument(source_type=SourceType.MARKDOWN, source_uri="new-jepa.md", title="New JEPA"),
        full_text="A concise note about joint embedding predictive architectures.",
        sections=[DocumentSection(text="A concise note about joint embedding predictive architectures.", heading="New JEPA")],
    )
    related_hit = RetrievalHit(
        chunk_id="chunk-a",
        text="Existing JEPA summary chunk",
        score=0.92,
        source=RetrievalSource.DENSE,
        metadata=ChunkMetadata(
            load_id="load-existing",
            hat="default",
            source_uri="existing.md",
            topic_folder="jepa",
        ),
        extras={"topic_length_estimate": 300},
    )

    plan = GroupingPlanner(max_md_length=1000).plan(document, related_hits=[related_hit])

    assert plan.decision is GroupingDecision.MERGE
    assert plan.merge_target_topic == "jepa"
    assert plan.related_chunk_ids == ["chunk-a"]


def test_grouping_planner_segments_large_book_like_document() -> None:
    sections = [
        DocumentSection(text="content " * 200, heading=f"Chapter {index}")
        for index in range(1, 10)
    ]
    document = ParsedDocument(
        load_id="load-3",
        hat="default",
        source=SourceDocument(source_type=SourceType.PDF, source_uri="book.pdf", local_path=Path("book.pdf"), title="RFIC Book"),
        full_text="\n\n".join(section.text for section in sections),
        sections=sections,
    )

    plan = GroupingPlanner(max_md_length=500).plan(document, related_hits=[])

    assert plan.decision is GroupingDecision.SEGMENT
    assert len(plan.segments) > 1
    assert plan.segments[0].title == "Chapter 1"


def test_grouping_planner_keeps_compact_pdf_in_single_markdown() -> None:
    sections = [
        DocumentSection(text="paper content " * 120, heading=f"Page {index}", page_number=index)
        for index in range(1, 31)
    ]
    document = ParsedDocument(
        load_id="load-4",
        hat="default",
        source=SourceDocument(source_type=SourceType.PDF, source_uri="paper.pdf", local_path=Path("paper.pdf"), title="JEPA Paper"),
        full_text="\n\n".join(section.text for section in sections),
        sections=sections,
    )

    plan = GroupingPlanner(max_md_length=500).plan(document, related_hits=[])

    assert plan.decision is GroupingDecision.STANDALONE
    assert not plan.segments
