from __future__ import annotations

from pathlib import Path

from arignan.grouping import GroupingDecision, GroupingPlanner
from arignan.indexing import DenseIndexer, HashingEmbedder, LocalDenseIndex
from arignan.models import ChunkMetadata, ChunkRecord, DocumentSection, ParsedDocument, SourceDocument, SourceType


def test_grouping_pipeline_uses_dense_hits_for_merge(app_home: Path) -> None:
    existing_chunk = ChunkRecord(
        chunk_id="existing-1",
        text="JEPA summary about joint embedding predictive architecture and representation learning",
        metadata=ChunkMetadata(
            load_id="load-existing",
            hat="default",
            source_uri="existing.md",
            topic_folder="jepa",
        ),
    )
    dense = DenseIndexer(HashingEmbedder(dimension=24), LocalDenseIndex(app_home / "vector_index"))
    dense.index_chunks([existing_chunk])

    document = ParsedDocument(
        load_id="load-new",
        hat="default",
        source=SourceDocument(source_type=SourceType.MARKDOWN, source_uri="new.md", title="New JEPA Note"),
        full_text="Concise note on joint embedding predictive architectures.",
        sections=[DocumentSection(text="Concise note on joint embedding predictive architectures.", heading="New JEPA Note")],
    )

    hits = dense.search("joint embedding predictive architecture", limit=3)
    for hit in hits:
        hit.extras["topic_length_estimate"] = 250

    plan = GroupingPlanner(max_md_length=1000).plan(document, related_hits=hits)

    assert plan.decision is GroupingDecision.MERGE
    assert plan.topic_folder == "jepa"
