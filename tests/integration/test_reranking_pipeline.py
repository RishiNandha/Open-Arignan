from __future__ import annotations

from pathlib import Path

from arignan.grouping import GroupingDecision, GroupingPlan
from arignan.indexing import Chunker, DenseIndexer, HashingEmbedder, LexicalIndex, LexicalIndexer, LocalDenseIndex
from arignan.markdown import MarkdownRepository
from arignan.models import DocumentSection, ParsedDocument, SourceDocument, SourceType
from arignan.retrieval import HeuristicReranker, RetrievalPipeline
from arignan.storage import StorageLayout


def test_reranking_pipeline_prunes_irrelevant_fused_hits(app_home: Path) -> None:
    layout = StorageLayout.from_home(app_home).ensure()
    relevant = ParsedDocument(
        load_id="load-rel",
        hat="default",
        source=SourceDocument(source_type=SourceType.MARKDOWN, source_uri=str(app_home / "rel.md"), title="JEPA Notes"),
        full_text="Joint embedding predictive architecture overview.",
        sections=[DocumentSection(text="Joint embedding predictive architecture overview.", heading="JEPA Notes")],
    )
    irrelevant = ParsedDocument(
        load_id="load-irr",
        hat="default",
        source=SourceDocument(source_type=SourceType.MARKDOWN, source_uri=str(app_home / "irr.md"), title="Therapy Notes"),
        full_text="Cognitive therapy methods and bias correction.",
        sections=[DocumentSection(text="Cognitive therapy methods and bias correction.", heading="Therapy Notes")],
    )
    for document in [relevant, irrelevant]:
        chunks = Chunker(chunk_size=80, chunk_overlap=20).chunk_document(document)
        DenseIndexer(HashingEmbedder(dimension=24), LocalDenseIndex(layout.hat("default").vector_index_dir)).index_chunks(chunks)
        LexicalIndexer(LexicalIndex(layout.hat("default").bm25_index_dir)).index_chunks(chunks)
        MarkdownRepository().write_topic(
            layout,
            hat="default",
            documents=[document],
            plan=GroupingPlan(
                decision=GroupingDecision.STANDALONE,
                topic_folder=document.source.title.lower().replace(" ", "-"),
                estimated_length=300,
            ),
        )

    bundle = RetrievalPipeline(layout, embedder=HashingEmbedder(dimension=24)).retrieve("joint embedding architecture", hat="default")
    reranked = HeuristicReranker().rerank(bundle.expanded_query, bundle.fused_hits, limit=5, min_score=0.3)

    assert reranked
    assert reranked[0].metadata.load_id == "load-rel"
    assert all(hit.metadata.load_id != "load-irr" for hit in reranked)
