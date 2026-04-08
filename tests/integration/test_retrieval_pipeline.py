from __future__ import annotations

from pathlib import Path

from arignan.grouping import GroupingDecision, GroupingPlan
from arignan.indexing import Chunker, DenseIndexer, HashingEmbedder, LexicalIndex, LexicalIndexer, LocalDenseIndex
from arignan.markdown import MarkdownRepository
from arignan.models import DocumentSection, ParsedDocument, SourceDocument, SourceType
from arignan.retrieval import RetrievalPipeline
from arignan.storage import StorageLayout


def _build_document(path: Path, load_id: str, title: str, text: str, heading: str, hat: str) -> ParsedDocument:
    path.write_text(f"# {heading}\n\n{text}\n", encoding="utf-8")
    return ParsedDocument(
        load_id=load_id,
        hat=hat,
        source=SourceDocument(
            source_type=SourceType.MARKDOWN,
            source_uri=str(path),
            local_path=path,
            title=title,
        ),
        full_text=text,
        sections=[DocumentSection(text=text, heading=heading)],
        keywords=[title.split()[0].lower()],
    )


def _index_document(layout: StorageLayout, document: ParsedDocument) -> None:
    chunks = Chunker(chunk_size=80, chunk_overlap=20).chunk_document(document)
    DenseIndexer(HashingEmbedder(dimension=24), LocalDenseIndex(layout.hat(document.hat).vector_index_dir)).index_chunks(chunks)
    LexicalIndexer(LexicalIndex(layout.hat(document.hat).bm25_index_dir)).index_chunks(chunks)


def test_retrieval_pipeline_combines_dense_keyword_and_map_hits(app_home: Path) -> None:
    layout = StorageLayout.from_home(app_home).ensure()
    document = _build_document(
        app_home / "jepa.md",
        load_id="load-jepa",
        title="JEPA Notes",
        text="Joint embedding predictive architecture helps with representation learning.",
        heading="JEPA Notes",
        hat="default",
    )
    _index_document(layout, document)
    MarkdownRepository().write_topic(
        layout,
        hat="default",
        documents=[document],
        plan=GroupingPlan(decision=GroupingDecision.STANDALONE, topic_folder="jepa-notes", estimated_length=300),
    )

    bundle = RetrievalPipeline(layout, embedder=HashingEmbedder(dimension=24)).retrieve("What is JEPA architecture?", hat="default")

    assert bundle.selected_hat == "default"
    assert bundle.dense_hits
    assert bundle.lexical_hits
    assert bundle.map_hits
    assert bundle.fused_hits[0].extras["rrf_score"] > 0


def test_retrieval_pipeline_auto_selects_hat_from_maps(app_home: Path) -> None:
    layout = StorageLayout.from_home(app_home).ensure(include_default_hat=False)
    default_doc = _build_document(
        app_home / "jepa.md",
        load_id="load-jepa",
        title="JEPA Notes",
        text="Joint embedding predictive architecture and representation learning.",
        heading="JEPA Notes",
        hat="research",
    )
    psych_doc = _build_document(
        app_home / "psych.md",
        load_id="load-psych",
        title="Psychology Notes",
        text="Cognitive biases and therapy methods.",
        heading="Psychology Notes",
        hat="psychology",
    )
    for document in [default_doc, psych_doc]:
        layout.hat(document.hat).ensure()
        _index_document(layout, document)
        MarkdownRepository().write_topic(
            layout,
            hat=document.hat,
            documents=[document],
            plan=GroupingPlan(
                decision=GroupingDecision.STANDALONE,
                topic_folder=document.source.title.lower().replace(" ", "-"),
                estimated_length=300,
            ),
        )

    bundle = RetrievalPipeline(layout, embedder=HashingEmbedder(dimension=24)).retrieve("representation learning with JEPA", hat="auto")

    assert bundle.selected_hat == "research"
