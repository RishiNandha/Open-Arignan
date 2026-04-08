from __future__ import annotations

from pathlib import Path

from arignan.indexing import Chunker, DenseIndexer, HashingEmbedder, LocalDenseIndex
from arignan.ingestion import DocumentParser
from arignan.models import SourceDocument, SourceType


def test_dense_index_search_and_delete(app_home: Path) -> None:
    fixture = Path("tests/fixtures/markdown/sample_notes.md").resolve()
    parser = DocumentParser()
    parsed = parser.parse(
        SourceDocument(source_type=SourceType.MARKDOWN, source_uri=str(fixture), local_path=fixture),
        load_id="load-dense",
        hat="default",
    )
    chunks = Chunker(chunk_size=80, chunk_overlap=20).chunk_document(parsed)
    index = LocalDenseIndex(app_home / "vector_index")
    dense = DenseIndexer(HashingEmbedder(dimension=24), index)

    embedded = dense.index_chunks(chunks)
    hits = dense.search("predictive architecture", limit=3)

    assert len(embedded) == len(chunks)
    assert hits
    assert hits[0].metadata.load_id == "load-dense"

    dense.delete_load("load-dense")

    assert index.all_chunks() == []
