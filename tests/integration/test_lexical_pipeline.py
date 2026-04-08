from __future__ import annotations

from pathlib import Path

from arignan.indexing import Chunker, LexicalIndex, LexicalIndexer
from arignan.ingestion import DocumentParser
from arignan.models import SourceDocument, SourceType


def test_lexical_pipeline_search_and_delete(app_home: Path) -> None:
    fixture = Path("tests/fixtures/markdown/sample_notes.md").resolve()
    parser = DocumentParser()
    parsed = parser.parse(
        SourceDocument(source_type=SourceType.MARKDOWN, source_uri=str(fixture), local_path=fixture),
        load_id="load-lexical",
        hat="default",
    )
    chunks = Chunker(chunk_size=80, chunk_overlap=20).chunk_document(parsed)
    lexical = LexicalIndexer(LexicalIndex(app_home / "bm25_index"))

    lexical.index_chunks(chunks)
    hits = lexical.search("predictive architecture", limit=3)

    assert hits
    assert hits[0].metadata.load_id == "load-lexical"

    lexical.delete_load("load-lexical")

    assert lexical.index.all_chunks() == []
