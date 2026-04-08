from __future__ import annotations

from pathlib import Path

from arignan.indexing import Chunker
from arignan.ingestion import DocumentParser
from arignan.models import SourceDocument, SourceType


def test_chunking_pipeline_from_markdown_fixture() -> None:
    fixture = Path("tests/fixtures/markdown/sample_notes.md").resolve()
    parser = DocumentParser()
    parsed = parser.parse(
        SourceDocument(source_type=SourceType.MARKDOWN, source_uri=str(fixture), local_path=fixture),
        load_id="load-fixture",
        hat="default",
    )

    chunks = Chunker(chunk_size=80, chunk_overlap=20).chunk_document(parsed)

    assert len(chunks) >= 2
    assert chunks[0].metadata.heading == "JEPA Notes"
    assert chunks[-1].metadata.section == "Practical Use"
