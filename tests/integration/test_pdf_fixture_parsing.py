from __future__ import annotations

from arignan.ingestion import DocumentParser
from arignan.models import SourceDocument, SourceType
from tests.fixtures.pdf_fixture import write_text_pdf


def test_pdf_fixture_parses_with_real_pdf_reader(tmp_path) -> None:
    pdf_path = write_text_pdf(tmp_path / "sample.pdf", "Sample PDF text about JEPA retrieval")
    parser = DocumentParser()

    parsed = parser.parse(
        SourceDocument(source_type=SourceType.PDF, source_uri=str(pdf_path), local_path=pdf_path),
        load_id="load-pdf",
        hat="default",
    )

    assert "Sample PDF text" in parsed.full_text
    assert parsed.sections[0].page_number == 1
