from __future__ import annotations

from pathlib import Path

from arignan.indexing import Chunker
from arignan.models import DocumentSection, ParsedDocument, SourceDocument, SourceType


def test_chunker_prefers_document_sections() -> None:
    document = ParsedDocument(
        load_id="load-1",
        hat="default",
        source=SourceDocument(
            source_type=SourceType.MARKDOWN,
            source_uri="notes.md",
            local_path=Path("notes.md"),
        ),
        full_text="Intro text\n\nMore text",
        sections=[
            DocumentSection(text="Intro text", heading="Intro"),
            DocumentSection(text="More text", heading="Details"),
        ],
        keywords=["jepa"],
    )

    chunks = Chunker(chunk_size=50, chunk_overlap=10).chunk_document(document)

    assert [chunk.metadata.heading for chunk in chunks] == ["Intro", "Details"]
    assert [chunk.metadata.section for chunk in chunks] == ["Intro", "Details"]
    assert all(chunk.metadata.keywords == ["jepa"] for chunk in chunks)


def test_chunker_falls_back_to_overlap_for_long_unstructured_text() -> None:
    document = ParsedDocument(
        load_id="load-2",
        hat="default",
        source=SourceDocument(source_type=SourceType.MARKDOWN, source_uri="flat.md"),
        full_text="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
        sections=[DocumentSection(text="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu")],
    )

    chunks = Chunker(chunk_size=25, chunk_overlap=8).chunk_document(document)

    assert len(chunks) >= 2
    overlap_word = chunks[0].text.split()[-1]
    assert chunks[1].text.startswith(overlap_word)


def test_chunker_prefers_full_sentences_when_possible() -> None:
    text = (
        "Joint embedding predictive architecture learns compact representations. "
        "It predicts latent targets from context rather than reconstructing raw pixels. "
        "This often makes the retrieved chunk much easier to read."
    )
    document = ParsedDocument(
        load_id="load-2b",
        hat="default",
        source=SourceDocument(source_type=SourceType.MARKDOWN, source_uri="sentences.md"),
        full_text=text,
        sections=[DocumentSection(text=text)],
    )

    chunks = Chunker(chunk_size=150, chunk_overlap=40).chunk_document(document)

    assert len(chunks) == 2
    assert chunks[0].text.endswith(".")
    assert "rather than reconstructing raw pixels." in chunks[0].text
    assert chunks[1].text.startswith("It predicts latent targets from context")


def test_chunker_removes_inline_academic_citation_noise() -> None:
    text = (
        "Joint embedding predictive architecture improves representations (Bardes et al., 2022) "
        "and outperforms prior baselines [12, 14]."
    )
    document = ParsedDocument(
        load_id="load-2c",
        hat="default",
        source=SourceDocument(source_type=SourceType.MARKDOWN, source_uri="paper.md"),
        full_text=text,
        sections=[DocumentSection(text=text)],
    )

    chunk = Chunker(chunk_size=300, chunk_overlap=40).chunk_document(document)[0]

    assert "Bardes et al., 2022" not in chunk.text
    assert "[12, 14]" not in chunk.text
    assert chunk.text == "Joint embedding predictive architecture improves representations and outperforms prior baselines."


def test_chunker_skips_reference_sections() -> None:
    document = ParsedDocument(
        load_id="load-2d",
        hat="default",
        source=SourceDocument(source_type=SourceType.MARKDOWN, source_uri="paper.md"),
        full_text="Main body\n\nReferences\n\nSmith, J. 2020.",
        sections=[
            DocumentSection(text="Main body explanation.", heading="Overview"),
            DocumentSection(text="Smith, J. 2020.\nDoe, A. 2021.", heading="References"),
        ],
    )

    chunks = Chunker(chunk_size=300, chunk_overlap=40).chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].metadata.heading == "Overview"
    assert "Smith" not in chunks[0].text


def test_chunker_preserves_page_metadata() -> None:
    document = ParsedDocument(
        load_id="load-3",
        hat="default",
        source=SourceDocument(source_type=SourceType.PDF, source_uri="book.pdf", local_path=Path("book.pdf")),
        full_text="Page one text",
        sections=[DocumentSection(text="Page one text", page_number=1, heading="Page 1")],
    )

    chunk = Chunker(chunk_size=100, chunk_overlap=10).chunk_document(document)[0]

    assert chunk.metadata.page_number == 1
    assert chunk.metadata.source_path == Path("book.pdf")
    assert chunk.metadata.section == "Page 1"


def test_chunker_merges_adjacent_page_sections_into_larger_span() -> None:
    document = ParsedDocument(
        load_id="load-4",
        hat="default",
        source=SourceDocument(source_type=SourceType.PDF, source_uri="paper.pdf", local_path=Path("paper.pdf")),
        full_text=(
            "Page one introduces JEPA as a predictive architecture for latent targets.\n\n"
            "Page two explains temporal context and representation quality.\n\n"
            "Page three connects the objective to downstream understanding tasks."
        ),
        sections=[
            DocumentSection(text="Page one introduces JEPA as a predictive architecture for latent targets.", page_number=1, heading="Page 1"),
            DocumentSection(text="Page two explains temporal context and representation quality.", page_number=2, heading="Page 2"),
            DocumentSection(text="Page three connects the objective to downstream understanding tasks.", page_number=3, heading="Page 3"),
        ],
    )

    chunks = Chunker(chunk_size=260, chunk_overlap=40).chunk_document(document)

    assert len(chunks) == 1
    assert "Page one introduces JEPA" in chunks[0].text
    assert "Page three connects the objective" in chunks[0].text
    assert chunks[0].metadata.page_number is None
    assert chunks[0].metadata.section == "Pages 1-3"
