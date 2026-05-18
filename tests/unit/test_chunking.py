from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from arignan.indexing import Chunker
from arignan.indexing.chunking import AUTHOR_YEAR_CITATION_PATTERN
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
    assert chunks[0].text.startswith("Context: notes.md | Intro")


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
    assert "It predicts latent targets from context" in chunks[1].text


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
    assert chunk.text.endswith(
        "Joint embedding predictive architecture improves representations and outperforms prior baselines."
    )


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


def test_chunker_preserves_academic_section_boundaries_and_context() -> None:
    document = ParsedDocument(
        load_id="load-academic",
        hat="default",
        source=SourceDocument(source_type=SourceType.PDF, source_uri="paper.pdf", local_path=Path("paper.pdf"), title="Word2Vec Notes"),
        full_text=(
            "Skip-gram learns word representations from nearby words.\n\n"
            "Training uses negative sampling and subsampling."
        ),
        sections=[
            DocumentSection(text="Skip-gram learns word representations from nearby words.", heading="Introduction"),
            DocumentSection(text="Training uses negative sampling and subsampling.", heading="Methods"),
        ],
    )

    chunks = Chunker(chunk_size=500, chunk_overlap=60).chunk_document(document)

    assert len(chunks) == 2
    assert chunks[0].metadata.heading == "Introduction"
    assert chunks[1].metadata.heading == "Methods"
    assert chunks[0].text.startswith("Context: Word2Vec Notes | Introduction")
    assert chunks[1].text.startswith("Context: Word2Vec Notes | Methods | Method")


# ---------------------------------------------------------------------------
# AUTHOR_YEAR_CITATION_PATTERN — ReDoS safety + correctness
# ---------------------------------------------------------------------------


class TestAuthorYearCitationPatternReDoSSafety:
    """Verify the citation regex is both correct and safe from ReDoS.

    The original nested-quantifier pattern caused catastrophic backtracking on
    adversarial input.  The replacement uses a bounded [^)] character class
    which is O(n) and cannot backtrack across paren boundaries.
    """

    # --- correctness: should match legitimate citations ---

    @pytest.mark.parametrize("citation", [
        "(Smith, 2021)",
        "(Smith, 2019a)",
        "(Jones and Brown, 2020)",
        "(Smith et al., 2018)",
        "(Smith, 2021; Jones, 2022)",
        "(De Villiers, 1999)",
        "(O'Brien, 2015)",
    ])
    def test_matches_legitimate_author_year_citation(self, citation: str) -> None:
        assert AUTHOR_YEAR_CITATION_PATTERN.search(citation), (
            f"Expected citation '{citation}' to match but it did not"
        )

    # --- correctness: should NOT match non-citation parens ---

    @pytest.mark.parametrize("non_citation", [
        "(see Figure 3)",          # no year
        "(p < 0.05)",              # not a citation
        "(n = 150)",               # sample size
        "(1984)",                  # year only, no author
        "",                        # empty
        "no parens here",
    ])
    def test_does_not_match_non_citation(self, non_citation: str) -> None:
        # These should either not match or match only the citation-like part —
        # the key property is that they don't cause a hang.
        start = time.monotonic()
        AUTHOR_YEAR_CITATION_PATTERN.search(non_citation)
        elapsed = time.monotonic() - start
        assert elapsed < 0.1, f"Regex took {elapsed:.3f}s on non-citation — possible backtracking"

    # --- ReDoS safety: adversarial inputs must complete in O(1) time ---

    @pytest.mark.parametrize("adversarial", [
        # Original catastrophic backtracking trigger: many uppercase tokens
        # separated by spaces but no valid year at the end.
        "(" + " and ".join(["Smith"] * 30),
        # Long parenthesised block with no year
        "(" + "A" * 180 + ")",
        # Semicolon-heavy string that would exhaust the old alternation
        "(" + "; ".join(["Smith, Jones"] * 25),
        # Starts correctly but never closes — must not hang
        "(Smith et al., 2021; Jones and Brown, 2020; Davis" * 5,
        # Maximum allowed content length: just under the 200-char limit
        "(S" + "m" * 196 + "1990)",
    ])
    def test_adversarial_input_completes_within_time_budget(self, adversarial: str) -> None:
        budget_seconds = 0.5  # generous — any ReDoS would take seconds to minutes
        start = time.monotonic()
        AUTHOR_YEAR_CITATION_PATTERN.search(adversarial)
        elapsed = time.monotonic() - start
        assert elapsed < budget_seconds, (
            f"Regex took {elapsed:.3f}s on adversarial input — ReDoS not fixed!\n"
            f"Input (first 80 chars): {adversarial[:80]!r}"
        )

    def test_very_long_adversarial_string_completes_quickly(self) -> None:
        # 10 000 character string that would exponentially blow up the old pattern
        adversarial = "(Smith and Jones, " * 500  # ~9000 chars, no closing paren + year
        start = time.monotonic()
        AUTHOR_YEAR_CITATION_PATTERN.search(adversarial)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, (
            f"10k-char adversarial input took {elapsed:.3f}s — ReDoS vulnerability present"
        )

    def test_pattern_strips_citations_from_chunk_text(self) -> None:
        """Chunker uses the pattern to clean text — verify end-to-end that
        legitimate citations are removed and normal text is preserved."""
        document = ParsedDocument(
            load_id="load-cite",
            hat="default",
            source=SourceDocument(
                source_type=SourceType.PDF,
                source_uri="paper.pdf",
                local_path=Path("paper.pdf"),
            ),
            full_text=(
                "Dense retrieval methods (Karpukhin et al., 2020) have shown strong results. "
                "Earlier work (Johnson, 2019; Xiong, 2021) established baselines."
            ),
            sections=[],
        )
        chunks = Chunker(chunk_size=300, chunk_overlap=20).chunk_document(document)
        assert chunks
        combined = " ".join(c.text for c in chunks)
        # Citations should be stripped from the chunk text
        assert "(Karpukhin et al., 2020)" not in combined
        assert "(Johnson, 2019; Xiong, 2021)" not in combined
        # Core content should remain
        assert "Dense retrieval methods" in combined
