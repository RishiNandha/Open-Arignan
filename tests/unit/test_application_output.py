from __future__ import annotations

from pathlib import Path

from arignan.application import format_citation, synthesize_answer
from arignan.models import ChunkMetadata, RetrievalHit, RetrievalSource


def test_format_citation_includes_hat_topic_file_and_location() -> None:
    hit = RetrievalHit(
        chunk_id="chunk-1",
        text="Joint embedding predictive architecture learns representations from context.",
        score=0.9,
        source=RetrievalSource.DENSE,
        metadata=ChunkMetadata(
            load_id="load-1",
            hat="research",
            source_uri="paper.pdf",
            source_path=Path("paper.pdf"),
            page_number=12,
            heading="Training Objective",
            section="Training Objective",
            topic_folder="jepa-paper",
        ),
    )

    citation = format_citation(hit)

    assert citation == "research/jepa-paper/paper.pdf: Page 12, Training Objective"


def test_synthesize_answer_avoids_heading_only_output() -> None:
    hit = RetrievalHit(
        chunk_id="chunk-2",
        text=(
            "Joint embedding predictive architecture learns representations without rebuilding every input. "
            "It predicts latent targets from context so the model focuses on semantics."
        ),
        score=0.8,
        source=RetrievalSource.DENSE,
        metadata=ChunkMetadata(
            load_id="load-2",
            hat="default",
            source_uri="notes.md",
            source_path=Path("notes.md"),
            heading="Training",
            section="Training",
            topic_folder="jepa-notes",
        ),
    )

    answer = synthesize_answer("What is JEPA?", [hit])

    assert "Training" not in answer
    assert "predicts latent targets from context" in answer
