from __future__ import annotations

from pathlib import Path

from arignan.grouping import GroupingDecision, GroupingPlan
from arignan.markdown import MarkdownRepository, derive_keywords
from arignan.models import DocumentSection, ParsedDocument, SourceDocument, SourceType
from arignan.storage import StorageLayout


def _document(path: Path, load_id: str, title: str, text: str, heading: str) -> ParsedDocument:
    path.write_text(f"# {heading}\n\n{text}\n", encoding="utf-8")
    return ParsedDocument(
        load_id=load_id,
        hat="default",
        source=SourceDocument(
            source_type=SourceType.MARKDOWN,
            source_uri=str(path),
            local_path=path,
            title=title,
        ),
        full_text=text,
        sections=[DocumentSection(text=text, heading=heading)],
        keywords=["jepa"],
    )


def test_markdown_repository_writes_topic_and_updates_maps(app_home: Path) -> None:
    layout = StorageLayout.from_home(app_home).ensure()
    document = _document(
        app_home / "input.md",
        load_id="load-1",
        title="JEPA Notes",
        text=(
            "Joint embedding predictive architecture is a self-supervised learning approach. "
            "It builds latent representations that support prediction without requiring labels. "
            "These notes focus on the core idea, training setup, and practical uses."
        ),
        heading="JEPA Notes",
    )
    plan = GroupingPlan(
        decision=GroupingDecision.STANDALONE,
        topic_folder="jepa-notes",
        estimated_length=300,
    )

    artifact = MarkdownRepository().write_topic(layout, hat="default", documents=[document], plan=plan)

    summary_path = layout.hat("default").summaries_dir / "jepa-notes" / "markdown_tree" / "summary.md"
    assert artifact.markdown_paths == [summary_path]
    assert summary_path.exists()
    summary_text = summary_path.read_text(encoding="utf-8")
    map_text = layout.hat("default").map_path.read_text(encoding="utf-8")
    global_map_text = layout.global_map_path.read_text(encoding="utf-8")
    assert "JEPA Notes" in summary_text
    assert "## Summary" in summary_text
    assert "## Key Ideas" in summary_text
    assert "## Sources" in summary_text
    assert "## Keywords" in summary_text
    assert "Decision:" not in summary_text
    assert "local knowledge-base entry" not in summary_text
    assert "This source is derived from" not in summary_text
    assert "| Source | What To Find | Key Sections | File |" in summary_text
    assert (layout.hat("default").summaries_dir / "jepa-notes" / "original_files" / "input.md").exists()
    assert "| Topic | Directory | What To Find | Source Files | Keywords |" in map_text
    assert "Useful entry points" not in map_text
    assert "| Hat | Map Path | What To Find | High-Level Keywords |" in global_map_text
    assert "default" in global_map_text.lower()


def test_markdown_repository_cleans_citation_noise_from_summary_and_map(app_home: Path) -> None:
    layout = StorageLayout.from_home(app_home).ensure()
    document = _document(
        app_home / "paper.md",
        load_id="load-noise",
        title="V-JEPA 2",
        text=(
            "Joint embedding predictive architecture improves world models (Bardes et al., 2024) [12].\n\n"
            "References\nSmith, J. 2020."
        ),
        heading="V-JEPA 2",
    )
    document.sections = [
        DocumentSection(
            text="Joint embedding predictive architecture improves world models (Bardes et al., 2024) [12].",
            heading="Overview",
        ),
        DocumentSection(text="Smith, J. 2020.", heading="References"),
    ]
    plan = GroupingPlan(
        decision=GroupingDecision.STANDALONE,
        topic_folder="v-jepa-2",
        estimated_length=300,
    )

    MarkdownRepository().write_topic(layout, hat="default", documents=[document], plan=plan)

    summary_text = (layout.hat("default").summaries_dir / "v-jepa-2" / "markdown_tree" / "summary.md").read_text(encoding="utf-8")
    map_text = layout.hat("default").map_path.read_text(encoding="utf-8")

    assert "Bardes et al., 2024" not in summary_text
    assert "[12]" not in summary_text
    assert "Smith, J. 2020." not in summary_text
    assert "Bardes et al., 2024" not in map_text


def test_markdown_repository_writes_summary_in_neutral_wiki_voice(app_home: Path) -> None:
    layout = StorageLayout.from_home(app_home).ensure()
    document = _document(
        app_home / "paper.md",
        load_id="load-wiki",
        title="V-JEPA 2",
        text=(
            "We propose a world model that learns predictive video representations from unlabeled clips. "
            "The method focuses on latent prediction, temporal abstraction, and evaluation across downstream tasks."
        ),
        heading="V-JEPA 2",
    )
    plan = GroupingPlan(
        decision=GroupingDecision.STANDALONE,
        topic_folder="v-jepa-2",
        estimated_length=300,
    )

    MarkdownRepository().write_topic(layout, hat="default", documents=[document], plan=plan)

    summary_text = (layout.hat("default").summaries_dir / "v-jepa-2" / "markdown_tree" / "summary.md").read_text(encoding="utf-8")

    assert "# V-JEPA 2" in summary_text
    assert "We propose" not in summary_text
    assert "The work proposes" in summary_text
    assert "| Source | What To Find | Key Sections | File |" in summary_text


def test_markdown_repository_regenerates_grouped_topic_after_removal(app_home: Path) -> None:
    layout = StorageLayout.from_home(app_home).ensure()
    first = _document(
        app_home / "doc1.md",
        load_id="load-1",
        title="JEPA Paper 1",
        text="First grouped note.",
        heading="Paper One",
    )
    second = _document(
        app_home / "doc2.md",
        load_id="load-2",
        title="JEPA Paper 2",
        text="Second grouped note.",
        heading="Paper Two",
    )
    plan = GroupingPlan(
        decision=GroupingDecision.MERGE,
        topic_folder="jepa",
        estimated_length=500,
        merge_target_topic="jepa",
    )
    repository = MarkdownRepository()
    repository.write_topic(layout, hat="default", documents=[first, second], plan=plan)

    artifact = repository.regenerate_topic(layout, hat="default", documents=[first], plan=plan)
    source_names = {path.name for path in artifact.source_paths}
    summary_text = (layout.hat("default").summaries_dir / "jepa" / "markdown_tree" / "summary.md").read_text(encoding="utf-8")

    assert source_names == {"doc1.md"}
    assert "JEPA Paper 1" in summary_text
    assert "JEPA Paper 2" not in summary_text


def test_derive_keywords_filters_page_noise_and_numbers() -> None:
    document = ParsedDocument(
        load_id="load-keywords",
        hat="default",
        source=SourceDocument(
            source_type=SourceType.PDF,
            source_uri="V-JEPA2.1.pdf",
            title="V-JEPA2.1",
        ),
        full_text=(
            "Joint embedding predictive architecture learns predictive video representations. "
            "The model uses latent prediction and world models for video understanding."
        ),
        sections=[
            DocumentSection(text="Joint embedding predictive architecture learns predictive video representations.", heading="Page 1", page_number=1),
            DocumentSection(text="The model uses latent prediction and world models for video understanding.", heading="Page 2", page_number=2),
        ],
    )

    keywords = derive_keywords([document])
    lowered = {keyword.lower() for keyword in keywords}

    assert "page" not in lowered
    assert "1" not in lowered
    assert "2" not in lowered
    assert "v" not in lowered
    assert any("jepa" in keyword.lower() for keyword in keywords)
