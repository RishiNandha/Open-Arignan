from __future__ import annotations

from pathlib import Path

from arignan.grouping import GroupingDecision, GroupingPlan
from arignan.markdown import MarkdownRepository
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
        text="Joint embedding predictive architecture notes.",
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
    assert "JEPA Notes" in summary_path.read_text(encoding="utf-8")
    assert (layout.hat("default").summaries_dir / "jepa-notes" / "original_files" / "input.md").exists()
    assert "jepa" in layout.hat("default").map_path.read_text(encoding="utf-8").lower()
    assert "default" in layout.global_map_path.read_text(encoding="utf-8").lower()


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
