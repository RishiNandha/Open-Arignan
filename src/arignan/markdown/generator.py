from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from arignan.grouping import GroupingDecision, GroupingPlan, slugify
from arignan.models import ParsedDocument, TopicArtifact
from arignan.storage import StorageLayout

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


class MarkdownRepository:
    def write_topic(
        self,
        layout: StorageLayout,
        hat: str,
        documents: list[ParsedDocument],
        plan: GroupingPlan,
    ) -> TopicArtifact:
        hat_layout = layout.hat(hat).ensure()
        topic_dir = hat_layout.summaries_dir / plan.topic_folder
        if topic_dir.exists():
            shutil.rmtree(topic_dir)
        topic_dir.mkdir(parents=True, exist_ok=True)
        original_files_dir = topic_dir / "original_files"
        original_files_dir.mkdir(parents=True, exist_ok=True)
        markdown_tree_dir = topic_dir / "markdown_tree"
        markdown_tree_dir.mkdir(parents=True, exist_ok=True)

        source_paths = self._write_sources(original_files_dir, documents)
        markdown_paths = self._write_markdowns(markdown_tree_dir, documents, plan)
        keywords = derive_keywords(documents)
        artifact = TopicArtifact(
            hat=hat,
            topic_folder=plan.topic_folder,
            source_paths=source_paths,
            markdown_paths=markdown_paths,
            keywords=keywords,
        )
        description = describe_documents(documents)
        self._write_manifest(topic_dir, artifact, description, documents, plan)
        self.update_hat_map(layout, hat)
        self.update_global_map(layout)
        return artifact

    def regenerate_topic(
        self,
        layout: StorageLayout,
        hat: str,
        documents: list[ParsedDocument],
        plan: GroupingPlan,
    ) -> TopicArtifact:
        return self.write_topic(layout=layout, hat=hat, documents=documents, plan=plan)

    def update_hat_map(self, layout: StorageLayout, hat: str) -> Path:
        hat_layout = layout.hat(hat).ensure()
        manifests = sorted(hat_layout.summaries_dir.glob("*/.topic_manifest.json"))
        lines = [f"# Map for Hat: {hat}", ""]
        for manifest_path in manifests:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            lines.extend(
                [
                    f"## {payload['topic_folder']}",
                    f"- Topic folder: `{payload['topic_folder']}`",
                    f"- Description: {payload['description']}",
                    "- Markdown files:",
                ]
            )
            for path in payload["markdown_paths"]:
                lines.append(f"  - `{path}`")
            lines.append("- Source files:")
            for path in payload["source_paths"]:
                lines.append(f"  - `{path}`")
            lines.append(f"- Keywords: {', '.join(payload['keywords'])}")
            lines.append("")
        hat_layout.map_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return hat_layout.map_path

    def update_global_map(self, layout: StorageLayout) -> Path:
        layout.ensure(include_default_hat=False)
        lines = ["# Global Map", ""]
        for hat_dir in sorted(path for path in layout.hats_dir.iterdir() if path.is_dir()):
            map_path = hat_dir / "map.md"
            manifests = sorted((hat_dir / "summaries").glob("*/.topic_manifest.json"))
            keywords = []
            topics = []
            for manifest_path in manifests:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                keywords.extend(payload.get("keywords", []))
                topics.append(payload["topic_folder"])
            lines.extend(
                [
                    f"## {hat_dir.name}",
                    f"- Map path: `{map_path}`",
                    f"- Topics: {', '.join(topics) if topics else 'None yet'}",
                    f"- High-level keywords: {', '.join(_dedupe(keywords)) if keywords else 'None yet'}",
                    "",
                ]
            )
        layout.global_map_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return layout.global_map_path

    def _write_sources(self, sources_dir: Path, documents: list[ParsedDocument]) -> list[Path]:
        written: list[Path] = []
        for document in documents:
            if document.source.local_path and document.source.local_path.exists():
                destination = sources_dir / document.source.local_path.name
                shutil.copy2(document.source.local_path, destination)
            else:
                stem = slugify(document.source.title or document.source.source_uri)
                destination = sources_dir / f"{stem}.source.txt"
                destination.write_text(document.source.source_uri + "\n", encoding="utf-8")
            written.append(destination)
        return written

    def _write_markdowns(
        self,
        markdown_tree_dir: Path,
        documents: list[ParsedDocument],
        plan: GroupingPlan,
    ) -> list[Path]:
        if plan.decision is GroupingDecision.SEGMENT and len(documents) == 1 and plan.segments:
            return self._write_segmented_markdowns(markdown_tree_dir, documents[0], plan)
        summary_path = markdown_tree_dir / "summary.md"
        summary_path.write_text(compose_topic_markdown(documents, plan), encoding="utf-8")
        return [summary_path]

    def _write_segmented_markdowns(
        self,
        markdown_tree_dir: Path,
        document: ParsedDocument,
        plan: GroupingPlan,
    ) -> list[Path]:
        written: list[Path] = []
        for index, segment in enumerate(plan.segments, start=1):
            path = markdown_tree_dir / f"{index:02d}-{segment.slug}.md"
            path.write_text(compose_segment_markdown(document, segment.section_indices, segment.title), encoding="utf-8")
            written.append(path)
        return written

    def _write_manifest(
        self,
        topic_dir: Path,
        artifact: TopicArtifact,
        description: str,
        documents: list[ParsedDocument],
        plan: GroupingPlan,
    ) -> Path:
        payload = artifact.to_dict()
        payload["description"] = description
        payload["documents"] = [document.to_dict() for document in documents]
        payload["decision"] = plan.decision.value
        manifest_path = topic_dir / ".topic_manifest.json"
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return manifest_path


def compose_topic_markdown(documents: list[ParsedDocument], plan: GroupingPlan) -> str:
    title = humanize_topic_folder(plan.topic_folder)
    lines = [f"# {title}", ""]
    lines.append(f"Decision: `{plan.decision.value}`")
    lines.append("")
    lines.append("## Sources")
    for document in documents:
        source_ref = document.source.local_path.name if document.source.local_path else document.source.source_uri
        lines.append(f"- {document.source.title or source_ref}: `{source_ref}`")
    lines.append("")
    lines.append("## Summary")
    lines.append(describe_documents(documents))
    lines.append("")
    lines.append("## Key Sections")
    for document in documents:
        lines.append(f"### {document.source.title or document.source.source_uri}")
        for section in document.sections[:5]:
            title_line = section.heading or (f"Page {section.page_number}" if section.page_number is not None else "Section")
            lines.append(f"- {title_line}: {summarize_text(section.text)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def compose_segment_markdown(document: ParsedDocument, section_indices: list[int], title: str) -> str:
    lines = [f"# {title}", ""]
    lines.append(f"Source: `{document.source.title or document.source.source_uri}`")
    lines.append("")
    for index in section_indices:
        section = document.sections[index]
        heading = section.heading or (f"Page {section.page_number}" if section.page_number is not None else f"Section {index + 1}")
        lines.append(f"## {heading}")
        lines.append(summarize_text(section.text))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def describe_documents(documents: list[ParsedDocument]) -> str:
    titles = [document.source.title or document.source.source_uri for document in documents]
    if len(titles) == 1:
        return f"Focused notes derived from {titles[0]}."
    return f"Grouped notes combining {', '.join(titles[:-1])}, and {titles[-1]}."


def summarize_text(text: str, max_length: int = 180) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


def derive_keywords(documents: list[ParsedDocument], limit: int = 8) -> list[str]:
    counts: Counter[str] = Counter()
    for document in documents:
        if document.keywords:
            counts.update(keyword.lower() for keyword in document.keywords)
        for source in [document.source.title or ""]:
            counts.update(word for word in slugify(source).split("-") if word and word not in STOPWORDS)
        for section in document.sections[:8]:
            if section.heading:
                counts.update(word for word in slugify(section.heading).split("-") if word and word not in STOPWORDS)
    return [word for word, _ in counts.most_common(limit)]


def humanize_topic_folder(topic_folder: str) -> str:
    return " ".join(part.capitalize() for part in topic_folder.split("-")) or "Topic"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
