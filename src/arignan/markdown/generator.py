from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

import arignan.markdown.rendering as _rendering
from arignan.grouping import GroupingDecision, GroupingPlan, slugify
from arignan.models import DocumentSection, ParsedDocument, TopicArtifact
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
GENERIC_KEYWORD_TOKENS = STOPWORDS | {
    "appendix",
    "bibliography",
    "chapter",
    "chapters",
    "citation",
    "citations",
    "conclusion",
    "conclusions",
    "default",
    "describes",
    "doc",
    "document",
    "documents",
    "evaluate",
    "evaluates",
    "figure",
    "figures",
    "file",
    "files",
    "focus",
    "focuses",
    "here",
    "introduction",
    "introduce",
    "introduces",
    "learn",
    "learns",
    "made",
    "md",
    "method",
    "methods",
    "model",
    "models",
    "note",
    "notes",
    "overview",
    "page",
    "pages",
    "paper",
    "papers",
    "pdf",
    "propose",
    "proposes",
    "reference",
    "references",
    "report",
    "reports",
    "result",
    "results",
    "section",
    "sections",
    "show",
    "shows",
    "study",
    "summary",
    "table",
    "tables",
    "that",
    "their",
    "there",
    "these",
    "this",
    "topic",
    "topics",
    "use",
    "used",
    "uses",
    "using",
    "version",
    "we",
    "work",
    "works",
}
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
NUMERIC_CITATION_PATTERN = re.compile(r"\[(?:\d+(?:\s*[-,]\s*\d+)*)\]")
MARKDOWN_CITATION_PATTERN = re.compile(r"\[@[^\]]+\]|\[\^[^\]]+\]")
AUTHOR_YEAR_CITATION_PATTERN = re.compile(
    r"\((?:[A-Z][A-Za-z'`\-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z'`\-]+|\s+et al\.)?,\s*(?:19|20)\d{2}[a-z]?)(?:;\s*(?:[A-Z][A-Za-z'`\-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z'`\-]+|\s+et al\.)?,\s*(?:19|20)\d{2}[a-z]?))*\)"
)
INLINE_REFERENCE_PATTERN = re.compile(
    r"\b(?:fig(?:ure)?|table|eq(?:uation)?|sec(?:tion)?)\.?\s*\d+[A-Za-z\-]*\b",
    re.IGNORECASE,
)
INLINE_AUTHOR_YEAR_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z'`\-]+(?:\s+et al\.)?,\s*(?:19|20)\d{2}[a-z]?\b"
)
REFERENCE_HEADING_PATTERN = re.compile(r"^(references|bibliography|works cited|citations?)$", re.IGNORECASE)
REFERENCE_BLOCK_PATTERN = re.compile(
    r"(?:^|\n)\s*(references|bibliography|works cited|citations?)\s*(?:\n|$).*$",
    re.IGNORECASE | re.DOTALL,
)
KEYWORD_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}")
ACRONYM_PATTERN = re.compile(r"\b(?:[A-Z]{2,}(?:-[A-Z0-9]+)*|[A-Z](?:-[A-Z0-9]+)+(?:\.\d+)*)\b")


class MarkdownRepository:
    def __init__(self, artifact_writer=None) -> None:
        if artifact_writer is None:
            from arignan.markdown.writer import HeuristicArtifactWriter

            artifact_writer = HeuristicArtifactWriter()
        self.artifact_writer = artifact_writer

    def write_topic(
        self,
        layout: StorageLayout,
        hat: str,
        documents: list[ParsedDocument],
        plan: GroupingPlan,
        refresh_maps: bool = True,
    ) -> TopicArtifact:
        hat_layout = layout.hat(hat).ensure()
        topic_dir = hat_layout.summaries_dir / plan.topic_folder
        if topic_dir.exists():
            shutil.rmtree(topic_dir)
        topic_dir.mkdir(parents=True, exist_ok=True)
        original_files_dir = topic_dir / "original_files"
        original_files_dir.mkdir(parents=True, exist_ok=True)

        source_paths = self._write_sources(original_files_dir, documents)
        rendered_topic = self.artifact_writer.render_topic(documents, plan)
        markdown_paths = self._write_markdowns(topic_dir, documents, plan, rendered_topic.summary_markdown)
        support_paths = self._write_support_markdowns(
            topic_dir,
            documents,
            plan,
            title=rendered_topic.title,
            locator=rendered_topic.locator,
            keywords=rendered_topic.keywords,
        )
        keywords = rendered_topic.keywords or derive_keywords(documents)
        artifact = TopicArtifact(
            hat=hat,
            topic_folder=plan.topic_folder,
            source_paths=source_paths,
            markdown_paths=markdown_paths,
            keywords=keywords,
        )
        description = rendered_topic.description or describe_documents(documents)
        self._write_manifest(
            topic_dir,
            artifact,
            description,
            documents,
            plan,
            title=rendered_topic.title,
            locator=rendered_topic.locator,
            support_paths=support_paths,
        )
        if refresh_maps:
            self.update_hat_map(layout, hat)
            self.update_global_map(layout)
        return artifact

    def regenerate_topic(
        self,
        layout: StorageLayout,
        hat: str,
        documents: list[ParsedDocument],
        plan: GroupingPlan,
        refresh_maps: bool = True,
    ) -> TopicArtifact:
        return self.write_topic(layout=layout, hat=hat, documents=documents, plan=plan, refresh_maps=refresh_maps)

    def update_hat_map(self, layout: StorageLayout, hat: str) -> Path:
        from arignan.markdown.writer import TopicMapEntry

        hat_layout = layout.hat(hat).ensure()
        manifests = sorted(hat_layout.summaries_dir.glob("*/.topic_manifest.json"))
        entries: list[TopicMapEntry] = []
        for manifest_path in manifests:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            documents = [ParsedDocument.from_dict(item) for item in payload.get("documents", [])]
            entries.append(
                TopicMapEntry(
                    topic_folder=payload["topic_folder"],
                    title=payload.get("title") or display_topic_title(payload["topic_folder"], documents),
                    locator=payload.get("locator") or compose_topic_locator(documents),
                    source_files=[Path(path).name for path in payload.get("source_paths", [])],
                    markdown_files=[Path(path).name for path in payload.get("markdown_paths", [])],
                    keywords=list(payload.get("keywords", [])),
                )
            )
        hat_layout.map_path.write_text(self.artifact_writer.render_hat_map(hat, entries), encoding="utf-8")
        return hat_layout.map_path

    def update_global_map(self, layout: StorageLayout) -> Path:
        from arignan.markdown.writer import HatMapEntry

        layout.ensure(include_default_hat=False)
        entries: list[HatMapEntry] = []
        for hat_dir in sorted(path for path in layout.hats_dir.iterdir() if path.is_dir()):
            manifests = sorted((hat_dir / "summaries").glob("*/.topic_manifest.json"))
            keywords: list[str] = []
            locators: list[str] = []
            for manifest_path in manifests:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                keywords.extend(payload.get("keywords", []))
                locator = payload.get("locator")
                if isinstance(locator, str) and locator.strip():
                    locators.append(locator.strip())
            entries.append(
                HatMapEntry(
                    hat=hat_dir.name,
                    map_path=(Path("hats") / hat_dir.name / "map.md").as_posix(),
                    what_to_find=natural_join(locators[:3]) if locators else "No topics yet",
                    keywords=_dedupe(keywords)[:8],
                )
            )
        layout.global_map_path.write_text(self.artifact_writer.render_global_map(entries), encoding="utf-8")
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
        topic_dir: Path,
        documents: list[ParsedDocument],
        plan: GroupingPlan,
        summary_markdown: str,
    ) -> list[Path]:
        if plan.decision is GroupingDecision.SEGMENT and len(documents) == 1 and plan.segments:
            return self._write_segmented_markdowns(topic_dir, documents[0], plan)
        summary_path = topic_dir / "summary.md"
        summary_path.write_text(summary_markdown, encoding="utf-8")
        return [summary_path]

    def _write_segmented_markdowns(
        self,
        topic_dir: Path,
        document: ParsedDocument,
        plan: GroupingPlan,
    ) -> list[Path]:
        written: list[Path] = []
        for index, segment in enumerate(plan.segments, start=1):
            path = topic_dir / f"{index:02d}-{segment.slug}.md"
            path.write_text(compose_segment_markdown(document, segment.section_indices, segment.title), encoding="utf-8")
            written.append(path)
        return written

    def _write_support_markdowns(
        self,
        topic_dir: Path,
        documents: list[ParsedDocument],
        plan: GroupingPlan,
        *,
        title: str,
        locator: str,
        keywords: list[str],
    ) -> list[Path]:
        topic_index_path = topic_dir / "topic_index.md"
        topic_index_path.write_text(
            compose_topic_index_markdown(
                documents,
                plan,
                title=title,
                locator=locator,
                keywords=keywords,
            ),
            encoding="utf-8",
        )
        return [topic_index_path]

    def _write_manifest(
        self,
        topic_dir: Path,
        artifact: TopicArtifact,
        description: str,
        documents: list[ParsedDocument],
        plan: GroupingPlan,
        title: str,
        locator: str,
        support_paths: list[Path],
    ) -> Path:
        payload = artifact.to_dict()
        payload["description"] = description
        payload["title"] = title
        payload["locator"] = locator
        payload["support_markdown_paths"] = [str(path) for path in support_paths]
        payload["documents"] = [document.to_dict() for document in documents]
        payload["decision"] = plan.decision.value
        manifest_path = topic_dir / ".topic_manifest.json"
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return manifest_path


def compose_topic_markdown(documents: list[ParsedDocument], plan: GroupingPlan) -> str:
    title = display_topic_title(plan.topic_folder, documents)
    keywords = derive_keywords(documents)
    lines = [f"# {title}", ""]
    lead = compose_topic_lead(title, documents)
    if lead:
        lines.append(lead)
        lines.append("")

    summary = compose_topic_summary(documents)
    if summary:
        lines.append("## Summary")
        lines.append(summary)
        lines.append("")

    core_ideas = topic_core_ideas(documents, limit=4)
    if core_ideas:
        lines.append("## Key Ideas")
        for sentence in core_ideas:
            lines.append(f"- {compose_key_point(sentence)}")
        lines.append("")

    lines.append("## Sources")
    lines.append("| Source | What To Find | Key Sections | File |")
    lines.append("| --- | --- | --- | --- |")
    for document in documents:
        document_title = document.source.title or source_name(document)
        headings = collect_semantic_headings([document], limit=4)
        source_ref = document.source.local_path.name if document.source.local_path else document.source.source_uri
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_table_cell(document_title),
                    _markdown_table_cell(compose_document_summary(document)),
                    _markdown_table_cell(natural_join(headings[:3]) if headings else document_outline(document)),
                    f"`{source_ref}`",
                ]
            )
            + " |"
        )
    lines.append("")

    if keywords:
        lines.append("## Keywords")
        lines.append(", ".join(keywords))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def compose_segment_markdown(document: ParsedDocument, section_indices: list[int], title: str) -> str:
    lines = [f"# {title}", ""]
    lines.append(compose_document_lead(document))
    lines.append("")
    segment_sections = [document.sections[index] for index in section_indices]
    overview = summarize_sentences(_meaningful_sentences_from_sections(segment_sections), limit=2)
    if overview:
        lines.append("## Overview")
        lines.append(" ".join(overview))
        lines.append("")
    lines.append("## Covered Sections")
    for index in section_indices:
        section = document.sections[index]
        heading = section.heading or (f"Page {section.page_number}" if section.page_number is not None else f"Section {index + 1}")
        lines.append(f"## {heading}")
        lines.append(summarize_text(clean_source_text(section.text)))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def describe_documents(documents: list[ParsedDocument]) -> str:
    locator = compose_topic_locator(documents)
    if not locator:
        return "Reference material."
    return _sentence_case(locator.rstrip(".")) + "."


def summarize_text(text: str, max_length: int = 180) -> str:
    normalized = " ".join(clean_source_text(text).split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


def derive_keywords(documents: list[ParsedDocument], limit: int = 8) -> list[str]:
    counts: Counter[str] = Counter()
    display_map: dict[str, str] = {}
    for document in documents:
        if document.keywords:
            for keyword in document.keywords:
                _add_keyword(counts, display_map, keyword, weight=6.0)
        _add_keywords_from_text(counts, display_map, document.source.title or "", weight=4.0)
        for heading in collect_semantic_headings([document], limit=8):
            _add_keyword(counts, display_map, heading, weight=3.0)
            _add_keywords_from_text(counts, display_map, heading, weight=2.5)
        _add_keywords_from_text(counts, display_map, document.full_text[:2400], weight=1.5)
    ordered = sorted(
        counts,
        key=lambda key: (_keyword_priority(display_map[key]), -counts[key], len(display_map[key]), display_map[key].lower()),
    )
    selected: list[str] = []
    seen_overlap: set[str] = set()
    for key in ordered:
        term = display_map[key]
        overlap_key = _keyword_overlap_key(term)
        if overlap_key in seen_overlap:
            continue
        if " " not in term and any(term.lower() in selected_term.lower().split() for selected_term in selected if " " in selected_term):
            continue
        selected.append(term)
        seen_overlap.add(overlap_key)
        if len(selected) >= limit:
            break
    return selected


def humanize_topic_folder(topic_folder: str) -> str:
    return " ".join(part.capitalize() for part in topic_folder.split("-")) or "Topic"


def display_topic_title(topic_folder: str, documents: list[ParsedDocument]) -> str:
    if len(documents) == 1 and documents[0].source.title:
        return documents[0].source.title
    title_candidates = [
        candidate
        for candidate in [
            *(document.source.title for document in documents if document.source.title),
            *collect_semantic_headings(documents, limit=6),
        ]
        if candidate
    ]
    for candidate in title_candidates:
        if slugify(candidate) == topic_folder:
            return candidate

    token_map: dict[str, str] = {}
    for candidate in title_candidates:
        for word in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", candidate):
            token = slugify(word)
            if token and token not in token_map:
                token_map[token] = word
    parts: list[str] = []
    for token in topic_folder.split("-"):
        if token in token_map:
            parts.append(token_map[token])
        elif token.isdigit():
            parts.append(token)
        elif len(token) <= 4:
            parts.append(token.upper())
        else:
            parts.append(token.capitalize())
    return " ".join(parts) or "Topic"


def topic_overview_sentences(documents: list[ParsedDocument], limit: int = 3) -> list[str]:
    sentences: list[str] = []
    seen: set[str] = set()
    for document in documents:
        for sentence in _meaningful_sentences(document):
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            sentences.append(sentence)
            if len(sentences) >= limit:
                return sentences
    return sentences


def topic_core_ideas(documents: list[ParsedDocument], limit: int = 4) -> list[str]:
    overview = topic_overview_sentences(documents, limit=limit + 2)
    if not overview:
        return []
    if len(overview) <= 1:
        return overview
    start_index = 1 if len(overview) <= 3 else 2
    return overview[start_index : start_index + limit]


def compose_topic_lead(title: str, documents: list[ParsedDocument]) -> str:
    overview = topic_overview_sentences(documents, limit=1)
    if overview:
        return overview[0]
    headings = collect_semantic_headings(documents, limit=4)
    if headings:
        return f"{title} covers {natural_join(headings)}."
    return f"{title} is a concise reference note."


def compose_topic_summary(documents: list[ParsedDocument]) -> str:
    sentences = topic_overview_sentences(documents, limit=5)
    remainder = sentences[1:3]
    parts: list[str] = []
    if remainder:
        parts.append(" ".join(remainder))
    headings = collect_semantic_headings(documents, limit=4)
    if headings:
        parts.append(f"Key sections cover {natural_join(headings)}.")
    elif len(documents) > 1:
        parts.append(f"The topic is assembled from {len(documents)} related sources.")
    return " ".join(parts).strip()


def compose_scope_paragraph(documents: list[ParsedDocument]) -> str:
    headings = collect_semantic_headings(documents, limit=4)
    total_sections = sum(len([section for section in document.sections if not _is_noise_section(section)]) for document in documents)
    total_pages = sum(
        1
        for document in documents
        for section in document.sections
        if section.page_number is not None and not _is_noise_section(section)
    )
    structure_parts: list[str] = []
    if total_pages:
        structure_parts.append(f"{total_pages} readable page(s)")
    if total_sections:
        structure_parts.append(f"{total_sections} meaningful section(s)")
    structure = " and ".join(structure_parts) if structure_parts else "the available readable text"
    if headings:
        return (
            f"The source material is organized around {natural_join(headings)}. "
            f"This entry condenses {structure} into a quick-reference summary."
        )
    return f"This entry condenses {structure} into a quick-reference summary."


def compose_document_lead(document: ParsedDocument) -> str:
    source_ref = document.source.local_path.name if document.source.local_path else document.source.source_uri
    headings = collect_semantic_headings([document], limit=3)
    if headings:
        return (
            f"This source is derived from `{source_ref}` and is mainly organized around {natural_join(headings)}. "
            f"The extracted notes below focus on the most readable high-signal material."
        )
    return (
        f"This source is derived from `{source_ref}`. "
        f"The extracted notes below focus on the most readable high-signal material."
    )


def compose_document_focus(document: ParsedDocument) -> str:
    sentences = topic_overview_sentences([document], limit=2)
    if sentences:
        return " ".join(sentences[:2])
    headings = collect_semantic_headings([document], limit=3)
    if headings:
        return f"Covers {natural_join(headings)}."
    return summarize_text(document.full_text)


def compose_document_summary(document: ParsedDocument) -> str:
    sentences = topic_overview_sentences([document], limit=2)
    if sentences:
        return summarize_text(" ".join(sentences), max_length=220)
    headings = collect_semantic_headings([document], limit=3)
    if headings:
        return f"Covers {natural_join(headings)}."
    return summarize_text(document.full_text, max_length=220)


def compose_topic_locator(documents: list[ParsedDocument]) -> str:
    expectation = describe_topic_expectation(documents)
    keywords = derive_keywords(documents, limit=3)
    if keywords:
        return f"{expectation} on {natural_join(keywords)}"
    headings = collect_semantic_headings(documents, limit=3)
    if headings:
        return f"{expectation} covering {natural_join(headings)}"
    return expectation


def compose_key_point(sentence: str) -> str:
    cleaned = normalize_wiki_sentence(sentence).rstrip(".")
    cleaned = re.sub(
        r"^(?:The (?:paper|work|method|model)\s+(?:proposes|presents|introduces|describes|studies|investigates|analyzes|shows|evaluates)\s+)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^(?:It|This approach|This method|The approach|The method)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" -")
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned or normalize_wiki_sentence(sentence)


def describe_document_expectation(document: ParsedDocument) -> str:
    meaningful_sections = [section for section in document.sections if not _is_noise_section(section)]
    page_numbers = sorted({section.page_number for section in meaningful_sections if section.page_number is not None})
    if document.source.source_type.value == "pdf":
        label = "long-form PDF reference" if len(page_numbers) >= 20 or len(meaningful_sections) >= 12 else "PDF paper or report"
    elif document.source.source_type.value == "markdown":
        label = "structured markdown notes" if len(meaningful_sections) >= 6 else "markdown notes"
    else:
        label = "reference material"
    headings = collect_semantic_headings([document], limit=3)
    if headings:
        return f"{label} covering {natural_join(headings)}"
    return label


def describe_topic_expectation(documents: list[ParsedDocument]) -> str:
    if not documents:
        return "reference material"
    if len(documents) == 1:
        return describe_document_expectation(documents[0])
    headings = collect_semantic_headings(documents, limit=4)
    if headings:
        return f"grouped reference notes covering {natural_join(headings)}"
    return f"grouped reference notes from {len(documents)} related sources"


def topic_entry_points(documents: list[ParsedDocument], limit: int = 4) -> list[str]:
    entries: list[str] = []
    seen: set[str] = set()
    for document in documents:
        for label, summary in document_section_highlights(document, limit=2):
            entry = f"{document.source.title or source_name(document)} -> {label}: {summary}"
            key = entry.lower()
            if key in seen:
                continue
            seen.add(key)
            entries.append(entry)
            if len(entries) >= limit:
                return entries
    return entries


def document_section_highlights(document: ParsedDocument, limit: int = 4) -> list[tuple[str, str]]:
    highlights: list[tuple[str, str]] = []
    for section in document.sections:
        if _is_noise_section(section):
            continue
        summary_sentences = _extract_sentences(section.text)
        summary = summary_sentences[0] if summary_sentences else summarize_text(section.text)
        if not summary:
            continue
        label = section.heading or (f"Page {section.page_number}" if section.page_number is not None else "Section")
        highlights.append((label, summary))
        if len(highlights) >= limit:
            break
    return highlights


def document_outline(document: ParsedDocument) -> str:
    meaningful_sections = [section for section in document.sections if not _is_noise_section(section)]
    if meaningful_sections and all(section.page_number is not None for section in meaningful_sections):
        pages = [section.page_number for section in meaningful_sections if section.page_number is not None]
        if pages:
            return f"{len(pages)} page(s)"
    if meaningful_sections:
        return f"{len(meaningful_sections)} section(s)"
    return "1 document"


def source_name(document: ParsedDocument) -> str:
    return document.source.local_path.name if document.source.local_path else document.source.source_uri


def summarize_sentences(sentences: list[str], limit: int) -> list[str]:
    return sentences[:limit]


def collect_semantic_headings(documents: list[ParsedDocument], limit: int = 4) -> list[str]:
    headings: list[str] = []
    seen: set[str] = set()
    for document in documents:
        source_label = (document.source.title or source_name(document)).strip().lower()
        for section in document.sections:
            heading = (section.heading or "").strip()
            if not heading or _is_page_heading(heading) or _is_noise_section(section):
                continue
            if heading.lower() == source_label:
                continue
            key = heading.lower()
            if key in seen:
                continue
            seen.add(key)
            headings.append(heading)
            if len(headings) >= limit:
                return headings
    return headings


def _meaningful_sentences(document: ParsedDocument) -> list[str]:
    return _meaningful_sentences_from_sections(document.sections) or _extract_sentences(document.full_text)


def _meaningful_sentences_from_sections(sections: list[DocumentSection]) -> list[str]:
    sentences: list[str] = []
    for section in sections:
        if _is_noise_section(section):
            continue
        sentences.extend(_extract_sentences(section.text))
        if len(sentences) >= 8:
            break
    return sentences


def _extract_sentences(text: str) -> list[str]:
    cleaned = clean_source_text(text)
    if not cleaned:
        return []
    candidates = [part.strip() for part in SENTENCE_BOUNDARY_PATTERN.split(cleaned) if part.strip()]
    sentences: list[str] = []
    for candidate in candidates:
        candidate = normalize_wiki_sentence(candidate)
        if len(candidate) < 40:
            continue
        if len(candidate.split()) < 7:
            continue
        if _looks_noisy(candidate):
            continue
        if candidate[-1] not in ".!?":
            candidate += "."
        sentences.append(candidate)
    return sentences


def clean_source_text(text: str) -> str:
    cleaned = REFERENCE_BLOCK_PATTERN.sub("\n", text)
    cleaned = MARKDOWN_CITATION_PATTERN.sub("", cleaned)
    cleaned = NUMERIC_CITATION_PATTERN.sub("", cleaned)
    cleaned = AUTHOR_YEAR_CITATION_PATTERN.sub("", cleaned)
    cleaned = INLINE_AUTHOR_YEAR_PATTERN.sub("", cleaned)
    cleaned = INLINE_REFERENCE_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s+([,;:.!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:])(?=[A-Za-z0-9])", r"\1 ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_wiki_sentence(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^\d+(?:\.\d+)*\s+", "", cleaned)
    cleaned = re.sub(r"^(?:in|within)\s+(?:this|the)\s+(?:paper|work|article|study),\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^here,\s*", "", cleaned, flags=re.IGNORECASE)
    substitutions = (
        (r"^we propose\b", "The work proposes"),
        (r"^we present\b", "The work presents"),
        (r"^we introduce\b", "The work introduces"),
        (r"^we describe\b", "The paper describes"),
        (r"^we study\b", "The paper studies"),
        (r"^we investigate\b", "The paper investigates"),
        (r"^we analyze\b", "The paper analyzes"),
        (r"^we show\b", "The paper shows"),
        (r"^we evaluate\b", "The paper evaluates"),
        (r"^this paper\b", "The paper"),
        (r"^this work\b", "The work"),
    )
    for pattern, replacement in substitutions:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _is_noise_section(section: DocumentSection) -> bool:
    heading = (section.heading or "").strip()
    if heading and REFERENCE_HEADING_PATTERN.fullmatch(heading):
        return True
    cleaned = clean_source_text(section.text)
    if not cleaned:
        return True
    return _looks_noisy(cleaned[:220])


def _looks_noisy(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    letters = sum(1 for char in stripped if char.isalpha())
    digits = sum(1 for char in stripped if char.isdigit())
    if letters == 0:
        return True
    if digits > letters:
        return True
    lower = stripped.lower()
    if lower.startswith(("page ", "figure ", "table ")) and len(lower.split()) <= 4:
        return True
    return False


def natural_join(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _is_page_heading(heading: str) -> bool:
    normalized = heading.strip().lower()
    return bool(re.fullmatch(r"page\s+\d+", normalized))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _markdown_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip() or "-"


def _add_keywords_from_text(
    counts: Counter[str],
    display_map: dict[str, str],
    text: str,
    weight: float,
) -> None:
    cleaned = clean_source_text(text)
    if not cleaned:
        return
    for acronym in ACRONYM_PATTERN.findall(text):
        _add_keyword(counts, display_map, acronym, weight=weight + 1.0)

    for sentence in SENTENCE_BOUNDARY_PATTERN.split(cleaned):
        for clause in re.split(r"[,:;()]+", sentence):
            raw_tokens: list[str | None] = []
            for token in KEYWORD_TOKEN_PATTERN.findall(clause):
                normalized = _normalize_keyword_display(token)
                if not _is_valid_keyword(normalized):
                    raw_tokens.append(None)
                    continue
                raw_tokens.append(normalized)
                _add_keyword(counts, display_map, normalized, weight=weight)

            for index in range(len(raw_tokens) - 1):
                first = raw_tokens[index]
                second = raw_tokens[index + 1]
                if not first or not second:
                    continue
                if len(first) < 4 or len(second) < 4:
                    continue
                phrase = f"{first} {second}"
                _add_keyword(counts, display_map, phrase, weight=weight + 0.8)


def _add_keyword(
    counts: Counter[str],
    display_map: dict[str, str],
    term: str,
    weight: float,
) -> None:
    normalized = _normalize_keyword_display(term)
    if not _is_valid_keyword(normalized):
        return
    key = _normalize_keyword_key(normalized)
    if not key:
        return
    counts[key] += weight
    existing = display_map.get(key)
    if existing is None or _keyword_priority(normalized) < _keyword_priority(existing):
        display_map[key] = normalized


def _normalize_keyword_key(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip().lower())


def _normalize_keyword_display(term: str) -> str:
    cleaned = term.strip().strip("`'\"()[]{}:;,.")
    cleaned = re.sub(r"\.(?:pdf|md|txt)$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<=\D)\d+(?:\.\d+)*$", "", cleaned)
    cleaned = cleaned.strip("-_. ")
    if not cleaned:
        return ""
    if " " in cleaned:
        words = [_normalize_keyword_display(part) for part in cleaned.split()]
        words = [word for word in words if word]
        return " ".join(words)
    if cleaned.upper() == cleaned and len(cleaned) <= 24:
        return cleaned
    return cleaned.lower()


def _is_valid_keyword(term: str) -> bool:
    if not term:
        return False
    normalized = term.strip().lower()
    if not normalized:
        return False
    if normalized in GENERIC_KEYWORD_TOKENS:
        return False
    if re.fullmatch(r"\d+(?:\.\d+)*", normalized):
        return False
    if re.fullmatch(r"page\s+\d+", normalized):
        return False
    if len(normalized) == 1 and normalized not in {"ai"}:
        return False
    words = normalized.split()
    if len(words) > 4:
        return False
    if all(word in GENERIC_KEYWORD_TOKENS for word in words):
        return False
    if any(re.fullmatch(r"\d+(?:\.\d+)*", word) for word in words):
        return False
    return True


def _keyword_priority(term: str) -> tuple[int, int]:
    normalized = term.strip()
    is_acronym = normalized.upper() == normalized and " " not in normalized
    is_phrase = " " in normalized
    return (0 if is_acronym else 1 if is_phrase else 2, len(normalized))


def _sentence_case(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return stripped[0].upper() + stripped[1:]


def _keyword_overlap_key(term: str) -> str:
    words = []
    for word in term.lower().split():
        if len(word) > 4 and word.endswith("s"):
            words.append(word[:-1])
        else:
            words.append(word)
    return " ".join(words)


# Shared deterministic rendering now resolves through the dedicated rendering module.
compose_topic_markdown = _rendering.compose_topic_markdown
compose_topic_index_markdown = _rendering.compose_topic_index_markdown
compose_segment_markdown = _rendering.compose_segment_markdown
describe_documents = _rendering.describe_documents
summarize_text = _rendering.summarize_text
derive_keywords = _rendering.derive_keywords
display_topic_title = _rendering.display_topic_title
topic_overview_sentences = _rendering.topic_overview_sentences
compose_topic_locator = _rendering.compose_topic_locator
compose_key_point = _rendering.compose_key_point
document_outline = _rendering.document_outline
source_name = _rendering.source_name
summarize_sentences = _rendering.summarize_sentences
collect_semantic_headings = _rendering.collect_semantic_headings
clean_source_text = _rendering.clean_source_text
normalize_wiki_sentence = _rendering.normalize_wiki_sentence
natural_join = _rendering.natural_join
_dedupe = _rendering.dedupe_preserve_order
_markdown_table_cell = _rendering.markdown_table_cell
