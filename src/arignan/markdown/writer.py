from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from arignan.grouping import GroupingPlan
from arignan.llm import LocalTextGenerator
from arignan.markdown.rendering import (
    compose_document_summary,
    compose_topic_locator,
    compose_topic_markdown,
    topic_related_threads,
    markdown_table_cell,
    describe_documents,
    derive_keywords,
    display_topic_title,
    document_outline,
    natural_join,
    source_name,
    summarize_text,
    topic_overview_sentences,
    collect_semantic_headings,
)
from arignan.models import ParsedDocument
from arignan.session.exception_log import SessionExceptionLogger
from arignan.tracing import ModelTraceCollector


@dataclass(slots=True)
class TopicRender:
    title: str
    description: str
    locator: str
    keywords: list[str]
    summary_markdown: str


@dataclass(slots=True)
class TopicMapEntry:
    topic_folder: str
    title: str
    locator: str
    source_files: list[str]
    markdown_files: list[str]
    keywords: list[str]


@dataclass(slots=True)
class HatMapEntry:
    hat: str
    map_path: str
    what_to_find: str
    keywords: list[str]


class ArtifactWriter(Protocol):
    def render_topic(self, documents: list[ParsedDocument], plan: GroupingPlan) -> TopicRender:
        """Render a topic summary and metadata."""

    def render_hat_map(self, hat: str, entries: list[TopicMapEntry]) -> str:
        """Render map.md for a hat."""

    def render_global_map(self, entries: list[HatMapEntry]) -> str:
        """Render global_map.md across hats."""


@dataclass(slots=True)
class HeuristicArtifactWriter:
    def render_topic(self, documents: list[ParsedDocument], plan: GroupingPlan) -> TopicRender:
        title = display_topic_title(plan.topic_folder, documents)
        description = describe_documents(documents)
        locator = compose_topic_locator(documents)
        keywords = derive_keywords(documents)
        return TopicRender(
            title=title,
            description=description,
            locator=locator,
            keywords=keywords,
            summary_markdown=compose_topic_markdown(documents, plan),
        )

    def render_hat_map(self, hat: str, entries: list[TopicMapEntry]) -> str:
        lines = [
            f"# Map for Hat: {hat}",
            "",
            "| Topic | Directory | What To Find | Source Files | Keywords |",
            "| --- | --- | --- | --- | --- |",
        ]
        for entry in entries:
            lines.append(
                "| "
                + " | ".join(
                    [
                        markdown_table_cell(entry.title),
                        f"`{Path('summaries') / entry.topic_folder}`".replace("\\", "/"),
                        markdown_table_cell(entry.locator),
                        markdown_table_cell(", ".join(entry.source_files) or "-"),
                        markdown_table_cell(", ".join(entry.keywords) or "-"),
                    ]
                )
                + " |"
            )
        return "\n".join(lines).rstrip() + "\n"

    def render_global_map(self, entries: list[HatMapEntry]) -> str:
        lines = [
            "# Global Map",
            "",
            "| Hat | Map Path | What To Find | High-Level Keywords |",
            "| --- | --- | --- | --- |",
        ]
        for entry in entries:
            lines.append(
                "| "
                + " | ".join(
                    [
                        markdown_table_cell(entry.hat),
                        f"`{entry.map_path.replace(chr(92), '/')}`",
                        markdown_table_cell(entry.what_to_find),
                        markdown_table_cell(", ".join(entry.keywords) or "-"),
                    ]
                )
                + " |"
            )
        return "\n".join(lines).rstrip() + "\n"


@dataclass(slots=True)
class LLMArtifactWriter:
    generator: LocalTextGenerator
    fallback: ArtifactWriter
    trace_sink: ModelTraceCollector | None = None
    progress_sink: Callable[[str], None] | None = None
    exception_logger: SessionExceptionLogger | None = None

    def render_topic(self, documents: list[ParsedDocument], plan: GroupingPlan) -> TopicRender:
        fallback = self.fallback.render_topic(documents, plan)
        prompt = _build_topic_prompt(documents, plan, fallback)
        try:
            self._emit_progress(f"Calling local LLM for topic summary markdown ({self.generator.model_name})...")
            raw = self.generator.generate(
                system_prompt=TOPIC_SYSTEM_PROMPT,
                user_prompt=prompt,
                max_new_tokens=1100,
                temperature=0.1,
                response_format=TOPIC_RESPONSE_SCHEMA,
            )
            payload = _extract_json_payload(raw)
        except Exception as exc:
            log_path = self._log_exception(
                task="topic summary markdown",
                exc=exc,
                context={"topic_folder": plan.topic_folder, "document_count": len(documents)},
            )
            self._emit_progress(self._fallback_message("topic summary markdown", log_path))
            self._record(
                task="topic summary markdown",
                status="fallback",
                item_count=len(documents),
                detail=plan.topic_folder,
            )
            return fallback

        title = _coerce_text(payload.get("title")) or fallback.title
        description = _coerce_text(payload.get("description")) or fallback.description
        locator = _coerce_text(payload.get("locator")) or fallback.locator
        keywords = _coerce_keywords(payload.get("keywords")) or fallback.keywords
        raw_summary_markdown = _coerce_text(payload.get("summary_markdown")) or fallback.summary_markdown
        summary_markdown = _normalize_summary_markdown(
            raw_summary_markdown,
            title=title,
            fallback=fallback.summary_markdown,
            documents=documents,
        )
        status = "ok" if summary_markdown != fallback.summary_markdown or raw_summary_markdown == fallback.summary_markdown else "fallback"
        self._record(
            task="topic summary markdown",
            status=status,
            item_count=len(documents),
            detail=plan.topic_folder,
        )
        return TopicRender(
            title=title,
            description=description,
            locator=locator,
            keywords=keywords,
            summary_markdown=summary_markdown,
        )

    def render_hat_map(self, hat: str, entries: list[TopicMapEntry]) -> str:
        fallback = self.fallback.render_hat_map(hat, entries)
        if not entries:
            self._record(
                task="hat map markdown",
                status="skipped",
                item_count=0,
                detail=f"{hat} (empty)",
            )
            return fallback
        try:
            self._emit_progress(f"Calling local LLM for hat map markdown ({self.generator.model_name})...")
            generated = self.generator.generate(
                system_prompt=HAT_MAP_SYSTEM_PROMPT,
                user_prompt=_build_hat_map_prompt(hat, entries),
                max_new_tokens=700,
                temperature=0.1,
            )
        except Exception as exc:
            log_path = self._log_exception(
                task="hat map markdown",
                exc=exc,
                context={"hat": hat, "entry_count": len(entries)},
            )
            self._emit_progress(self._fallback_message("hat map markdown", log_path))
            self._record(
                task="hat map markdown",
                status="fallback",
                item_count=len(entries),
                detail=hat,
            )
            return fallback
        normalized = _normalize_markdown_output(generated)
        if normalized.startswith("# Map for Hat:"):
            self._record(
                task="hat map markdown",
                status="ok",
                item_count=len(entries),
                detail=hat,
            )
            return normalized
        self._record(
            task="hat map markdown",
            status="fallback",
            item_count=len(entries),
            detail=f"{hat} (invalid output)",
        )
        return fallback

    def render_global_map(self, entries: list[HatMapEntry]) -> str:
        fallback = self.fallback.render_global_map(entries)
        if not entries:
            self._record(
                task="global map markdown",
                status="skipped",
                item_count=0,
                detail="0 hat(s)",
            )
            return fallback
        try:
            self._emit_progress(f"Calling local LLM for global map markdown ({self.generator.model_name})...")
            generated = self.generator.generate(
                system_prompt=GLOBAL_MAP_SYSTEM_PROMPT,
                user_prompt=_build_global_map_prompt(entries),
                max_new_tokens=700,
                temperature=0.1,
            )
        except Exception as exc:
            log_path = self._log_exception(
                task="global map markdown",
                exc=exc,
                context={"hat_count": len(entries)},
            )
            self._emit_progress(self._fallback_message("global map markdown", log_path))
            self._record(
                task="global map markdown",
                status="fallback",
                item_count=len(entries),
                detail=f"{len(entries)} hat(s)",
            )
            return fallback
        normalized = _normalize_markdown_output(generated)
        if normalized.startswith("# Global Map"):
            self._record(
                task="global map markdown",
                status="ok",
                item_count=len(entries),
                detail=f"{len(entries)} hat(s)",
            )
            return normalized
        self._record(
            task="global map markdown",
            status="fallback",
            item_count=len(entries),
            detail="invalid output",
        )
        return fallback

    def _record(self, *, task: str, status: str, item_count: int | None, detail: str | None) -> None:
        if self.trace_sink is None:
            return
        self.trace_sink.record(
            component="llm",
            task=task,
            model_name=getattr(self.generator, "model_name", type(self.generator).__name__),
            backend=getattr(self.generator, "backend_name", type(self.generator).__name__),
            status=status,
            item_count=item_count,
            detail=detail,
        )

    def _emit_progress(self, message: str) -> None:
        if self.progress_sink is not None:
            self.progress_sink(message)

    def _log_exception(self, *, task: str, exc: BaseException, context: dict[str, object]) -> Path | None:
        if self.exception_logger is None:
            return None
        return self.exception_logger.log_exception(
            component="llm",
            task=task,
            exc=exc,
            context=context,
        )

    @staticmethod
    def _fallback_message(task: str, log_path: Path | None) -> str:
        message = f"Local LLM unavailable for {task}; using fallback renderer."
        if log_path is None:
            return message
        return f"{message} Log: {log_path.resolve()}"


TOPIC_SYSTEM_PROMPT = """You write compact knowledge-base markdown for a local private wiki.
Return strict JSON only, with no code fences and no commentary.
The markdown must be neatly rewritten for human auditability.
Never mention chunks, extraction, parsing, prompt instructions, or the existence of an LLM.
Preserve technical acronyms and paper names exactly when possible.
Use a neutral, reference-style voice like a concise internal wiki page.
Prefer definition-first lead sentences, compact explanatory paragraphs, and crisp bullets.
Write each page as a durable lookup surface for later LLM retrieval, not as a compressed abstract.
Make conceptual relationships, adjacent ideas, and source-to-source connections easy to scan.
When sources are grouped, rewrite them into one coherent article rather than a pile of per-source mini-summaries."""

HAT_MAP_SYSTEM_PROMPT = """You write concise lookup-table markdown for a knowledge-base hat map.
Return markdown only.
Keep it compact and scannable.
Do not add prose paragraphs before or after the table."""

GLOBAL_MAP_SYSTEM_PROMPT = """You write concise lookup-table markdown for a global knowledge-base map.
Return markdown only.
Keep it compact and scannable.
Do not add prose paragraphs before or after the table."""


def _build_topic_prompt(documents: list[ParsedDocument], plan: GroupingPlan, fallback: TopicRender) -> str:
    related_threads = topic_related_threads(documents, limit=4)
    lines = [
        "Task: write a clean wiki-style knowledge-base page for the topic below.",
        "The page should act like a rich lookup article for future retrieval, not just a short summary.",
        "When multiple sources are grouped, draw clear lines between adjacent ideas, complementary angles, and recurring themes.",
        "Write it as one coherent topic page that helps a future reader or LLM quickly orient, retrieve, and connect ideas.",
        "",
        "Topic metadata:",
        f"- Topic folder: {plan.topic_folder}",
        f"- Suggested title: {fallback.title}",
        f"- Grouping decision: {plan.decision.value}",
        f"- Source count: {len(documents)}",
        "",
        "Return JSON with exactly these keys:",
        '- "title": short topic title',
        '- "description": one concise sentence describing what the topic covers',
        '- "locator": a short "what to find here" phrase for map.md',
        '- "keywords": 4 to 8 specific technical keywords or phrases',
        '- "summary_markdown": wiki-style markdown only',
        "",
        "Writing rules for summary_markdown:",
        "- Start with '# <title>'",
        "- Then one short lead paragraph that defines the topic immediately and reads like an internal wiki article",
        "- Then '## Summary' with one short paragraph that explains scope, significance, and why grouped sources belong together",
        "- Then '## Key Ideas' with 3 to 6 bullets that rewrite the ideas cleanly instead of copying source sentences",
        "- Then '## Related Threads' with 3 to 6 bullets that connect adjacent ideas, subthemes, contrasts, dependencies, extensions, or useful lookup paths inside this topic",
        "- Then '## Sources' with this exact table header:",
        "  | Source | What To Find | Key Sections | File |",
        "- Then '## Keywords' with a comma-separated line",
        "- Keep it concise, readable, and neutral",
        "- Make the markdown useful as a future lookup page for an LLM that will retrieve this topic later",
        "- If several sources are grouped, explain how they fit together instead of summarizing each source in isolation",
        "- Make each section feel like a stable reference entry, not reading notes or extracted chunks",
        "- Prefer declarative sentences over promotional or first-person phrasing",
        "- Do not paste raw chunks or long quotations",
        "- Do not mention page numbers unless they are semantically important",
        "- Keywords must not include generic junk like page, section, paper, notes, method, work, or standalone digits",
        "",
        "Bad patterns to avoid:",
        "- Do not say 'this document discusses' or 'these notes contain' unless unavoidable",
        "- Do not sound like an extraction pipeline or a paper abstract pasted verbatim",
        "- Do not write one bullet per source file when the topic is clearly unified",
        "- Do not produce fragmented bullets where a paragraph would be clearer",
        "",
        "Example of a good summary_markdown shape:",
        "# Temporal Sparse Attention",
        "",
        "Temporal Sparse Attention is an attention strategy that focuses computation on selected time-local interactions.",
        "",
        "## Summary",
        "This page covers the main idea behind temporal sparsity, the practical tradeoff it makes, and the kinds of sequence-modeling tasks where it is useful.",
        "",
        "## Key Ideas",
        "- Restricts attention computation to selected temporal neighborhoods instead of every pairwise interaction.",
        "- Trades full-context coverage for lower compute and clearer locality bias.",
        "- Commonly appears in discussions of efficient long-sequence modeling and event streams.",
        "",
        "## Related Threads",
        "- Closely tied to sparse attention patterns, event-based sequence modeling, and efficient temporal context handling.",
        "- Often contrasted with dense attention because it prioritizes selective connectivity over uniform global coverage.",
        "- Useful for connecting architecture choices, compute tradeoffs, and downstream sequence behavior within one page.",
        "- Serves as a bridge page when questions move between model efficiency, temporal structure, and representation quality.",
        "",
        "## Sources",
        "| Source | What To Find | Key Sections | File |",
        "| --- | --- | --- | --- |",
        "| Sparse Attention Notes | Core idea and tradeoffs | Overview, Tradeoffs | `notes.md` |",
        "",
        "## Keywords",
        "temporal sparse attention, efficient sequence modeling, event stream, locality bias",
        "",
        "Helpful related-thread cues for this topic:",
    ]
    for item in related_threads:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
        "Topic context:",
        ]
    )
    for index, document in enumerate(documents, start=1):
        lines.extend(_document_digest_lines(document, index=index))
    return "\n".join(lines)


def _build_hat_map_prompt(hat: str, entries: list[TopicMapEntry]) -> str:
    lines = [
        f"Write map.md for the hat '{hat}'.",
        "Return markdown only.",
        "Use exactly this compact table layout:",
        "# Map for Hat: <hat>",
        "",
        "| Topic | Directory | What To Find | Source Files | Keywords |",
        "| --- | --- | --- | --- | --- |",
        "",
        "Topic entries:",
    ]
    for entry in entries:
        lines.extend(
            [
                f"- Topic: {entry.title}",
                f"  Directory: summaries/{entry.topic_folder}",
                f"  What to find: {entry.locator}",
                f"  Source files: {', '.join(entry.source_files) or '-'}",
                f"  Keywords: {', '.join(entry.keywords) or '-'}",
            ]
        )
    return "\n".join(lines)


def _build_global_map_prompt(entries: list[HatMapEntry]) -> str:
    lines = [
        "Write global_map.md for the knowledge base.",
        "Return markdown only.",
        "Use exactly this compact table layout:",
        "# Global Map",
        "",
        "| Hat | Map Path | What To Find | High-Level Keywords |",
        "| --- | --- | --- | --- |",
        "",
        "Hat entries:",
    ]
    for entry in entries:
        lines.extend(
            [
                f"- Hat: {entry.hat}",
                f"  Map path: {entry.map_path}",
                f"  What to find: {entry.what_to_find}",
                f"  High-level keywords: {', '.join(entry.keywords) or '-'}",
            ]
        )
    return "\n".join(lines)


def _document_digest_lines(document: ParsedDocument, index: int) -> list[str]:
    headings = collect_semantic_headings([document], limit=5)
    overview = topic_overview_sentences([document], limit=3)
    keywords = derive_keywords([document], limit=6)
    source_ref = source_name(document)
    lines = [
        f"Document {index}:",
        f"- Title: {document.source.title or source_ref}",
        f"- File: {source_ref}",
        f"- Type: {document.source.source_type.value}",
        f"- Structure: {natural_join(headings) if headings else document_outline(document)}",
        f"- Keywords: {', '.join(keywords) if keywords else 'none'}",
        f"- Contribution to topic page: {compose_document_summary(document)}",
        "- Key material:",
    ]
    material = overview or [summarize_text(document.full_text, max_length=280)]
    for sentence in material[:3]:
        lines.append(f"  - {summarize_text(sentence, max_length=220)}")
    return lines


def _extract_json_payload(text: str) -> dict[str, object]:
    normalized = text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", normalized, re.DOTALL)
    if fenced_match:
        normalized = fenced_match.group(1).strip()
    start = normalized.find("{")
    end = normalized.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no json object found in llm output")
    return json.loads(normalized[start : end + 1])


def _normalize_summary_markdown(markdown: str, *, title: str, fallback: str, documents: list[ParsedDocument]) -> str:
    normalized = _normalize_markdown_output(markdown)
    if not normalized:
        return fallback
    if not normalized.startswith("# "):
        normalized = f"# {title}\n\n{normalized}"
    required_sections = ("## Summary", "## Key Ideas", "## Sources", "## Keywords")
    if any(section not in normalized for section in required_sections):
        return fallback
    if "## Related Threads" not in normalized:
        threads = topic_related_threads(documents, limit=4)
        if threads:
            insertion = ["## Related Threads", *[f"- {item}" for item in threads], ""]
            if "## Sources" in normalized:
                normalized = normalized.replace("## Sources", "\n".join(insertion) + "## Sources", 1)
    return normalized


def _normalize_markdown_output(markdown: str) -> str:
    normalized = markdown.strip()
    fenced = re.match(r"```(?:markdown)?\s*(.*?)\s*```$", normalized, re.DOTALL)
    if fenced:
        normalized = fenced.group(1).strip()
    return normalized.rstrip() + "\n" if normalized else ""


def _coerce_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _coerce_keywords(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        keyword = item.strip()
        if not keyword:
            continue
        if re.fullmatch(r"\d+(?:\.\d+)*", keyword):
            continue
        if keyword.lower() in {"page", "pages", "section", "sections", "paper", "papers", "notes", "work", "method"}:
            continue
        cleaned.append(keyword)
    deduped: list[str] = []
    seen: set[str] = set()
    for keyword in cleaned:
        key = keyword.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(keyword)
    return deduped[:8]


TOPIC_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "locator": {"type": "string"},
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 4,
            "maxItems": 8,
        },
        "summary_markdown": {"type": "string"},
    },
    "required": ["title", "description", "locator", "keywords", "summary_markdown"],
}
