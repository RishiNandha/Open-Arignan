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
from arignan.prompts import DEFAULT_PROMPT_SET, PromptSet, render_prompt_template
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
    prompts: PromptSet = DEFAULT_PROMPT_SET
    trace_sink: ModelTraceCollector | None = None
    progress_sink: Callable[[str], None] | None = None
    exception_logger: SessionExceptionLogger | None = None

    def render_topic(self, documents: list[ParsedDocument], plan: GroupingPlan) -> TopicRender:
        fallback = self.fallback.render_topic(documents, plan)
        prompt = _build_topic_prompt(documents, plan, fallback, template=self.prompts.topic_user_template)
        try:
            self._emit_progress(f"Calling local LLM for topic summary markdown ({self.generator.model_name})...")
            raw = self.generator.generate(
                system_prompt=self.prompts.topic_system_prompt,
                user_prompt=prompt,
                max_new_tokens=900,
                temperature=0.1,
                response_format=TOPIC_RESPONSE_SCHEMA,
            )
            payload = _coerce_topic_payload(raw, fallback=fallback, documents=documents)
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
                system_prompt=self.prompts.hat_map_system_prompt,
                user_prompt=_build_hat_map_prompt(hat, entries, template=self.prompts.hat_map_user_template),
                max_new_tokens=360,
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
                system_prompt=self.prompts.global_map_system_prompt,
                user_prompt=_build_global_map_prompt(entries, template=self.prompts.global_map_user_template),
                max_new_tokens=360,
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


def _build_topic_prompt(
    documents: list[ParsedDocument],
    plan: GroupingPlan,
    fallback: TopicRender,
    *,
    template: str = DEFAULT_PROMPT_SET.topic_user_template,
) -> str:
    related_threads = topic_related_threads(documents, limit=4)
    related_threads_block = "\n".join(f"- {item}" for item in related_threads) or "- No strong related-thread cues detected."
    document_context_lines: list[str] = []
    for index, document in enumerate(documents, start=1):
        document_context_lines.extend(_document_digest_lines(document, index=index))
    return render_prompt_template(
        "topic_user_template",
        template,
        topic_folder=plan.topic_folder,
        suggested_title=fallback.title,
        grouping_decision=plan.decision.value,
        source_count=str(len(documents)),
        related_threads_block=related_threads_block,
        document_context_block="\n".join(document_context_lines),
    )


def _build_hat_map_prompt(
    hat: str,
    entries: list[TopicMapEntry],
    *,
    template: str = DEFAULT_PROMPT_SET.hat_map_user_template,
) -> str:
    lines: list[str] = []
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
    return render_prompt_template(
        "hat_map_user_template",
        template,
        hat=hat,
        topic_entries_block="\n".join(lines),
    )


def _build_global_map_prompt(
    entries: list[HatMapEntry],
    *,
    template: str = DEFAULT_PROMPT_SET.global_map_user_template,
) -> str:
    lines: list[str] = []
    for entry in entries:
        lines.extend(
            [
                f"- Hat: {entry.hat}",
                f"  Map path: {entry.map_path}",
                f"  What to find: {entry.what_to_find}",
                f"  High-level keywords: {', '.join(entry.keywords) or '-'}",
            ]
        )
    return render_prompt_template(
        "global_map_user_template",
        template,
        hat_entries_block="\n".join(lines),
    )


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


def _coerce_topic_payload(text: str, *, fallback: TopicRender, documents: list[ParsedDocument]) -> dict[str, object]:
    try:
        return _extract_json_payload(text)
    except (ValueError, json.JSONDecodeError):
        markdown = _normalize_summary_markdown(
            _normalize_markdown_output(text),
            title=fallback.title,
            fallback="",
            documents=documents,
        )
        if not markdown:
            raise
        return {
            "title": _extract_markdown_title(markdown) or fallback.title,
            "description": fallback.description,
            "locator": fallback.locator,
            "keywords": fallback.keywords,
            "summary_markdown": markdown,
        }


def _extract_markdown_title(markdown: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


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
