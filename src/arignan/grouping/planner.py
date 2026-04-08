from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from arignan.models import DocumentSection, ParsedDocument, RetrievalHit

SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


class GroupingDecision(str, Enum):
    STANDALONE = "standalone"
    MERGE = "merge"
    SEGMENT = "segment"


@dataclass(slots=True)
class SegmentPlan:
    slug: str
    title: str
    section_indices: list[int]
    estimated_length: int


@dataclass(slots=True)
class MergeCandidate:
    topic_folder: str
    score: float
    length_estimate: int
    related_chunk_ids: list[str] = field(default_factory=list)
    title: str = ""
    locator: str = ""
    keywords: list[str] = field(default_factory=list)
    description: str = ""
    summary_excerpt: str = ""
    source_count: int = 0


@dataclass(slots=True)
class GroupingHint:
    topic_folder: str
    confidence: float
    rationale: str = ""


@dataclass(slots=True)
class GroupingPlan:
    decision: GroupingDecision
    topic_folder: str
    estimated_length: int
    segments: list[SegmentPlan] = field(default_factory=list)
    merge_target_topic: str | None = None
    related_chunk_ids: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)


class GroupingPlanner:
    def __init__(self, max_md_length: int = 4000, min_merge_score: float = 0.55) -> None:
        self.max_md_length = max_md_length
        self.min_merge_score = min_merge_score

    def plan(
        self,
        document: ParsedDocument,
        related_hits: list[RetrievalHit] | None = None,
        *,
        merge_candidates: list[MergeCandidate] | None = None,
        llm_merge_hint: GroupingHint | None = None,
    ) -> GroupingPlan:
        related_hits = related_hits or []
        estimated_length = estimate_markdown_length(document.full_text)

        if self._should_segment(document, estimated_length):
            segments = self._build_segments(document)
            return GroupingPlan(
                decision=GroupingDecision.SEGMENT,
                topic_folder=derive_topic_folder(document),
                estimated_length=estimated_length,
                segments=segments,
                rationale=[
                    "Estimated markdown exceeds max_md_length or resembles a book-like source.",
                    "Segmented into multiple markdown units for maintainability.",
                ],
            )

        candidates = merge_candidates or self._candidates_from_related_hits(related_hits)
        merge_target = self._best_merge_target(candidates, estimated_length, llm_merge_hint=llm_merge_hint)
        if merge_target is not None:
            topic_folder, candidate_score, candidate_length, related_chunk_ids, hint_rationale = merge_target
            rationale = [
                "Related indexed material suggests semantic overlap with an existing topic folder.",
                f"Combined estimated markdown length ({candidate_length}) stays within max_md_length.",
                f"Aggregate merge evidence score: {candidate_score:.2f}.",
            ]
            if hint_rationale:
                rationale.append(f"Light LLM grouping vote: {hint_rationale}")
            return GroupingPlan(
                decision=GroupingDecision.MERGE,
                topic_folder=topic_folder,
                merge_target_topic=topic_folder,
                estimated_length=estimated_length,
                related_chunk_ids=related_chunk_ids,
                rationale=rationale,
            )

        return GroupingPlan(
            decision=GroupingDecision.STANDALONE,
            topic_folder=derive_topic_folder(document),
            estimated_length=estimated_length,
            rationale=["No suitable merge candidate found and segmentation is unnecessary."],
        )

    def _should_segment(self, document: ParsedDocument, estimated_length: int) -> bool:
        semantic_headings = [section for section in document.sections if section.heading and not _is_page_heading(section.heading)]
        page_sections = [section for section in document.sections if section.page_number is not None]
        page_count = len(page_sections)
        is_compact_pdf = (
            document.source.source_type.value == "pdf"
            and page_count
            and page_count <= 40
            and not semantic_headings
        )
        if is_compact_pdf:
            return False

        is_book_like = document.source.source_type.value == "pdf" and len(semantic_headings) >= 8
        is_very_large = estimated_length > (self.max_md_length * 2)
        return is_book_like or is_very_large

    def _build_segments(self, document: ParsedDocument) -> list[SegmentPlan]:
        sections = document.sections or [DocumentSection(text=document.full_text)]
        segments: list[SegmentPlan] = []
        current_indices: list[int] = []
        current_length = 0

        for index, section in enumerate(sections):
            section_length = estimate_markdown_length(section.text)
            if current_indices and current_length + section_length > self.max_md_length:
                segments.append(self._segment_from_indices(document, current_indices, current_length))
                current_indices = []
                current_length = 0
            current_indices.append(index)
            current_length += section_length

        if current_indices:
            segments.append(self._segment_from_indices(document, current_indices, current_length))

        return segments

    def _segment_from_indices(
        self,
        document: ParsedDocument,
        section_indices: list[int],
        estimated_length: int,
    ) -> SegmentPlan:
        first_section = document.sections[section_indices[0]]
        title = first_section.heading or f"part-{len(section_indices)}"
        return SegmentPlan(
            slug=slugify(title),
            title=title,
            section_indices=list(section_indices),
            estimated_length=estimated_length,
        )

    def _candidates_from_related_hits(self, related_hits: list[RetrievalHit]) -> list[MergeCandidate]:
        candidates: dict[str, MergeCandidate] = {}
        for hit in related_hits:
            topic_folder = hit.metadata.topic_folder
            if not topic_folder:
                continue
            candidate = candidates.setdefault(
                topic_folder,
                MergeCandidate(topic_folder=topic_folder, score=0.0, length_estimate=0),
            )
            candidate.score += hit.score
            candidate.length_estimate = max(
                candidate.length_estimate,
                int(hit.extras.get("topic_length_estimate", estimate_markdown_length(hit.text))),
            )
            candidate.related_chunk_ids.append(hit.chunk_id)
        return sorted(candidates.values(), key=lambda item: item.score, reverse=True)

    def _best_merge_target(
        self,
        candidates: list[MergeCandidate],
        estimated_length: int,
        llm_merge_hint: GroupingHint | None = None,
    ) -> tuple[str, float, int, list[str], str | None] | None:
        best: tuple[str, float, int, list[str], str | None] | None = None
        for candidate in candidates:
            candidate_score = candidate.score
            hint_rationale: str | None = None
            if llm_merge_hint is not None and candidate.topic_folder == llm_merge_hint.topic_folder:
                candidate_score += 0.8 + (max(0.0, min(llm_merge_hint.confidence, 1.0)) * 0.4)
                hint_rationale = llm_merge_hint.rationale or (
                    f"merge with '{candidate.topic_folder}' (confidence {llm_merge_hint.confidence:.2f})"
                )
            candidate_length = estimated_length + candidate.length_estimate
            if candidate_length > self.max_md_length:
                continue
            min_score = 0.2 if hint_rationale else self.min_merge_score
            if candidate_score < min_score:
                continue
            if best is None or candidate_score > best[1]:
                best = (
                    candidate.topic_folder,
                    candidate_score,
                    candidate_length,
                    list(candidate.related_chunk_ids),
                    hint_rationale,
                )
        return best


def estimate_markdown_length(text: str) -> int:
    normalized = " ".join(text.split())
    return max(200, int(len(normalized) * 0.35))


def derive_topic_folder(document: ParsedDocument) -> str:
    title = document.source.title
    if not title:
        for section in document.sections:
            if section.heading:
                title = section.heading
                break
    if not title and document.keywords:
        title = document.keywords[0]
    if not title:
        title = document.source.source_uri.rsplit("/", maxsplit=1)[-1].rsplit("\\", maxsplit=1)[-1]
    return slugify(title)


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    slug = SLUG_PATTERN.sub("-", lowered).strip("-")
    return slug or "topic"


def _is_page_heading(heading: str) -> bool:
    normalized = heading.strip().lower()
    return bool(re.fullmatch(r"page\s+\d+", normalized))
