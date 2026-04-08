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
class GroupingPlan:
    decision: GroupingDecision
    topic_folder: str
    estimated_length: int
    segments: list[SegmentPlan] = field(default_factory=list)
    merge_target_topic: str | None = None
    related_chunk_ids: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)


class GroupingPlanner:
    def __init__(self, max_md_length: int = 4000, min_merge_score: float = 0.7) -> None:
        self.max_md_length = max_md_length
        self.min_merge_score = min_merge_score

    def plan(self, document: ParsedDocument, related_hits: list[RetrievalHit] | None = None) -> GroupingPlan:
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

        merge_target = self._best_merge_target(document, related_hits, estimated_length)
        if merge_target is not None:
            topic_folder, candidate_score, candidate_length, related_chunk_ids = merge_target
            return GroupingPlan(
                decision=GroupingDecision.MERGE,
                topic_folder=topic_folder,
                merge_target_topic=topic_folder,
                estimated_length=estimated_length,
                related_chunk_ids=related_chunk_ids,
                rationale=[
                    "Related indexed material suggests semantic overlap with an existing topic folder.",
                    f"Combined estimated markdown length ({candidate_length}) stays within max_md_length.",
                    f"Aggregate merge evidence score: {candidate_score:.2f}.",
                ],
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

    def _best_merge_target(
        self,
        document: ParsedDocument,
        related_hits: list[RetrievalHit],
        estimated_length: int,
    ) -> tuple[str, float, int, list[str]] | None:
        candidates: dict[str, dict[str, object]] = {}
        for hit in related_hits:
            topic_folder = hit.metadata.topic_folder
            if not topic_folder or hit.metadata.source_uri == document.source.source_uri:
                continue
            candidate = candidates.setdefault(
                topic_folder,
                {"score": 0.0, "length": 0, "chunk_ids": []},
            )
            candidate["score"] = float(candidate["score"]) + hit.score
            candidate["length"] = max(
                int(candidate["length"]),
                int(hit.extras.get("topic_length_estimate", estimate_markdown_length(hit.text))),
            )
            candidate["chunk_ids"] = list(candidate["chunk_ids"]) + [hit.chunk_id]

        best: tuple[str, float, int, list[str]] | None = None
        for topic_folder, details in candidates.items():
            candidate_score = float(details["score"])
            candidate_length = estimated_length + int(details["length"])
            related_chunk_ids = list(details["chunk_ids"])
            if candidate_score < self.min_merge_score:
                continue
            if candidate_length > self.max_md_length:
                continue
            if best is None or candidate_score > best[1]:
                best = (topic_folder, candidate_score, candidate_length, related_chunk_ids)
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
