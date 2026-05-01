from __future__ import annotations

from dataclasses import dataclass
import re


GRAPH_STOPWORDS = {
    "a",
    "an",
    "and",
    "approach",
    "architecture",
    "background",
    "by",
    "chapter",
    "concept",
    "default",
    "design",
    "discussion",
    "document",
    "embedding",
    "for",
    "from",
    "idea",
    "implementation",
    "in",
    "introduction",
    "learning",
    "method",
    "model",
    "note",
    "notes",
    "of",
    "on",
    "overview",
    "paper",
    "reference",
    "report",
    "result",
    "section",
    "study",
    "summary",
    "system",
    "technique",
    "the",
    "to",
    "topic",
    "training",
    "use",
    "using",
    "with",
    "work",
}
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


@dataclass(slots=True)
class TopicGraphEntry:
    topic_folder: str
    title: str
    locator: str
    description: str
    keywords: list[str]
    summary_excerpt: str


def build_topic_graph(
    entries: list[TopicGraphEntry],
    *,
    limit_per_topic: int = 4,
    min_confidence: float = 0.34,
) -> dict[str, list[dict[str, object]]]:
    relations: dict[str, list[dict[str, object]]] = {entry.topic_folder: [] for entry in entries}
    for index, left in enumerate(entries):
        for right in entries[index + 1 :]:
            relation = _score_relation(left, right, min_confidence=min_confidence)
            if relation is None:
                continue
            left_item, right_item = relation
            relations[left.topic_folder].append(left_item)
            relations[right.topic_folder].append(right_item)

    for topic_folder, items in relations.items():
        items.sort(key=lambda item: (float(item["confidence"]), item["title"]), reverse=True)
        relations[topic_folder] = items[:limit_per_topic]
    return relations


def _score_relation(
    left: TopicGraphEntry,
    right: TopicGraphEntry,
    *,
    min_confidence: float,
) -> tuple[dict[str, object], dict[str, object]] | None:
    left_keywords = {_normalize_term(keyword) for keyword in left.keywords if _normalize_term(keyword)}
    right_keywords = {_normalize_term(keyword) for keyword in right.keywords if _normalize_term(keyword)}
    shared_keywords = sorted(left_keywords & right_keywords)

    left_title_terms = _signal_terms(" ".join([left.title, left.locator]))
    right_title_terms = _signal_terms(" ".join([right.title, right.locator]))
    shared_title_terms = sorted(left_title_terms & right_title_terms)

    left_summary_terms = _signal_terms(f"{left.description} {left.summary_excerpt}")
    right_summary_terms = _signal_terms(f"{right.description} {right.summary_excerpt}")
    shared_summary_terms = sorted((left_summary_terms & right_summary_terms) - set(shared_title_terms))

    confidence = 0.0
    confidence += min(len(shared_keywords), 3) * 0.24
    confidence += min(len(shared_title_terms), 3) * 0.12
    confidence += min(len(shared_summary_terms), 4) * 0.06
    if shared_keywords and shared_title_terms:
        confidence += 0.08
    if len(shared_keywords) >= 2:
        confidence += 0.06
    confidence = min(confidence, 0.95)

    if confidence < min_confidence:
        return None

    relation_type = "EXTRACTED" if shared_keywords or len(shared_title_terms) >= 2 else "INFERRED"
    shared_terms = _display_terms(shared_keywords, shared_title_terms, shared_summary_terms)
    rationale = _relation_rationale(shared_keywords, shared_title_terms, shared_summary_terms)

    left_item = {
        "topic_folder": right.topic_folder,
        "title": right.title,
        "confidence": round(confidence, 2),
        "relation_type": relation_type,
        "shared_terms": shared_terms,
        "rationale": rationale,
    }
    right_item = {
        "topic_folder": left.topic_folder,
        "title": left.title,
        "confidence": round(confidence, 2),
        "relation_type": relation_type,
        "shared_terms": shared_terms,
        "rationale": rationale,
    }
    return left_item, right_item


def _display_terms(
    shared_keywords: list[str],
    shared_title_terms: list[str],
    shared_summary_terms: list[str],
) -> list[str]:
    ordered = [*shared_keywords, *shared_title_terms, *shared_summary_terms]
    display: list[str] = []
    seen: set[str] = set()
    for term in ordered:
        normalized = term.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        display.append(term)
        if len(display) >= 6:
            break
    return display


def _relation_rationale(
    shared_keywords: list[str],
    shared_title_terms: list[str],
    shared_summary_terms: list[str],
) -> str:
    if shared_keywords:
        return f"Shared keywords: {', '.join(shared_keywords[:4])}."
    if shared_title_terms:
        return f"Shared title or locator terms: {', '.join(shared_title_terms[:4])}."
    return f"Shared summary language: {', '.join(shared_summary_terms[:4])}."


def _signal_terms(text: str) -> set[str]:
    return {
        token
        for token in (_normalize_term(match.group(0)) for match in TOKEN_PATTERN.finditer(text))
        if token and token not in GRAPH_STOPWORDS
    }


def _normalize_term(value: str) -> str:
    normalized = value.strip().lower().strip("-_. ")
    if not normalized or normalized in GRAPH_STOPWORDS:
        return ""
    if normalized.endswith("s") and len(normalized) > 5:
        normalized = normalized[:-1]
    return normalized
