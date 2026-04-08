from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from arignan.indexing import DenseIndexer, Embedder, HashingEmbedder, LexicalIndex, LexicalIndexer, LocalDenseIndex, tokenize
from arignan.models import ChunkMetadata, RetrievalHit, RetrievalSource
from arignan.storage import StorageLayout
from arignan.tracing import ModelTraceCollector

ABBREVIATIONS = {
    "bm25": "best matching 25",
    "jepa": "joint embedding predictive architecture",
    "llm": "large language model",
    "rag": "retrieval augmented generation",
    "rfic": "rf integrated circuit",
}


@dataclass(slots=True)
class RetrievalBundle:
    query: str
    expanded_query: str
    selected_hat: str
    dense_hits: list[RetrievalHit]
    lexical_hits: list[RetrievalHit]
    map_hits: list[RetrievalHit]
    fused_hits: list[RetrievalHit]


class QueryExpander:
    def expand(self, query: str) -> str:
        normalized_tokens: list[str] = []
        for token in tokenize(query):
            normalized_tokens.append(token)
            if token in ABBREVIATIONS:
                normalized_tokens.extend(tokenize(ABBREVIATIONS[token]))
        return " ".join(normalized_tokens)


class HatSelector:
    def __init__(self, layout: StorageLayout) -> None:
        self.layout = layout

    def select(self, query: str, hat: str = "auto") -> str:
        if hat != "auto":
            return hat
        hats = sorted(path.name for path in self.layout.hats_dir.iterdir() if path.is_dir())
        if not hats:
            return "default"
        if len(hats) == 1:
            return hats[0]

        query_terms = set(tokenize(query))
        best_hat = hats[0]
        best_score = -1
        for candidate in hats:
            map_path = self.layout.hat(candidate).map_path
            text = map_path.read_text(encoding="utf-8") if map_path.exists() else ""
            score = sum(1 for term in query_terms if term in tokenize(text))
            if score > best_score:
                best_score = score
                best_hat = candidate
        return best_hat


class MapRetriever:
    def __init__(self, layout: StorageLayout) -> None:
        self.layout = layout

    def search(self, hat: str, query: str, limit: int) -> list[RetrievalHit]:
        hat_layout = self.layout.hat(hat).ensure()
        query_terms = set(tokenize(query))
        candidates: list[RetrievalHit] = []

        markdown_paths = [hat_layout.map_path]
        markdown_paths.extend(
            path
            for path in hat_layout.summaries_dir.glob("*/markdown_tree/**/*.md")
            if path.name != "map.md"
        )

        for path in markdown_paths:
            if path == hat_layout.map_path:
                topic_folder = None
            elif path.parent.name == "markdown_tree":
                topic_folder = path.parent.parent.name
            else:
                topic_folder = path.parent.name
            sections = split_markdown_sections(path.read_text(encoding="utf-8"))
            for index, (heading, text) in enumerate(sections):
                terms = set(tokenize(f"{heading or ''} {text}"))
                overlap = query_terms & terms
                if not overlap:
                    continue
                score = len(overlap) / max(len(query_terms), 1)
                chunk_id = map_chunk_id(path, index, heading, text)
                candidates.append(
                    RetrievalHit(
                        chunk_id=chunk_id,
                        text=text,
                        score=score,
                        source=RetrievalSource.MAP,
                        metadata=ChunkMetadata(
                            load_id="map",
                            hat=hat,
                            source_uri=str(path),
                            source_path=path,
                            section=heading or path.name,
                            heading=heading,
                            topic_folder=topic_folder,
                            is_map_context=True,
                        ),
                        extras={"overlap_terms": sorted(overlap)},
                    )
                )

        return sorted(candidates, key=lambda item: item.score, reverse=True)[:limit]


class RetrievalPipeline:
    def __init__(
        self,
        layout: StorageLayout,
        embedder: Embedder | None = None,
        dense_limit: int = 8,
        lexical_limit: int = 8,
        map_limit: int = 6,
        fused_limit: int = 10,
        trace_sink: ModelTraceCollector | None = None,
        progress_sink: Callable[[str], None] | None = None,
    ) -> None:
        self.layout = layout
        self.embedder = embedder or HashingEmbedder()
        self.expander = QueryExpander()
        self.selector = HatSelector(layout)
        self.map_retriever = MapRetriever(layout)
        self.dense_limit = dense_limit
        self.lexical_limit = lexical_limit
        self.map_limit = map_limit
        self.fused_limit = fused_limit
        self.trace_sink = trace_sink
        self.progress_sink = progress_sink

    def retrieve(self, query: str, hat: str = "auto") -> RetrievalBundle:
        self._emit_progress("Expanding query...")
        expanded_query = self.expander.expand(query)
        self._emit_progress("Selecting hat...")
        selected_hat = self.selector.select(expanded_query, hat=hat)
        self._emit_progress(f"Hat chosen: {selected_hat}")
        dense = DenseIndexer(
            self.embedder,
            LocalDenseIndex(self.layout.hat(selected_hat).vector_index_dir),
            trace_sink=self.trace_sink,
        )
        lexical = LexicalIndexer(LexicalIndex(self.layout.hat(selected_hat).bm25_index_dir))
        self._emit_progress(f"Searching dense index in hat '{selected_hat}'...")
        dense_hits = dense.search(expanded_query, self.dense_limit)
        self._emit_progress(f"Searching lexical index in hat '{selected_hat}'...")
        lexical_hits = lexical.search(expanded_query, self.lexical_limit)
        self._emit_progress(f"Searching map context in hat '{selected_hat}'...")
        map_hits = self.map_retriever.search(selected_hat, expanded_query, self.map_limit)
        self._emit_progress("Fusing retrieval candidates...")
        fused_hits = reciprocal_rank_fusion(
            [dense_hits, lexical_hits, map_hits],
            limit=self.fused_limit,
        )
        return RetrievalBundle(
            query=query,
            expanded_query=expanded_query,
            selected_hat=selected_hat,
            dense_hits=dense_hits,
            lexical_hits=lexical_hits,
            map_hits=map_hits,
            fused_hits=fused_hits,
        )

    def _emit_progress(self, message: str) -> None:
        if self.progress_sink is not None:
            self.progress_sink(message)


def reciprocal_rank_fusion(result_sets: list[list[RetrievalHit]], limit: int, k: int = 60) -> list[RetrievalHit]:
    fused: dict[str, RetrievalHit] = {}
    scores: dict[str, float] = {}
    channels: dict[str, set[str]] = {}

    for results in result_sets:
        for rank, hit in enumerate(results, start=1):
            key = hit.chunk_id
            scores[key] = scores.get(key, 0.0) + (1.0 / (k + rank))
            channels.setdefault(key, set()).add(hit.source.value)
            if key not in fused:
                fused[key] = hit

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused_hits: list[RetrievalHit] = []
    for key, score in ordered[:limit]:
        hit = fused[key]
        hit.extras["rrf_score"] = score
        hit.extras["channels"] = sorted(channels[key])
        fused_hits.append(hit)
    return fused_hits


def split_markdown_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current_heading, body))

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            marker, _, rest = stripped.partition(" ")
            if marker and set(marker) == {"#"} and rest:
                flush()
                current_heading = rest.strip()
                buffer = []
                continue
        buffer.append(line)
    flush()

    if sections:
        return sections
    plain = text.strip()
    return [(None, plain)] if plain else []


def map_chunk_id(path: Path, index: int, heading: str | None, text: str) -> str:
    digest = hashlib.sha1(f"{path}|{index}|{heading}|{text}".encode("utf-8")).hexdigest()[:12]
    return f"map-{digest}"
