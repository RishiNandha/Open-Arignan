from __future__ import annotations

import json
from pathlib import Path

from arignan.application import (
    ArignanApp,
    TopicGroupingRecord,
    _build_grouping_review_prompt,
    _parse_grouping_review,
    compose_answer,
    format_citation,
    generate_answer,
    render_raw_hits,
    synthesize_answer,
)
from arignan.config import load_config
from arignan.grouping import GroupingDecision, GroupingPlan
from arignan.models import (
    ChatTurn,
    ChunkMetadata,
    DocumentSection,
    LoadEvent,
    LoadOperation,
    ParsedDocument,
    RetrievalHit,
    RetrievalSource,
    SourceDocument,
    SourceType,
)
from arignan.retrieval import RetrievalBundle
from arignan.session import SessionExceptionLogger, SessionStore
from arignan.tracing import ModelTraceCollector


class FakeGenerator:
    model_name = "fake-llm"
    backend_name = "fake-backend"

    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[tuple[str, str]] = []

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 800,
        temperature: float = 0.1,
        response_format=None,
    ) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.output


class FailingGenerator:
    model_name = "fake-llm"
    backend_name = "fake-backend"

    def __init__(self, message: str) -> None:
        self.message = message

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 800,
        temperature: float = 0.1,
        response_format=None,
    ) -> str:
        raise RuntimeError(self.message)


class GroupingReviewGenerator:
    model_name = "qwen3:4b-q4_K_M"
    backend_name = "fake-backend"

    def __init__(self, target_topic_folder: str) -> None:
        self.target_topic_folder = target_topic_folder

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 800,
        temperature: float = 0.1,
        response_format=None,
    ) -> str:
        return json.dumps(
            {
                "recommendations": [
                    {
                        "members": ["latent-prediction-training", self.target_topic_folder],
                        "target_topic_folder": self.target_topic_folder,
                        "confidence": 0.81,
                        "rationale": "same conceptual family",
                    }
                ]
            }
        )


class ArtifactGenerator:
    backend_name = "fake-backend"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 800,
        temperature: float = 0.1,
        response_format=None,
    ) -> str:
        suggested_title = "Topic"
        for line in user_prompt.splitlines():
            if line.startswith("- Suggested title: "):
                suggested_title = line.split(": ", maxsplit=1)[1].strip() or "Topic"
                break
        if "Return strict JSON only" in system_prompt:
            return json.dumps(
                {
                    "title": suggested_title,
                    "description": "Topic description.",
                    "locator": "topic lookup",
                    "keywords": ["spiking neural networks", "sequence model", "topic", "lookup"],
                    "summary_markdown": (
                        f"# {suggested_title}\n\n"
                        "A concise topic page.\n\n"
                        "## Summary\n"
                        "Short summary.\n\n"
                        "## Key Ideas\n"
                        "- First idea\n"
                        "- Second idea\n\n"
                        "## Sources\n"
                        "| Source | What To Find | Key Sections | File |\n"
                        "| --- | --- | --- | --- |\n"
                        f"| {suggested_title} | Short summary | Overview | `notes.md` |\n\n"
                        "## Keywords\n"
                        "spiking neural networks, sequence model, topic, lookup"
                    ),
                }
            )
        if "knowledge-base hat map" in system_prompt:
            return (
                "# Map for Hat: default\n\n"
                "| Topic | Directory | What To Find | Source Files | Keywords |\n"
                "| --- | --- | --- | --- | --- |\n"
            )
        if "global knowledge-base map" in system_prompt:
            return (
                "# Global Map\n\n"
                "| Hat | Map Path | What To Find | High-Level Keywords |\n"
                "| --- | --- | --- | --- |\n"
            )
        return "Answer."


class PostLoadRegroupGenerator:
    model_name = "qwen3:4b-q4_K_M"
    backend_name = "fake-backend"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 800,
        temperature: float = 0.1,
        response_format=None,
    ) -> str:
        self.calls.append(user_prompt)
        if "review topic pages inside one local research-wiki hat" in system_prompt.lower():
            if "Title: First Topic" in user_prompt and "Title: Second Topic" in user_prompt:
                return json.dumps(
                    {
                        "recommendations": [
                            {
                                "members": ["first-topic", "second-topic"],
                                "target_topic_folder": "second-topic",
                                "confidence": 0.91,
                                "rationale": "the first topic fits better as part of the second topic page",
                            }
                        ]
                    }
                )
            return json.dumps({"recommendations": []})
        if "Return strict JSON only" in system_prompt:
            suggested_title = "Topic"
            for line in user_prompt.splitlines():
                if line.startswith("- Suggested title: "):
                    suggested_title = line.split(": ", maxsplit=1)[1].strip() or "Topic"
                    break
            return json.dumps(
                {
                    "title": suggested_title,
                    "description": "Topic description.",
                    "locator": "topic lookup",
                    "keywords": ["topic", "lookup"],
                    "summary_markdown": (
                        f"# {suggested_title}\n\n"
                        "A concise topic page.\n\n"
                        "## Summary\n"
                        "Short summary.\n\n"
                        "## Key Ideas\n"
                        "- First idea\n"
                        "- Second idea\n\n"
                        "## Sources\n"
                        "| Source | What To Find | Key Sections | File |\n"
                        "| --- | --- | --- | --- |\n"
                        f"| {suggested_title} | Short summary | Overview | `notes.md` |\n\n"
                        "## Keywords\n"
                        "topic, lookup"
                    ),
                }
            )
        if "knowledge-base hat map" in system_prompt:
            return (
                "# Map for Hat: default\n\n"
                "| Topic | Directory | What To Find | Source Files | Keywords |\n"
                "| --- | --- | --- | --- | --- |\n"
            )
        if "global knowledge-base map" in system_prompt:
            return (
                "# Global Map\n\n"
                "| Hat | Map Path | What To Find | High-Level Keywords |\n"
                "| --- | --- | --- | --- |\n"
            )
        return "Answer."


class EmptyCrossEncoderReranker:
    model_name = "mixedbread-ai/mxbai-rerank-base-v1"
    backend_name = "cross-encoder"

    def rerank(self, query: str, hits: list[RetrievalHit], limit: int, min_score: float = 0.0) -> list[RetrievalHit]:
        return []


def _hit(text: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id="chunk-1",
        text=text,
        score=0.9,
        source=RetrievalSource.DENSE,
        metadata=ChunkMetadata(
            load_id="load-1",
            hat="default",
            source_uri="notes.md",
            source_path=Path("notes.md"),
            heading="Overview",
            section="Overview",
            topic_folder="jepa-notes",
        ),
        extras={"rerank_score": 0.93},
    )


def test_format_citation_includes_hat_topic_file_and_location() -> None:
    hit = RetrievalHit(
        chunk_id="chunk-1",
        text="Joint embedding predictive architecture learns representations from context.",
        score=0.9,
        source=RetrievalSource.DENSE,
        metadata=ChunkMetadata(
            load_id="load-1",
            hat="research",
            source_uri="paper.pdf",
            source_path=Path("paper.pdf"),
            page_number=12,
            heading="Training Objective",
            section="Training Objective",
            topic_folder="jepa-paper",
        ),
    )

    citation = format_citation(hit)

    assert citation == "research/jepa-paper/paper.pdf: Page 12, Training Objective"


def test_synthesize_answer_avoids_heading_only_output() -> None:
    hit = RetrievalHit(
        chunk_id="chunk-2",
        text=(
            "Joint embedding predictive architecture learns representations without rebuilding every input. "
            "It predicts latent targets from context so the model focuses on semantics."
        ),
        score=0.8,
        source=RetrievalSource.DENSE,
        metadata=ChunkMetadata(
            load_id="load-2",
            hat="default",
            source_uri="notes.md",
            source_path=Path("notes.md"),
            heading="Training",
            section="Training",
            topic_folder="jepa-notes",
        ),
    )

    answer = synthesize_answer("What is JEPA?", [hit])

    assert "Training" not in answer
    assert "predicts latent targets from context" in answer


def test_generate_answer_uses_local_llm_when_available() -> None:
    traces = ModelTraceCollector()
    generator = FakeGenerator("JEPA stands for Joint Embedding Predictive Architecture.")

    answer = generate_answer(
        "What is JEPA?",
        [_hit("Joint embedding predictive architecture learns representations from context.")],
        context_limit=8,
        expanded_query="what is jepa joint embedding predictive architecture",
        selected_hat="default",
        generator=generator,
        trace_sink=traces,
    )

    calls = traces.snapshot()

    assert answer == "JEPA stands for Joint Embedding Predictive Architecture."
    assert len(generator.calls) == 1
    assert "<question_brief>" in generator.calls[0][1]
    assert "<retrieved_passages>" in generator.calls[0][1]
    assert "<example>" in generator.calls[0][1]
    assert calls[-1].task == "answer generation"
    assert calls[-1].status == "ok"


def test_generate_answer_falls_back_and_logs_when_local_llm_fails(app_home: Path) -> None:
    traces = ModelTraceCollector()
    progress: list[str] = []
    store = SessionStore(app_home)
    logger = SessionExceptionLogger(store, terminal_pid=5555)

    answer = generate_answer(
        "What is JEPA?",
        [_hit("Joint embedding predictive architecture predicts latent targets from context.")],
        context_limit=8,
        expanded_query="what is jepa joint embedding predictive architecture",
        selected_hat="default",
        generator=FailingGenerator("runtime offline"),
        trace_sink=traces,
        exception_logger=logger,
        progress_sink=progress.append,
    )

    log_path = store.active_exception_log_path(5555)
    calls = traces.snapshot()

    assert "predicts latent targets from context" in answer
    assert log_path.exists()
    assert "runtime offline" in log_path.read_text(encoding="utf-8")
    assert calls[-1].task == "answer generation"
    assert calls[-1].status == "fallback"
    assert any(f"Log: {log_path.resolve()}" in message for message in progress)


def test_compose_answer_supports_none_mode_without_llm_calls() -> None:
    traces = ModelTraceCollector()
    default_generator = FakeGenerator("unused")
    light_generator = FakeGenerator("unused")

    answer, citations = compose_answer(
        "What is JEPA?",
        [_hit("Joint embedding predictive architecture predicts latent targets from context.")],
        answer_mode="none",
        context_limit=8,
        expanded_query="what is jepa joint embedding predictive architecture",
        selected_hat="default",
        default_generator=default_generator,
        light_generator=light_generator,
        trace_sink=traces,
    )

    assert "predicts latent targets from context" in answer
    assert citations == ["default/jepa-notes/notes.md: Overview"]
    assert default_generator.calls == []
    assert light_generator.calls == []
    assert traces.snapshot() == []


def test_compose_answer_supports_raw_mode() -> None:
    answer, citations = compose_answer(
        "What is JEPA?",
        [_hit("Joint embedding predictive architecture predicts latent targets from context.")],
        answer_mode="raw",
        context_limit=8,
        expanded_query="what is jepa joint embedding predictive architecture",
        selected_hat="default",
        default_generator=FakeGenerator("unused"),
        light_generator=FakeGenerator("unused"),
    )

    assert answer.startswith("Top retrieved context:")
    assert "[0.930] default/jepa-notes/notes.md: Overview" in answer
    assert citations == []


def test_application_uses_per_mode_answer_context_limits(app_home: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "arignan.application.create_local_text_generator",
        lambda config, progress_sink=None, **kwargs: ArtifactGenerator(kwargs.get("model_name") or config.local_llm_model),
    )
    app = ArignanApp(load_config(app_home=app_home))

    assert app._answer_context_limit("default") == 8
    assert app._answer_context_limit("light") == 6
    assert app._answer_context_limit("none") == 8
    assert app._answer_context_limit("raw") == 8
    assert app._answer_context_limit("light", rerank_top_k=9) == 9
    assert app._effective_fused_top_k(11) == 22


def test_application_ask_respects_rerank_override(app_home: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}
    hit = _hit("Joint embedding predictive architecture predicts latent targets from context.")

    class StubRetrievalPipeline:
        def __init__(self, *args, **kwargs) -> None:
            captured["fused_limit"] = kwargs["fused_limit"]

        def retrieve(self, query: str, hat: str = "auto") -> RetrievalBundle:
            return RetrievalBundle(
                query=query,
                expanded_query=query.lower(),
                selected_hat="default",
                dense_hits=[hit],
                lexical_hits=[],
                map_hits=[],
                fused_hits=[hit],
            )

    class CapturingReranker:
        model_name = "mixedbread-ai/mxbai-rerank-base-v1"
        backend_name = "cross-encoder"

        def rerank(self, query: str, hits: list[RetrievalHit], limit: int, min_score: float = 0.0) -> list[RetrievalHit]:
            captured["rerank_limit"] = limit
            return hits[:1]

    monkeypatch.setattr(
        "arignan.application.create_local_text_generator",
        lambda config, progress_sink=None, **kwargs: ArtifactGenerator(kwargs.get("model_name") or config.local_llm_model),
    )
    monkeypatch.setattr("arignan.application.RetrievalPipeline", StubRetrievalPipeline)

    app = ArignanApp(load_config(app_home=app_home))
    app.reranker = CapturingReranker()

    result = app.ask("What is JEPA?", answer_mode="none", rerank_top_k=11)

    assert captured["fused_limit"] == 22
    assert captured["rerank_limit"] == 11
    assert result.citations == ["default/jepa-notes/notes.md: Overview"]


def test_application_skips_retrieval_for_conversational_followup(app_home: Path, monkeypatch) -> None:
    generator = FakeGenerator("You're right. Let me answer it directly instead of echoing the prompt.")

    def fail_retrieval_pipeline(*args, **kwargs):
        raise AssertionError("Retrieval should not run for conversational follow-up")

    monkeypatch.setattr(
        "arignan.application.create_local_text_generator",
        lambda config, progress_sink=None, **kwargs: generator,
    )
    monkeypatch.setattr("arignan.application.RetrievalPipeline", fail_retrieval_pipeline)

    app = ArignanApp(load_config(app_home=app_home))
    app.session_manager.append_turn(app.terminal_pid, role="user", content="What is JEPA?")
    app.session_manager.append_turn(
        app.terminal_pid,
        role="assistant",
        content="JEPA stands for Joint Embedding Predictive Architecture.",
    )

    result = app.ask("No but answer properly, don't just repeat my prompt", answer_mode="default")

    assert result.answer == "You're right. Let me answer it directly instead of echoing the prompt."
    assert result.citations == []
    assert result.debug.fused_hits == []
    assert "conversational follow-up" in generator.calls[0][0].lower()


def test_application_no_context_still_uses_llm_with_warning(app_home: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}
    generator = FakeGenerator(
        "No local context was found for this turn, so I’m answering from our earlier chat and general knowledge. "
        "JEPA is a predictive learning approach built around joint embeddings."
    )

    class EmptyRetrievalPipeline:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def retrieve(self, query: str, hat: str = "auto") -> RetrievalBundle:
            return RetrievalBundle(
                query=query,
                expanded_query=query.lower(),
                selected_hat="default",
                dense_hits=[],
                lexical_hits=[],
                map_hits=[],
                fused_hits=[],
            )

    class EmptyReranker:
        model_name = "mixedbread-ai/mxbai-rerank-base-v1"
        backend_name = "cross-encoder"

        def rerank(self, query: str, hits: list[RetrievalHit], limit: int, min_score: float = 0.0) -> list[RetrievalHit]:
            captured["rerank_called"] = True
            return []

    monkeypatch.setattr(
        "arignan.application.create_local_text_generator",
        lambda config, progress_sink=None, **kwargs: generator,
    )
    monkeypatch.setattr("arignan.application.RetrievalPipeline", EmptyRetrievalPipeline)

    app = ArignanApp(load_config(app_home=app_home))
    app.reranker = EmptyReranker()

    result = app.ask("What is JEPA?", answer_mode="default")

    assert result.answer.startswith("No local context was found for this turn")
    assert result.citations == []
    assert result.debug.fused_hits == []
    assert "no useful retrieved local context was found" in generator.calls[0][0].lower()


def test_application_ask_falls_back_to_fused_hits_when_reranker_returns_none(app_home: Path, monkeypatch) -> None:
    hit = RetrievalHit(
        chunk_id="chunk-spiking-mamba",
        text="A spiking Mamba implementation usually combines selective state-space updates with spike-timing dynamics.",
        score=0.71,
        source=RetrievalSource.DENSE,
        metadata=ChunkMetadata(
            load_id="load-spiking",
            hat="SNNs",
            source_uri="spiking_mamba_notes.md",
            source_path=Path("spiking_mamba_notes.md"),
            heading="Implementation Notes",
            section="Implementation Notes",
            topic_folder="spiking-mamba",
        ),
        extras={"rrf_score": 0.12},
    )

    class StubRetrievalPipeline:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def retrieve(self, query: str, hat: str = "auto") -> RetrievalBundle:
            return RetrievalBundle(
                query=query,
                expanded_query="how to implement a spiking mamba",
                selected_hat="SNNs",
                dense_hits=[hit],
                lexical_hits=[],
                map_hits=[],
                fused_hits=[hit],
            )

    monkeypatch.setattr(
        "arignan.application.create_local_text_generator",
        lambda config, progress_sink=None, **kwargs: ArtifactGenerator(kwargs.get("model_name") or config.local_llm_model),
    )
    monkeypatch.setattr("arignan.application.RetrievalPipeline", StubRetrievalPipeline)

    app = ArignanApp(load_config(app_home=app_home))
    app.reranker = EmptyCrossEncoderReranker()

    result = app.ask("How to implement a spiking mamba", answer_mode="none")

    assert result.debug.reranked_hits == []
    assert result.selected_hat == "SNNs"
    assert "No relevant local knowledge was found" not in result.answer
    assert "spiking Mamba implementation" in result.answer


def test_render_raw_hits_handles_empty_inputs() -> None:
    assert render_raw_hits([]) == "No relevant local knowledge was found for that question."


def test_parse_grouping_review_accepts_valid_merge_payload() -> None:
    recommendations = _parse_grouping_review(
        json.dumps(
            {
                "recommendations": [
                    {
                        "members": ["latent-prediction-training", "jepa"],
                        "target_topic_folder": "jepa",
                        "confidence": 0.72,
                        "rationale": "same ideas",
                    }
                ]
            }
        ),
        topics=[
            TopicGroupingRecord(
                topic_folder="latent-prediction-training",
                title="Latent Prediction Training",
                locator="predictive representation learning note",
                description="A note on latent prediction training.",
                keywords=["latent prediction"],
                summary_excerpt="latent prediction training note",
                source_count=1,
                estimated_length=220,
                current_load=True,
            ),
            TopicGroupingRecord(
                topic_folder="jepa",
                title="JEPA",
                locator="predictive representation learning ideas",
                description="Notes on JEPA.",
                keywords=["JEPA"],
                summary_excerpt="predictive representation learning ideas",
                source_count=1,
                estimated_length=220,
                current_load=False,
            ),
        ],
    )

    assert len(recommendations) == 1
    assert recommendations[0].target_topic_folder == "jepa"
    assert recommendations[0].confidence == 0.72


def test_application_grouping_review_uses_full_topic_list(app_home: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "arignan.application.create_local_text_generator",
        lambda config, progress_sink=None, **kwargs: GroupingReviewGenerator("jepa"),
    )
    app = ArignanApp(load_config(app_home=app_home))
    hat_layout = app.layout.hat("default").ensure()
    topic_dir = hat_layout.summaries_dir / "jepa"
    topic_dir.mkdir(parents=True, exist_ok=True)
    existing_document = ParsedDocument(
        load_id="load-existing",
        hat="default",
        source=SourceDocument(
            source_type=SourceType.MARKDOWN,
            source_uri="existing.md",
            local_path=Path("existing.md"),
            title="JEPA Overview",
        ),
        full_text="JEPA is a predictive representation learning approach based on latent targets.",
        sections=[DocumentSection(text="JEPA is a predictive representation learning approach.", heading="Overview")],
        keywords=["JEPA", "latent target prediction"],
    )
    manifest_payload = {
        "hat": "default",
        "topic_folder": "jepa",
        "source_paths": ["existing.md"],
        "markdown_paths": ["summary.md"],
        "keywords": ["JEPA", "latent target prediction", "representation learning"],
        "title": "JEPA",
        "locator": "predictive representation learning ideas",
        "description": "Notes on JEPA and related predictive learning concepts.",
        "documents": [existing_document.to_dict()],
    }
    (topic_dir / ".topic_manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    (topic_dir / "summary.md").write_text(
        "# JEPA\n\n## Summary\nJEPA covers predictive representation learning ideas and latent target prediction.\n",
        encoding="utf-8",
    )

    unrelated_dir = hat_layout.summaries_dir / "positional-encoding"
    unrelated_dir.mkdir(parents=True, exist_ok=True)
    unrelated_document = ParsedDocument(
        load_id="load-existing-2",
        hat="default",
        source=SourceDocument(
            source_type=SourceType.MARKDOWN,
            source_uri="encoding.md",
            local_path=Path("encoding.md"),
            title="Positional Encoding Ideas",
        ),
        full_text="Positional encoding injects order information into sequence models.",
        sections=[DocumentSection(text="Positional encoding injects order information.", heading="Overview")],
        keywords=["positional encoding", "sequence order"],
    )
    unrelated_payload = {
        "hat": "default",
        "topic_folder": "positional-encoding",
        "source_paths": ["encoding.md"],
        "markdown_paths": ["summary.md"],
        "keywords": ["positional encoding", "sequence order"],
        "title": "Positional Encoding Ideas",
        "locator": "sequence order and position signals",
        "description": "Notes on positional encodings and order injection.",
        "documents": [unrelated_document.to_dict()],
    }
    (unrelated_dir / ".topic_manifest.json").write_text(json.dumps(unrelated_payload, indent=2), encoding="utf-8")
    (unrelated_dir / "summary.md").write_text(
        "# Positional Encoding Ideas\n\n## Summary\nPositional encoding adds order information to sequence representations.\n",
        encoding="utf-8",
    )

    new_document = ParsedDocument(
        load_id="load-new",
        hat="default",
        source=SourceDocument(
            source_type=SourceType.MARKDOWN,
            source_uri="new.md",
            local_path=Path("new.md"),
            title="Latent Prediction Training",
        ),
        full_text="Latent prediction training in joint embedding systems improves representation learning.",
        sections=[DocumentSection(text="Latent prediction training in joint embedding systems.", heading="Training")],
    )
    provisional_plan = GroupingPlan(
        decision=GroupingDecision.STANDALONE,
        topic_folder="latent-prediction-training",
        estimated_length=300,
    )
    app.markdown_repository.write_topic(app.layout, hat="default", documents=[new_document], plan=provisional_plan, refresh_maps=False)

    topics = app._collect_grouping_topics("default", load_id="load-new")
    prompt = _build_grouping_review_prompt("default", topics)
    recommendations = app._grouping_recommendations("default", "load-new", topics)

    assert len(topics) == 3
    assert {topic.topic_folder for topic in topics} == {"jepa", "positional-encoding", "latent-prediction-training"}
    assert "Title: Positional Encoding Ideas" in prompt
    assert "<pair_hints>" in prompt
    assert "Current load: yes" in prompt
    assert "Current load: no" in prompt
    assert len(recommendations) == 1
    assert recommendations[0].target_topic_folder == "jepa"


def test_application_load_preserves_document_derived_topic_without_topic_naming(app_home: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "arignan.application.create_local_text_generator",
        lambda config, progress_sink=None, **kwargs: ArtifactGenerator(kwargs.get("model_name") or config.local_llm_model),
    )
    app = ArignanApp(load_config(app_home=app_home))
    source = tmp_path / "alpha.md"
    source.write_text(
        "# AlphaActivationTTFS\n\nAlphaActivationTTFS discusses spiking activation timing and threshold-first spike behavior.\n",
        encoding="utf-8",
    )

    result = app.load(str(source), hat="default")

    assert result.topic_folders == ["alphaactivationttfs"]
    assert all(call.task != "topic naming" for call in result.model_calls)


def test_application_load_post_regroups_after_all_topic_summaries_exist(
    app_home: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    light_generator = PostLoadRegroupGenerator()

    def fake_generator_factory(config, progress_sink=None, **kwargs):
        requested_model = kwargs.get("model_name")
        if requested_model in {None, config.local_llm_model}:
            return light_generator
        return ArtifactGenerator(requested_model or config.local_llm_model)

    monkeypatch.setattr("arignan.application.create_local_text_generator", fake_generator_factory)
    app = ArignanApp(load_config(app_home=app_home))
    app.grouping_planner.min_merge_score = 10.0

    folder = tmp_path / "batch"
    folder.mkdir()
    first = folder / "a-first.md"
    second = folder / "b-second.md"
    first.write_text("# First Topic\n\nAlpha spikes are discussed here.\n", encoding="utf-8")
    second.write_text("# Second Topic\n\nAlpha spikes and sequence coding are grouped here.\n", encoding="utf-8")

    result = app.load(str(folder), hat="default")

    assert result.topic_folders == ["second-topic"]
    assert result.total_markdown_segments == 1
    assert any(trace.title == "First Topic" and trace.topic_folder == "second-topic" for trace in result.traces)
    assert any(call.task == "batch grouping review" for call in result.model_calls)
    second_manifest = app.layout.hat("default").summaries_dir / "second-topic" / ".topic_manifest.json"
    payload = json.loads(second_manifest.read_text(encoding="utf-8"))
    assert len(payload["documents"]) == 2
    assert not (app.layout.hat("default").summaries_dir / "first-topic").exists()


def test_application_delete_does_not_recreate_missing_hat_for_stale_log(app_home: Path) -> None:
    app = ArignanApp(load_config(app_home=app_home))
    stale_event = LoadEvent(
        load_id="load-stale",
        operation=LoadOperation.INGEST,
        hat="SNNs",
        created_at="2026-04-11T12:00:00+00:00",
        source_items=["paper.pdf"],
        topic_folders=["word2vec"],
    )
    app.ingestion_log.append(stale_event)

    result = app.delete(["load-stale"])

    assert result.deleted_load_ids == ["load-stale"]
    assert "word2vec" in result.deleted_topics
    assert not app.layout.hat("SNNs").root.exists()
    assert app.list_live_ingestions() == []
