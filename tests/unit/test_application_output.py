from __future__ import annotations

import json
from pathlib import Path

from arignan.application import ArignanApp, _build_grouping_prompt, _parse_grouping_hint, compose_answer, format_citation, generate_answer, render_raw_hits, synthesize_answer
from arignan.config import load_config
from arignan.grouping import GroupingDecision, GroupingPlan
from arignan.models import ChunkMetadata, DocumentSection, ParsedDocument, RetrievalHit, RetrievalSource, SourceDocument, SourceType
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


class GroupingGenerator:
    model_name = "qwen3:0.6b"
    backend_name = "fake-backend"

    def __init__(self, topic_folder: str) -> None:
        self.topic_folder = topic_folder

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
                "decision": "merge",
                "topic_folder": self.topic_folder,
                "confidence": 0.81,
                "rationale": "same conceptual family",
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
    model_name = "qwen3:0.6b"
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
        if "Title: First Topic" in user_prompt and "Title: Second Topic" in user_prompt:
            return json.dumps(
                {
                    "decision": "merge",
                    "topic_folder": "second-topic",
                    "confidence": 0.91,
                    "rationale": "the first topic fits better as part of the second topic page",
                }
            )
        return json.dumps(
            {
                "decision": "standalone",
                "topic_folder": "",
                "confidence": 0.25,
                "rationale": "keep separate",
            }
        )


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


def test_render_raw_hits_handles_empty_inputs() -> None:
    assert render_raw_hits([]) == "No relevant local knowledge was found for that question."


def test_parse_grouping_hint_accepts_valid_merge_payload() -> None:
    hint = _parse_grouping_hint(
        json.dumps(
            {
                "decision": "merge",
                "topic_folder": "jepa",
                "confidence": 0.72,
                "rationale": "same ideas",
            }
        ),
        candidates=[type("Candidate", (), {"topic_folder": "jepa"})()],
    )

    assert hint is not None
    assert hint.topic_folder == "jepa"
    assert hint.confidence == 0.72


def test_application_topic_merge_candidates_and_light_llm_hint(app_home: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "arignan.application.create_local_text_generator",
        lambda config, progress_sink=None, **kwargs: GroupingGenerator("jepa"),
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
    (topic_dir / "markdown_tree").mkdir(parents=True, exist_ok=True)
    (topic_dir / "markdown_tree" / "summary.md").write_text(
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
    (unrelated_dir / "markdown_tree").mkdir(parents=True, exist_ok=True)
    (unrelated_dir / "markdown_tree" / "summary.md").write_text(
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

    candidates = app._topic_merge_candidates("default", new_document, related_hits=[])
    provisional_plan = GroupingPlan(
        decision=GroupingDecision.STANDALONE,
        topic_folder="latent-prediction-training",
        estimated_length=300,
    )
    provisional_topic = app.heuristic_artifact_writer.render_topic([new_document], provisional_plan)
    prompt = _build_grouping_prompt(new_document, candidates, provisional_topic.summary_markdown)
    hint = app._grouping_hint(
        new_document,
        candidates,
        provisional_topic.summary_markdown,
        index=1,
        total=1,
    )

    assert len(candidates) == 2
    assert {candidate.topic_folder for candidate in candidates} == {"jepa", "positional-encoding"}
    assert candidates[0].topic_folder == "jepa"
    assert "predictive representation learning ideas" in candidates[0].summary_excerpt
    assert hint is not None
    assert hint.topic_folder == "jepa"
    assert "<incoming_topic_summary>" in prompt
    assert "<existing_topic_summaries>" in prompt
    assert "Positional Encoding Ideas" in prompt


def test_application_load_preserves_document_derived_topic_without_topic_naming(app_home: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "arignan.application.create_local_text_generator",
        lambda config, progress_sink=None, **kwargs: GroupingGenerator("unused"),
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
        if requested_model == config.local_llm_light_model:
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
    assert any(call.task == "grouping decision" for call in result.model_calls)
    second_manifest = app.layout.hat("default").summaries_dir / "second-topic" / ".topic_manifest.json"
    payload = json.loads(second_manifest.read_text(encoding="utf-8"))
    assert len(payload["documents"]) == 2
    assert not (app.layout.hat("default").summaries_dir / "first-topic").exists()
