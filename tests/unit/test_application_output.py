from __future__ import annotations

from pathlib import Path

from arignan.application import compose_answer, format_citation, generate_answer, render_raw_hits, synthesize_answer
from arignan.models import ChunkMetadata, RetrievalHit, RetrievalSource
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
    ) -> str:
        raise RuntimeError(self.message)


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
        expanded_query="what is jepa joint embedding predictive architecture",
        selected_hat="default",
        generator=generator,
        trace_sink=traces,
    )

    calls = traces.snapshot()

    assert answer == "JEPA stands for Joint Embedding Predictive Architecture."
    assert len(generator.calls) == 1
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
