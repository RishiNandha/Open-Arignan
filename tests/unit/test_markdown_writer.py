from __future__ import annotations

from pathlib import Path

from arignan.config import AppConfig
from arignan.grouping import GroupingDecision, GroupingPlan
from arignan.llm.runtime import resolve_local_model_source, sanitize_model_id
from arignan.markdown.writer import HeuristicArtifactWriter, LLMArtifactWriter, TopicMapEntry, HatMapEntry
from arignan.models import DocumentSection, ParsedDocument, SourceDocument, SourceType
from arignan.session import SessionExceptionLogger, SessionStore
from arignan.tracing import ModelTraceCollector


class FakeGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[str, str]] = []
        self.response_formats: list[object] = []
        self.model_name = "fake-llm"
        self.backend_name = "fake-backend"

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
        self.response_formats.append(response_format)
        return self.outputs.pop(0)


class FailingGenerator:
    def __init__(self, message: str = "llm failed") -> None:
        self.model_name = "fake-llm"
        self.backend_name = "fake-backend"
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


def _document(tmp_path: Path) -> ParsedDocument:
    source = tmp_path / "notes.md"
    source.write_text("# V-JEPA 2\n\nJoint embedding predictive architecture notes.\n", encoding="utf-8")
    return ParsedDocument(
        load_id="load-1",
        hat="default",
        source=SourceDocument(
            source_type=SourceType.MARKDOWN,
            source_uri=str(source),
            local_path=source,
            title="V-JEPA 2",
        ),
        full_text=(
            "We propose a world model for predictive video representations. "
            "The method focuses on latent prediction and temporal abstraction."
        ),
        sections=[
            DocumentSection(
                text="We propose a world model for predictive video representations. The method focuses on latent prediction and temporal abstraction.",
                heading="Overview",
            )
        ],
    )


def test_resolve_local_model_source_prefers_downloaded_model_dir(tmp_path: Path) -> None:
    app_home = tmp_path / ".arignan"
    model_dir = app_home / "models" / sanitize_model_id("Qwen/Qwen3-1.7B")
    model_dir.mkdir(parents=True)
    config = AppConfig(app_home=app_home, local_llm_backend="transformers", local_llm_model="Qwen/Qwen3-1.7B")

    source = resolve_local_model_source(config)

    assert source == str(model_dir)


def test_llm_artifact_writer_uses_json_response_for_topic_render(tmp_path: Path) -> None:
    document = _document(tmp_path)
    generator = FakeGenerator(
        [
            (
                '{"title":"V-JEPA 2","description":"Video representation notes.","locator":"paper on world models and latent prediction",'
                '"keywords":["V-JEPA","world model","latent prediction"],'
                '"summary_markdown":"# V-JEPA 2\\n\\nA concise lead.\\n\\n## Summary\\nShort summary.\\n\\n## Key Ideas\\n- World models\\n- Latent prediction\\n\\n## Sources\\n| Source | What To Find | Key Sections | File |\\n| --- | --- | --- | --- |\\n| V-JEPA 2 | Video representation notes | Overview | `notes.md` |\\n\\n## Keywords\\nV-JEPA, world model, latent prediction"}'
            )
        ]
    )
    writer = LLMArtifactWriter(
        generator=generator,
        fallback=HeuristicArtifactWriter(),
    )

    rendered = writer.render_topic(
        [document],
        GroupingPlan(decision=GroupingDecision.STANDALONE, topic_folder="v-jepa-2", estimated_length=400),
    )

    assert rendered.title == "V-JEPA 2"
    assert rendered.description == "Video representation notes."
    assert rendered.locator == "paper on world models and latent prediction"
    assert rendered.keywords == ["V-JEPA", "world model", "latent prediction"]
    assert rendered.summary_markdown.startswith("# V-JEPA 2")
    assert "## Related Threads" in rendered.summary_markdown
    assert "## Sources" in rendered.summary_markdown
    assert generator.response_formats[0] is not None
    assert generator.response_formats[0]["type"] == "object"
    assert "Example of a good summary_markdown shape:" in generator.calls[0][1]
    assert "rich lookup article" in generator.calls[0][1]
    assert "one coherent topic page" in generator.calls[0][1]
    assert "topic index file" in generator.calls[0][1]


def test_llm_artifact_writer_falls_back_on_invalid_json(tmp_path: Path) -> None:
    document = _document(tmp_path)
    writer = LLMArtifactWriter(
        generator=FakeGenerator(["not json"]),
        fallback=HeuristicArtifactWriter(),
    )

    rendered = writer.render_topic(
        [document],
        GroupingPlan(decision=GroupingDecision.STANDALONE, topic_folder="v-jepa-2", estimated_length=400),
    )

    assert rendered.title == "V-JEPA 2"
    assert rendered.summary_markdown.startswith("# V-JEPA 2")
    assert "## Related Threads" in rendered.summary_markdown


def test_llm_artifact_writer_renders_map_and_global_map(tmp_path: Path) -> None:
    generator = FakeGenerator(
        [
            "# Map for Hat: default\n\n| Topic | Directory | What To Find | Source Files | Keywords |\n| --- | --- | --- | --- | --- |\n| V-JEPA 2 | `summaries/v-jepa-2` | paper on world models | notes.md | V-JEPA, world model |\n",
            "# Global Map\n\n| Hat | Map Path | What To Find | High-Level Keywords |\n| --- | --- | --- | --- |\n| default | `hats/default/map.md` | paper on world models | V-JEPA, world model |\n",
        ]
    )
    writer = LLMArtifactWriter(generator=generator, fallback=HeuristicArtifactWriter())

    map_text = writer.render_hat_map(
        "default",
        [
            TopicMapEntry(
                topic_folder="v-jepa-2",
                title="V-JEPA 2",
                locator="paper on world models",
                source_files=["notes.md"],
                markdown_files=["summary.md"],
                keywords=["V-JEPA", "world model"],
            )
        ],
    )
    global_map_text = writer.render_global_map(
        [HatMapEntry(hat="default", map_path="hats/default/map.md", what_to_find="paper on world models", keywords=["V-JEPA", "world model"])]
    )

    assert map_text.startswith("# Map for Hat: default")
    assert global_map_text.startswith("# Global Map")


def test_llm_artifact_writer_records_model_calls(tmp_path: Path) -> None:
    document = _document(tmp_path)
    traces = ModelTraceCollector()
    writer = LLMArtifactWriter(
        generator=FakeGenerator(
            [
                (
                    '{"title":"V-JEPA 2","description":"Video representation notes.","locator":"paper on world models",'
                    '"keywords":["V-JEPA","world model"],'
                    '"summary_markdown":"# V-JEPA 2\\n\\nLead.\\n\\n## Summary\\nShort summary.\\n\\n## Key Ideas\\n- World model\\n\\n## Sources\\n| Source | What To Find | Key Sections | File |\\n| --- | --- | --- | --- |\\n| V-JEPA 2 | Video representation notes | Overview | `notes.md` |\\n\\n## Keywords\\nV-JEPA, world model"}'
                ),
                "# Map for Hat: default\n\n| Topic | Directory | What To Find | Source Files | Keywords |\n| --- | --- | --- | --- | --- |\n| V-JEPA 2 | `summaries/v-jepa-2` | paper on world models | notes.md | V-JEPA, world model |\n",
                "# Global Map\n\n| Hat | Map Path | What To Find | High-Level Keywords |\n| --- | --- | --- | --- |\n| default | `hats/default/map.md` | paper on world models | V-JEPA, world model |\n",
            ]
        ),
        fallback=HeuristicArtifactWriter(),
        trace_sink=traces,
    )

    writer.render_topic([document], GroupingPlan(decision=GroupingDecision.STANDALONE, topic_folder="v-jepa-2", estimated_length=400))
    writer.render_hat_map(
        "default",
        [
            TopicMapEntry(
                topic_folder="v-jepa-2",
                title="V-JEPA 2",
                locator="paper on world models",
                source_files=["notes.md"],
                markdown_files=["summary.md"],
                keywords=["V-JEPA", "world model"],
            )
        ],
    )
    writer.render_global_map(
        [HatMapEntry(hat="default", map_path="hats/default/map.md", what_to_find="paper on world models", keywords=["V-JEPA", "world model"])]
    )

    calls = traces.snapshot()

    assert [call.task for call in calls] == ["topic summary markdown", "hat map markdown", "global map markdown"]
    assert all(call.model_name == "fake-llm" for call in calls)


def test_llm_artifact_writer_skips_empty_map_generation_and_emits_progress() -> None:
    progress: list[str] = []
    generator = FakeGenerator([])
    writer = LLMArtifactWriter(
        generator=generator,
        fallback=HeuristicArtifactWriter(),
        trace_sink=ModelTraceCollector(),
        progress_sink=progress.append,
    )

    hat_map = writer.render_hat_map("default", [])
    global_map = writer.render_global_map([])

    assert generator.calls == []
    assert hat_map.startswith("# Map for Hat: default")
    assert global_map.startswith("# Global Map")
    assert progress == []


def test_llm_artifact_writer_logs_full_traceback_to_session_log(app_home: Path, tmp_path: Path) -> None:
    progress: list[str] = []
    document = _document(tmp_path)
    store = SessionStore(app_home)
    logger = SessionExceptionLogger(store, terminal_pid=4321)
    writer = LLMArtifactWriter(
        generator=FailingGenerator("local model exploded"),
        fallback=HeuristicArtifactWriter(),
        exception_logger=logger,
        progress_sink=progress.append,
    )

    rendered = writer.render_topic(
        [document],
        GroupingPlan(decision=GroupingDecision.STANDALONE, topic_folder="v-jepa-2", estimated_length=400),
    )
    log_path = store.active_exception_log_path(4321)
    log_text = log_path.read_text(encoding="utf-8")

    assert rendered.summary_markdown.startswith("# V-JEPA 2")
    assert "## Related Threads" in rendered.summary_markdown
    assert log_path.exists()
    assert '"component": "llm"' in log_text
    assert '"task": "topic summary markdown"' in log_text
    assert '"exception_type": "RuntimeError"' in log_text
    assert "local model exploded" in log_text
    assert "Traceback (most recent call last)" in log_text
    assert any(f"Log: {log_path.resolve()}" in message for message in progress)
