from __future__ import annotations

import json
import shutil
from pathlib import Path

from arignan.application import ArignanApp
from arignan.config import load_config


class FakeLocalGenerator:
    backend_name = "fake-local"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 800,
        temperature: float = 0.1,
    ) -> str:
        if "Return strict JSON only" in system_prompt:
            return json.dumps(
                {
                    "title": "JEPA",
                    "description": "Merged notes on JEPA.",
                    "locator": "what JEPA focuses on",
                    "keywords": ["JEPA", "latent prediction", "representation learning"],
                    "summary_markdown": (
                        "# JEPA\n\n"
                        "A compact reference page.\n\n"
                        "## Summary\n"
                        "JEPA focuses on predictive representation learning.\n\n"
                        "## Key Ideas\n"
                        "- Latent prediction\n"
                        "- Context-based targets\n"
                        "- Representation learning\n\n"
                        "## Sources\n"
                        "| Source | What To Find | Key Sections | File |\n"
                        "| --- | --- | --- | --- |\n"
                        "| JEPA | What JEPA focuses on | Overview | `paper_a.md` |\n\n"
                        "## Keywords\n"
                        "JEPA, latent prediction, representation learning"
                    ),
                }
            )
        if "knowledge-base hat map" in system_prompt:
            return (
                "# Map for Hat: default\n\n"
                "| Topic | Directory | What To Find | Source Files | Keywords |\n"
                "| --- | --- | --- | --- | --- |\n"
                "| JEPA | `summaries/jepa` | what JEPA focuses on | paper_a.md, paper_b.md | JEPA, latent prediction |\n"
            )
        if "global knowledge-base map" in system_prompt:
            return (
                "# Global Map\n\n"
                "| Hat | Map Path | What To Find | High-Level Keywords |\n"
                "| --- | --- | --- | --- |\n"
                "| default | `hats/default/map.md` | what JEPA focuses on | JEPA, latent prediction |\n"
            )
        return "JEPA focuses on latent prediction and predictive representation learning."


def test_end_to_end_load_group_ask_save_and_delete(tmp_path: Path, monkeypatch) -> None:
    app_home = tmp_path / ".arignan"
    monkeypatch.setattr(
        "arignan.application.create_local_text_generator",
        lambda config, progress_sink=None, **kwargs: FakeLocalGenerator(kwargs.get("model_name") or config.local_llm_model),
    )
    app = ArignanApp(load_config(app_home=app_home))

    source_a = tmp_path / "paper_a.md"
    source_b = tmp_path / "paper_b.md"
    shutil.copy2(Path("tests/fixtures/grouped_topics/jepa/paper_a.md"), source_a)
    shutil.copy2(Path("tests/fixtures/grouped_topics/jepa/paper_b.md"), source_b)

    load_a = app.load(str(source_a), hat="default")
    load_b = app.load(str(source_b), hat="default")
    ask = app.ask("What does JEPA focus on?", hat="default", terminal_pid=2222)
    saved = app.save_session(terminal_pid=2222, destination=tmp_path / "session.json")

    topic_folder = load_b.topic_folders[0]
    manifest_path = app.layout.hat("default").summaries_dir / topic_folder / ".topic_manifest.json"
    manifest_before = json.loads(manifest_path.read_text(encoding="utf-8"))

    delete_result = app.delete([load_a.load_id])
    manifest_after = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert load_a.load_id != load_b.load_id
    assert len(manifest_before["documents"]) >= 1
    assert ask.citations
    assert saved.exists()
    assert load_a.load_id in delete_result.deleted_load_ids
    assert all(document["load_id"] != load_a.load_id for document in manifest_after["documents"])
