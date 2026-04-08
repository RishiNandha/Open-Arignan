from __future__ import annotations

import json
import shutil
from pathlib import Path

from arignan.application import ArignanApp
from arignan.config import load_config


def test_end_to_end_load_group_ask_save_and_delete(tmp_path: Path) -> None:
    app_home = tmp_path / ".arignan"
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
