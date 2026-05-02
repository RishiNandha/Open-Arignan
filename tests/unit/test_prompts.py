from __future__ import annotations

import json
from pathlib import Path

import pytest

from arignan.prompts import DEFAULT_PROMPT_SET, load_prompt_set, render_prompt_template, write_default_prompts


def test_write_default_prompts_creates_prompts_json(tmp_path: Path) -> None:
    path = write_default_prompts(tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / "prompts.json"
    assert payload["answer_system_prompt"] == DEFAULT_PROMPT_SET.answer_system_prompt
    assert "topic_user_template" in payload
    assert "route_classification_user_template" in payload


def test_load_prompt_set_merges_user_overrides(tmp_path: Path) -> None:
    path = write_default_prompts(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["answer_system_prompt"] = "Custom answer system prompt."
    payload["hat_map_user_template"] = "Hat map for {hat}\n{topic_entries_block}"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    prompts = load_prompt_set(tmp_path)

    assert prompts.answer_system_prompt == "Custom answer system prompt."
    assert prompts.hat_map_user_template == "Hat map for {hat}\n{topic_entries_block}"
    assert prompts.topic_system_prompt == DEFAULT_PROMPT_SET.topic_system_prompt


def test_load_prompt_set_recreates_missing_prompts_json(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    if path.exists():
        path.unlink()

    prompts = load_prompt_set(tmp_path)

    assert prompts.answer_system_prompt == DEFAULT_PROMPT_SET.answer_system_prompt
    assert path.exists()


def test_conversational_prompt_defaults_do_not_inline_session_history() -> None:
    assert "{session_summary_block}" not in DEFAULT_PROMPT_SET.route_classification_user_template
    assert "{session_summary_block}" not in DEFAULT_PROMPT_SET.conversational_answer_user_template
    assert "{session_summary_block}" not in DEFAULT_PROMPT_SET.no_context_answer_user_template


def test_render_prompt_template_raises_for_unknown_placeholder() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        render_prompt_template("answer_user_template", "Question: {question}\nExtra: {missing}", question="What is JEPA?")

    assert "unknown placeholder 'missing'" in str(exc_info.value)
