from __future__ import annotations

import json
from pathlib import Path

from arignan.mcp_config import DEFAULT_MCP_CONFIG, load_mcp_config, write_default_mcp_config


def test_write_default_mcp_config_creates_mcp_json(tmp_path: Path) -> None:
    path = write_default_mcp_config(tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / "mcp.json"
    assert payload["server_name"] == DEFAULT_MCP_CONFIG.server_name
    assert "retrieve_context" in payload["tools"]
    assert payload["tools"]["ask"]["enabled"] is False
    assert "global_map" in payload["resources"]


def test_load_mcp_config_merges_user_overrides(tmp_path: Path) -> None:
    path = write_default_mcp_config(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["instructions"] = "Custom MCP instructions."
    payload["tools"]["retrieve_context"]["description"] = "Custom retrieve description."
    payload["resources"]["global_map"]["description"] = "Custom global map description."
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    config = load_mcp_config(tmp_path)

    assert config.instructions == "Custom MCP instructions."
    assert config.tools["retrieve_context"].description == "Custom retrieve description."
    assert config.resources["global_map"].description == "Custom global map description."
    assert config.tools["ask"].description == DEFAULT_MCP_CONFIG.tools["ask"].description


def test_load_mcp_config_merges_enabled_flags(tmp_path: Path) -> None:
    path = write_default_mcp_config(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tools"]["ask"]["enabled"] = True
    payload["tools"]["retrieve_context"]["enabled"] = False
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    config = load_mcp_config(tmp_path)

    assert config.tools["ask"].enabled is True
    assert config.tools["retrieve_context"].enabled is False


def test_load_mcp_config_recreates_missing_mcp_json(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    if path.exists():
        path.unlink()

    config = load_mcp_config(tmp_path)

    assert config.server_name == DEFAULT_MCP_CONFIG.server_name
    assert path.exists()
