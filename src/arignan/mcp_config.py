from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class McpResourceDescription:
    name: str
    description: str


@dataclass(slots=True)
class McpPromptDescription:
    name: str
    description: str


@dataclass(slots=True)
class McpToolDescription:
    description: str
    enabled: bool = True


@dataclass(slots=True)
class McpConfig:
    server_name: str
    instructions: str
    tools: dict[str, McpToolDescription]
    resources: dict[str, McpResourceDescription]
    prompts: dict[str, McpPromptDescription]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_MCP_CONFIG = McpConfig(
    server_name="arignan",
    instructions=(
        "Arignan is a local-first knowledge-base MCP server. Prefer retrieve_context when the client "
        "wants grounded local context without invoking a local answer LLM. Other tools map to the "
        "local knowledge-base maintenance flows."
    ),
    tools={
        "retrieve_context": McpToolDescription(
            description="Retrieve reranked local Arignan knowledge context for a query without calling an answer LLM.",
            enabled=True,
        ),
        "ask": McpToolDescription(
            description=(
                "Answer a question against the local knowledge base. Depending on settings, this either uses the "
                "local LLM lazily or prepares a client-LLM answer package from retrieved context."
            ),
            enabled=False,
        ),
        "load_content": McpToolDescription(
            description="Ingest a local file, folder, or URL into the knowledge base.",
            enabled=True,
        ),
        "list_loads": McpToolDescription(
            description="List load events from the ingestion history.",
            enabled=True,
        ),
        "delete_loads": McpToolDescription(
            description="Delete one or more prior ingestions by load_id.",
            enabled=True,
        ),
        "delete_hat": McpToolDescription(
            description="Delete an entire hat and all of its stored local knowledge.",
            enabled=True,
        ),
    },
    resources={
        "global_map": McpResourceDescription(
            name="global_map.md",
            description="High-level map across all hats in the local knowledge base.",
        )
    },
    prompts={
        "client_answer_with_context": McpPromptDescription(
            name="client_answer_with_context",
            description="Prompt template metadata for client-side answer synthesis from Arignan retrieval context.",
        )
    },
)


def mcp_config_path(app_home: Path) -> Path:
    return Path(app_home).expanduser().resolve() / "mcp.json"


def write_default_mcp_config(app_home: Path, *, overwrite: bool = False) -> Path:
    path = mcp_config_path(app_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return path
    path.write_text(json.dumps(DEFAULT_MCP_CONFIG.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_mcp_config(app_home: Path) -> McpConfig:
    path = mcp_config_path(app_home)
    if not path.exists():
        write_default_mcp_config(app_home, overwrite=False)
        return DEFAULT_MCP_CONFIG

    payload = json.loads(path.read_text(encoding="utf-8"))
    merged = DEFAULT_MCP_CONFIG.to_dict()
    if isinstance(payload.get("server_name"), str) and payload["server_name"].strip():
        merged["server_name"] = payload["server_name"]
    if isinstance(payload.get("instructions"), str) and payload["instructions"].strip():
        merged["instructions"] = payload["instructions"]

    for group_name, factory in {
        "tools": McpToolDescription,
        "resources": McpResourceDescription,
        "prompts": McpPromptDescription,
    }.items():
        raw_group = payload.get(group_name)
        if not isinstance(raw_group, dict):
            continue
        merged_group = merged[group_name]
        for key, current in list(merged_group.items()):
            update = raw_group.get(key)
            if not isinstance(update, dict):
                continue
            next_payload = dict(current)
            for field_name, value in update.items():
                if field_name == "enabled" and isinstance(value, bool):
                    next_payload[field_name] = value
                elif field_name in next_payload and isinstance(value, str) and value.strip():
                    next_payload[field_name] = value
            merged_group[key] = next_payload
        for key, update in raw_group.items():
            if key in merged_group or not isinstance(update, dict):
                continue
            candidate = {
                field_name: value
                for field_name, value in update.items()
                if (field_name == "enabled" and isinstance(value, bool))
                or (isinstance(value, str) and value.strip())
            }
            try:
                merged_group[key] = asdict(factory(**candidate))
            except TypeError:
                continue

    return McpConfig(
        server_name=str(merged["server_name"]),
        instructions=str(merged["instructions"]),
        tools={key: McpToolDescription(**value) for key, value in merged["tools"].items()},
        resources={key: McpResourceDescription(**value) for key, value in merged["resources"].items()},
        prompts={key: McpPromptDescription(**value) for key, value in merged["prompts"].items()},
    )
