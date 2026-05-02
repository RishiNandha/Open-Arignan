from __future__ import annotations

import json
import sys
from typing import Any, BinaryIO

from arignan.mcp.server import ArignanMCPServer


PROTOCOL_VERSION = "2024-11-05"


def run_stdio_server(server: ArignanMCPServer) -> int:
    input_stream = sys.stdin.buffer
    output_stream = sys.stdout.buffer
    while True:
        message = _read_message(input_stream)
        if message is None:
            return 0
        if "method" not in message:
            continue
        method = message["method"]
        if method == "notifications/initialized":
            continue
        if method == "ping":
            _write_message(output_stream, _response(message.get("id"), {}))
            continue
        try:
            result = _dispatch(server, method, message.get("params") or {})
        except Exception as exc:
            _write_message(
                output_stream,
                {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "error": {"code": -32000, "message": str(exc)},
                },
            )
            continue
        if "id" in message:
            _write_message(output_stream, _response(message["id"], result))


def _dispatch(server: ArignanMCPServer, method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "initialize":
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"listChanged": False},
            },
            "serverInfo": {
                "name": "arignan",
                "version": "1.0.0",
            },
        }
    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
                for tool in server.list_tools()
            ]
        }
    if method == "tools/call":
        name = params["name"]
        arguments = params.get("arguments") or {}
        structured = server.call_tool(name, arguments)
        return {
            "structuredContent": structured,
            "content": [{"type": "text", "text": json.dumps(structured, indent=2)}],
        }
    if method == "resources/list":
        return {
            "resources": [
                {
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description,
                    "mimeType": "text/markdown",
                }
                for resource in server.list_resources()
            ]
        }
    if method == "resources/read":
        uri = params["uri"]
        text = server.read_resource(uri)
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "text/markdown",
                    "text": text,
                }
            ]
        }
    raise ValueError(f"unsupported MCP method: {method}")


def _response(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _read_message(stream: BinaryIO) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        raw_line = stream.readline()
        if raw_line == b"":
            return None
        line = raw_line.decode("utf-8").strip()
        if not line:
            break
        if ":" not in line:
            continue
        name, value = line.split(":", maxsplit=1)
        headers[name.strip().lower()] = value.strip()
    content_length = int(headers.get("content-length", "0"))
    if content_length <= 0:
        return None
    payload = stream.read(content_length)
    if not payload:
        return None
    return json.loads(payload.decode("utf-8"))


def _write_message(stream: BinaryIO, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    stream.write(header)
    stream.write(body)
    stream.flush()
