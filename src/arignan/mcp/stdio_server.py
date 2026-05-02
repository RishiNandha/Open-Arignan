from __future__ import annotations

import json
import sys
from typing import Any, BinaryIO

from arignan.mcp.server import ArignanMCPServer


PROTOCOL_VERSION = "2024-11-05"


def run_stdio_server(server: ArignanMCPServer) -> int:
    input_stream = sys.stdin.buffer
    output_stream = sys.stdout.buffer
    _log("Server started")
    while True:
        message = _read_message(input_stream)
        if message is None:
            _log("Input stream closed; shutting down")
            return 0
        if "method" not in message:
            continue
        method = message["method"]
        if method == "notifications/initialized":
            _log("Client initialization notification received")
            continue
        if method == "ping":
            _log("Ping received")
            _write_message(output_stream, _response(message.get("id"), {}))
            continue
        try:
            _log(f"Handling method: {method}")
            result = _dispatch(server, method, message.get("params") or {})
        except Exception as exc:
            _log(f"Method failed: {method}: {exc}")
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
            _log(f"Response sent for method: {method}")


def _dispatch(server: ArignanMCPServer, method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "initialize":
        _log("Initialize request received")
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
        _log("Listing tools")
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
        _log(f"Calling tool: {name}")
        arguments = params.get("arguments") or {}
        structured = server.call_tool(name, arguments)
        return {
            "structuredContent": structured,
            "content": [{"type": "text", "text": json.dumps(structured, indent=2)}],
        }
    if method == "resources/list":
        _log("Listing resources")
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
        _log(f"Reading resource: {uri}")
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
            _log("EOF while waiting for MCP headers")
            return None
        line = raw_line.decode("utf-8").strip()
        if not line:
            break
        if ":" not in line:
            _log(f"Ignoring malformed header line: {line!r}")
            continue
        name, value = line.split(":", maxsplit=1)
        headers[name.strip().lower()] = value.strip()
    if headers:
        _log(f"Received headers: {headers}")
    content_length = int(headers.get("content-length", "0"))
    if content_length <= 0:
        _log(f"No valid content-length found in headers: {headers}")
        return None
    payload = stream.read(content_length)
    if not payload:
        _log("Content-length was set but payload body was empty")
        return None
    payload_text = payload.decode("utf-8")
    preview = payload_text if len(payload_text) <= 1000 else payload_text[:997] + "..."
    _log(f"Received payload: {preview}")
    return json.loads(payload_text)


def _write_message(stream: BinaryIO, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    stream.write(header)
    stream.write(body)
    stream.flush()


def _log(message: str) -> None:
    print(f"[arignan-mcp] {message}", file=sys.stderr, flush=True)
