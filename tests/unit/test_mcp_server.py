"""The MCP server: protocol behaviour in-process, and the real stdio transport in a subprocess."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from matsci_agent.mcp.server import INVALID_PARAMS, METHOD_NOT_FOUND, PARSE_ERROR, McpServer, McpTool, _object, build_server


def server() -> McpServer:
    srv = McpServer()

    async def echo(args):
        return {"you_said": args["text"]}

    async def boom(args):
        raise ValueError("kaboom")

    srv.add(McpTool("echo", "Echo text.", _object({"text": {"type": "string"}}, ["text"]), echo))
    srv.add(McpTool("boom", "Always fails.", _object({}, []), boom))
    return srv


def call(srv, message):
    return asyncio.run(srv.handle_line(json.dumps(message) if not isinstance(message, str) else message))


def rpc(method, params=None, id=1):
    return {"jsonrpc": "2.0", "id": id, "method": method, **({"params": params} if params is not None else {})}


def test_initialize_negotiates_and_advertises_tools() -> None:
    reply = call(server(), rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t"}}))
    assert reply["id"] == 1 and reply["result"]["protocolVersion"] == "2025-03-26"
    assert "tools" in reply["result"]["capabilities"] and reply["result"]["serverInfo"]["name"] == "crucibleai"


def test_notifications_get_no_reply() -> None:
    assert call(server(), {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert call(server(), {"jsonrpc": "2.0", "method": "something/unknown"}) is None  # an unknown notification is ignored


def test_ping_and_tools_list() -> None:
    srv = server()
    assert call(srv, rpc("ping"))["result"] == {}
    tools = call(srv, rpc("tools/list", id=2))["result"]["tools"]
    assert [t["name"] for t in tools] == ["echo", "boom"]
    assert tools[0]["inputSchema"]["required"] == ["text"] and tools[0]["description"]


def test_a_tool_call_returns_text_content() -> None:
    reply = call(server(), rpc("tools/call", {"name": "echo", "arguments": {"text": "héllo"}}))
    result = reply["result"]
    assert result["isError"] is False and json.loads(result["content"][0]["text"]) == {"you_said": "héllo"}


def test_a_failing_tool_is_an_error_result_not_a_protocol_error() -> None:
    reply = call(server(), rpc("tools/call", {"name": "boom", "arguments": {}}))
    assert "error" not in reply and reply["result"]["isError"] is True and "kaboom" in reply["result"]["content"][0]["text"]


def test_protocol_errors_use_json_rpc_codes() -> None:
    srv = server()
    assert call(srv, "{not json")["error"]["code"] == PARSE_ERROR
    assert call(srv, rpc("nope"))["error"]["code"] == METHOD_NOT_FOUND
    assert call(srv, rpc("tools/call", {"name": "missing", "arguments": {}}))["error"]["code"] == INVALID_PARAMS
    assert "text" in call(srv, rpc("tools/call", {"name": "echo", "arguments": {}}))["error"]["message"]
    assert call(srv, '{"jsonrpc": "1.0", "method": "ping", "id": 1}')["error"]["code"] == -32600


def test_the_real_tool_set_is_listed_without_loading_heavy_dependencies() -> None:
    names = [t.name for t in build_server().tools.values()]
    assert names == ["search_literature", "lookup_material", "predict_property", "screen_materials", "plan_research", "run_research"]
    for tool in build_server().tools.values():
        assert tool.schema["type"] == "object" and tool.description


def test_invalid_formulas_are_rejected_by_the_lookup_tool() -> None:
    srv = build_server()
    reply = call(srv, rpc("tools/call", {"name": "lookup_material", "arguments": {"formula": "notaformula"}}))
    assert reply["result"]["isError"] is True and "not a valid chemical formula" in reply["result"]["content"][0]["text"]


def test_the_stdio_transport_speaks_newline_delimited_json_rpc() -> None:
    """Spawn the server as a real process, as an MCP client would, and hold a short conversation."""
    root = Path(__file__).resolve().parents[2]
    messages = [
        rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "pytest"}}, id=1),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        rpc("tools/list", id=2),
        rpc("tools/call", {"name": "lookup_material", "arguments": {"formula": "notaformula"}}, id=3),
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "matsci_agent.mcp"], input="\n".join(json.dumps(m) for m in messages) + "\n",
        capture_output=True, text=True, timeout=120, cwd=root, env={**__import__("os").environ, "PYTHONPATH": str(root / "src")},
    )
    replies = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    assert [r["id"] for r in replies] == [1, 2, 3]  # the notification produced no reply
    assert replies[0]["result"]["serverInfo"]["name"] == "crucibleai"
    assert len(replies[1]["result"]["tools"]) == 6
    assert replies[2]["result"]["isError"] is True
