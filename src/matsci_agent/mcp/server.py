"""A Model Context Protocol (MCP) server exposing CrucibleAI's capabilities as tools.

MCP is JSON-RPC 2.0 over a transport; this speaks the stdio transport (one JSON message per line on
stdin, replies on stdout, logs on stderr only). Any MCP client (Claude Desktop, an agent framework,
the MCP inspector) can then call the same literature search, materials lookup, ML prediction,
screening and full research pipeline that the web UI uses.

Implemented: initialize, notifications/initialized, ping, tools/list, tools/call. Tool failures are
reported the MCP way (a result with isError=true), protocol errors with JSON-RPC error codes.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TextIO

from matsci_agent import __version__

PROTOCOL_VERSION = "2024-11-05"

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602


@dataclass
class McpTool:
    name: str
    description: str
    schema: dict[str, Any]
    run: Callable[[dict[str, Any]], Awaitable[Any]]

    def listing(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "inputSchema": self.schema}


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


@dataclass
class McpServer:
    tools: dict[str, McpTool] = field(default_factory=dict)
    name: str = "crucibleai"

    def add(self, tool: McpTool) -> None:
        self.tools[tool.name] = tool

    # ---- one message in, at most one message out ----------------------------------------------

    async def handle_line(self, line: str) -> dict | None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return self._error(None, PARSE_ERROR, "Parse error")
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or "method" not in message:
            return self._error(message.get("id") if isinstance(message, dict) else None, INVALID_REQUEST, "Invalid request")
        return await self.handle(message)

    async def handle(self, message: dict) -> dict | None:
        msg_id, method, params = message.get("id"), message["method"], message.get("params") or {}
        is_notification = "id" not in message

        if method.startswith("notifications/"):
            return None
        if is_notification:  # unknown notifications are ignored, never answered
            return None
        if method == "initialize":
            return self._result(msg_id, {
                "protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": self.name, "version": __version__},
                "instructions": "Materials-discovery research tools: search papers, look up and predict material properties, screen by band gap, and run the full research pipeline.",
            })
        if method == "ping":
            return self._result(msg_id, {})
        if method == "tools/list":
            return self._result(msg_id, {"tools": [t.listing() for t in self.tools.values()]})
        if method == "tools/call":
            return await self._call(msg_id, params)
        return self._error(msg_id, METHOD_NOT_FOUND, f"Method not found: {method}")

    async def _call(self, msg_id: Any, params: dict) -> dict:
        tool = self.tools.get(str(params.get("name", "")))
        if tool is None:
            return self._error(msg_id, INVALID_PARAMS, f"Unknown tool: {params.get('name')!r}")
        arguments = params.get("arguments") or {}
        missing = [k for k in tool.schema.get("required", []) if k not in arguments]
        if missing:
            return self._error(msg_id, INVALID_PARAMS, f"Missing argument(s) for {tool.name}: {', '.join(missing)}")
        try:
            output = await tool.run(arguments)
            text = output if isinstance(output, str) else json.dumps(output, indent=1, default=str, ensure_ascii=False)
            return self._result(msg_id, {"content": [{"type": "text", "text": text}], "isError": False})
        except Exception as exc:  # noqa: BLE001 - a failing tool is a tool result, not a protocol error
            return self._result(msg_id, {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True})

    @staticmethod
    def _result(msg_id: Any, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _error(msg_id: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}

    # ---- transport ------------------------------------------------------------------------------

    async def serve(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, stdin.readline)
            if not line:  # EOF: the client closed the pipe
                return
            if not line.strip():
                continue
            reply = await self.handle_line(line)
            if reply is not None:
                stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
                stdout.flush()


def build_server() -> McpServer:
    """The CrucibleAI tool set. Heavy dependencies load lazily, on the first call that needs them."""
    server = McpServer()

    async def search_literature(args: dict) -> Any:
        from matsci_agent.api.dependencies import get_hybrid_search

        hits = await asyncio.to_thread(get_hybrid_search().search, str(args["query"]), int(args.get("limit", 6)))
        return [
            {"id": h["id"], "title": h["metadata"].get("title"), "year": h["metadata"].get("year"),
             "arxiv_id": h["metadata"].get("arxiv_id"), "similarity": round(float(h.get("similarity", 0)), 3), "text": h["text"][:500]}
            for h in hits
        ]

    async def lookup_material(args: dict) -> Any:
        from matsci_agent.api.dependencies import get_mp_client
        from matsci_agent.domain.chemistry import elements_in

        formula = str(args["formula"]).strip()
        if not elements_in(formula):
            raise ValueError(f"'{formula}' is not a valid chemical formula")
        client = get_mp_client()
        props = await asyncio.to_thread(client.get_properties, formula)
        return {"formula": formula, "source": "Materials Project" if client.online else "offline reference table",
                "properties": [p.model_dump() for p in props]}

    async def predict_property(args: dict) -> Any:
        from matsci_agent.api.dependencies import get_krr_predictor
        from matsci_agent.domain.chemistry import elements_in

        formula = str(args["formula"]).strip()
        if not elements_in(formula):
            raise ValueError(f"'{formula}' is not a valid chemical formula")
        result = await asyncio.to_thread(get_krr_predictor().predict, formula, str(args.get("property", "band_gap")))
        if "error" in result:
            raise ValueError(result["error"])
        return result

    async def screen_materials(args: dict) -> Any:
        from matsci_agent.api.dependencies import get_mp_client

        client = get_mp_client()
        rows = await asyncio.to_thread(
            client.screen_by_criteria, args.get("bandgap_min"), args.get("bandgap_max"), True, None, int(args.get("limit", 10))
        )
        return {"source": "Materials Project" if client.online else "offline reference table", "results": rows}

    async def plan_research(args: dict) -> Any:
        from matsci_agent.api.dependencies import get_planner

        warnings: list[str] = []
        plan = await get_planner().plan(str(args["goal"]), warnings)
        return {"plan": plan, "warnings": warnings}

    async def run_research(args: dict) -> Any:
        from matsci_agent.api.dependencies import get_research_service
        from matsci_agent.domain.schemas import ResearchRequest

        report = await get_research_service().run(ResearchRequest(query=str(args["goal"]), max_results=int(args.get("max_results", 10))))
        return {
            "mode": report["mode"], "criteria": [c["label"] for c in report["criteria"]],
            "candidates": [{"formula": c["formula"] or c["name"], "status": c["status"], "score": c["score"], "origin": c["origin"]} for c in report["candidates"]],
            "hypotheses": [{"statement": h["statement"], "confidence": h["confidence"], "grounded": h.get("grounded")} for h in report["hypotheses"]],
            "grounding": {k: report["grounding"].get(k) for k in ("score", "is_grounded", "history")},
            "knowledge_gaps": report["knowledge_gaps"], "warnings": report["warnings"], "duration_ms": report["duration_ms"],
        }

    server.add(McpTool("search_literature", "Search the materials-science paper corpus (hybrid keyword + semantic). Returns passages with titles and ids.",
                       _object({"query": {"type": "string", "description": "What to look for"}, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}, ["query"]), search_literature))
    server.add(McpTool("lookup_material", "Look a chemical formula up in Materials Project (or the offline experimental table without a key).",
                       _object({"formula": {"type": "string", "description": "e.g. GaN or Li7La3Zr2O12"}}, ["formula"]), lookup_material))
    server.add(McpTool("predict_property", "Machine-learning estimate (kernel ridge regression) of a property from the formula alone, with uncertainty. An estimate, not a measurement.",
                       _object({"formula": {"type": "string"}, "property": {"type": "string", "enum": ["band_gap", "formation_energy_per_atom", "density"]}}, ["formula"]), predict_property))
    server.add(McpTool("screen_materials", "Find materials by band-gap range (eV).",
                       _object({"bandgap_min": {"type": "number"}, "bandgap_max": {"type": "number"}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, []), screen_materials))
    server.add(McpTool("plan_research", "Turn a research goal into criteria, literature queries and material families, without running it.",
                       _object({"goal": {"type": "string"}}, ["goal"]), plan_research))
    server.add(McpTool("run_research", "Run the full multi-agent research pipeline for a goal (can take minutes). Returns candidates, grounded hypotheses and knowledge gaps.",
                       _object({"goal": {"type": "string"}, "max_results": {"type": "integer", "minimum": 1, "maximum": 30}}, ["goal"]), run_research))
    return server


def main() -> None:
    asyncio.run(build_server().serve())


if __name__ == "__main__":
    main()
