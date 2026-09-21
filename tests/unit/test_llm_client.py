"""LLMClient against the real Anthropic SDK, pointed at a local stub HTTP server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import BaseModel

from matsci_agent.infrastructure.llm.anthropic_client import LLMClient, LLMError


class Answer(BaseModel):
    queries: list[str]


def _message(text: str, stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


@pytest.fixture
def stub_api(monkeypatch):
    """Serve one canned reply; expose the requests received and the reply to change."""
    server_state = {"status": 200, "body": _message('{"queries": ["a", "b"]}'), "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            server_state["requests"].append((self.path, json.loads(self.rfile.read(length))))
            data = json.dumps(server_state["body"]).encode()
            self.send_response(server_state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    yield server_state
    server.shutdown()


async def test_structured_sends_schema_and_returns_parsed_model(stub_api) -> None:
    client = LLMClient(api_key="test-key", model="claude-sonnet-4-6", max_retries=0)
    result = await client.structured("find queries", Answer, system="be terse", max_tokens=100)

    assert result == Answer(queries=["a", "b"])
    path, body = stub_api["requests"][0]
    assert path == "/v1/messages"
    assert body["model"] == "claude-sonnet-4-6"
    assert body["system"] == "be terse"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["messages"][0]["content"] == "find queries"


async def test_api_errors_become_llm_errors(stub_api) -> None:
    stub_api["status"] = 400
    stub_api["body"] = {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
    client = LLMClient(api_key="test-key", model="claude-sonnet-4-6", max_retries=0)
    with pytest.raises(LLMError, match="Anthropic API error"):
        await client.structured("x", Answer)


async def test_unparsable_output_becomes_llm_error(stub_api) -> None:
    stub_api["body"] = _message("not json at all", stop_reason="max_tokens")
    client = LLMClient(api_key="test-key", model="claude-sonnet-4-6", max_retries=0)
    with pytest.raises(LLMError, match="did not match Answer"):
        await client.structured("x", Answer)


async def test_unconfigured_client_reports_unavailable() -> None:
    client = LLMClient(api_key=None, model="m")
    assert client.available is False
    with pytest.raises(LLMError, match="not configured"):
        await client.structured("x", Answer)
