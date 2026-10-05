"""GroqClient against a local stub HTTP server shaped like Groq's OpenAI-compatible API."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import BaseModel

from matsci_agent.infrastructure.llm import groq_client
from matsci_agent.infrastructure.llm.anthropic_client import LLMError
from matsci_agent.infrastructure.llm.groq_client import GroqClient


class Answer(BaseModel):
    queries: list[str]


def ok_body(text: str, finish: str = "stop") -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}]}


def error_body(code: int, message: str) -> dict:
    return {"error": {"message": message, "type": "error", "code": code}}


@pytest.fixture
def stub():
    """Replies are (status, body[, headers]) consumed in order; the last one repeats."""
    state = {"replies": [(200, ok_body('{"queries": ["a"]}'))], "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            state["requests"].append({"path": self.path, "auth": self.headers.get("Authorization"),
                                      "body": json.loads(self.rfile.read(length))})
            reply = state["replies"].pop(0) if len(state["replies"]) > 1 else state["replies"][0]
            status, body = reply[0], reply[1]
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (reply[2] if len(reply) > 2 else {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/openai/v1"
    yield state
    server.shutdown()


def client(stub, model="some/model", **kw) -> GroqClient:
    return GroqClient("test-key", model, base_url=stub["url"], **kw)


@pytest.fixture
def no_sleep(monkeypatch):
    waits: list[float] = []

    async def fake_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(groq_client.asyncio, "sleep", fake_sleep)
    return waits


async def test_structured_sends_schema_system_and_key_and_parses_the_reply(stub) -> None:
    result = await client(stub).structured("find queries", Answer, system="be terse", max_tokens=100)

    assert result == Answer(queries=["a"])
    req = stub["requests"][0]
    assert req["path"] == "/openai/v1/chat/completions" and req["auth"] == "Bearer test-key"
    body = req["body"]
    assert body["model"] == "some/model" and body["temperature"] == 0
    assert body["messages"] == [{"role": "system", "content": "be terse"}, {"role": "user", "content": "find queries"}]
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["name"] == "Answer"
    assert fmt["json_schema"]["schema"]["properties"]["queries"]["type"] == "array"
    assert "max_tokens" not in body and "reasoning_effort" not in body


async def test_reasoning_effort_is_only_sent_to_gpt_oss_models(stub) -> None:
    await client(stub, model="openai/gpt-oss-120b").structured("x", Answer)
    assert stub["requests"][0]["body"]["reasoning_effort"] == "low"


async def test_a_rate_limit_is_retried_after_the_retry_after_header(stub, no_sleep) -> None:
    stub["replies"] = [(429, error_body(429, "Rate limit reached on tokens per minute (TPM)"), {"Retry-After": "7"}),
                       (503, error_body(503, "overloaded")), (200, ok_body('{"queries": ["c"]}'))]
    assert await client(stub).structured("x", Answer) == Answer(queries=["c"])
    assert no_sleep[0] == 7.0 and len(stub["requests"]) == 3


async def test_suggested_delays_are_capped(stub, no_sleep) -> None:
    stub["replies"] = [(429, error_body(429, "tokens per minute"), {"Retry-After": "3600"}), (200, ok_body('{"queries": []}'))]
    await client(stub).structured("x", Answer)
    assert no_sleep == [groq_client.MAX_RETRY_WAIT_S]


async def test_persistent_rate_limiting_pauses_the_client_instead_of_retrying_every_step(stub, no_sleep) -> None:
    stub["replies"] = [(429, error_body(429, "tokens per minute (TPM)"))]
    c = client(stub, max_retries=2)
    with pytest.raises(LLMError, match="429"):
        await c.structured("x", Answer)
    assert len(stub["requests"]) == 3
    assert not c.available and "paused" in c.unavailable_reason
    with pytest.raises(LLMError, match="paused"):
        await c.structured("x", Answer)
    assert len(stub["requests"]) == 3  # no further request while paused


async def test_the_client_recovers_when_the_pause_ends(stub, no_sleep, monkeypatch) -> None:
    stub["replies"] = [(503, error_body(503, "overloaded")), (503, error_body(503, "overloaded")),
                       (200, ok_body('{"queries": ["back"]}'))]
    c = client(stub, max_retries=1)
    with pytest.raises(LLMError):
        await c.structured("x", Answer)
    assert not c.available
    now = groq_client.time.monotonic()
    monkeypatch.setattr(groq_client.time, "monotonic", lambda: now + groq_client.COOLDOWN_S + 1)
    assert c.available and c.unavailable_reason is None
    assert await c.structured("x", Answer) == Answer(queries=["back"])


async def test_a_long_retry_after_gives_up_instead_of_stalling_the_run(stub, no_sleep) -> None:
    stub["replies"] = [(429, error_body(429, "tokens per minute"), {"Retry-After": "25"})]
    c = client(stub)
    with pytest.raises(LLMError):
        await c.structured("x", Answer)
    assert sum(no_sleep) <= groq_client.MAX_TOTAL_WAIT_S and not c.available


async def test_a_daily_limit_disables_immediately_without_retrying(stub, no_sleep) -> None:
    stub["replies"] = [(429, error_body(429, "Rate limit reached for model `x` on requests per day (RPD): Limit 1000"))]
    c = client(stub)
    with pytest.raises(LLMError, match="daily limit"):
        await c.structured("x", Answer)
    assert not c.available and "resets daily" in c.unavailable_reason
    assert len(stub["requests"]) == 1 and no_sleep == []


@pytest.mark.parametrize("status", [401, 403])
async def test_a_rejected_key_disables_the_client(stub, status) -> None:
    stub["replies"] = [(status, error_body(status, "Invalid API Key"))]
    c = client(stub)
    with pytest.raises(LLMError, match="rejected the API key"):
        await c.structured("x", Answer)
    assert not c.available
    with pytest.raises(LLMError):
        await c.structured("x", Answer)
    assert len(stub["requests"]) == 1  # no further calls


async def test_an_unavailable_model_disables_the_client_with_a_clear_reason(stub) -> None:
    stub["replies"] = [(404, error_body(404, "The model `some/model` does not exist"))]
    c = client(stub)
    with pytest.raises(LLMError, match="not available to this key"):
        await c.structured("x", Answer)
    assert "some/model" in c.unavailable_reason


async def test_truncated_output_is_an_llm_error(stub) -> None:
    stub["replies"] = [(200, ok_body('{"queries": ["unterminated', finish="length"))]
    with pytest.raises(LLMError, match="length"):
        await client(stub).structured("x", Answer)


async def test_no_key_means_unavailable_and_no_request(stub) -> None:
    c = GroqClient(None, "m", base_url=stub["url"])
    assert not c.available and "GROQ_API_KEY" in c.unavailable_reason
    with pytest.raises(LLMError):
        await c.structured("x", Answer)
    assert stub["requests"] == []
