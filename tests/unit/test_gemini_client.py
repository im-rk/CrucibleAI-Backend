"""GeminiClient against a local stub HTTP server shaped like the Gemini REST API."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import BaseModel

from matsci_agent.infrastructure.llm import gemini_client
from matsci_agent.infrastructure.llm.anthropic_client import LLMError
from matsci_agent.infrastructure.llm.gemini_client import GeminiClient


class Answer(BaseModel):
    queries: list[str]


def ok_body(text: str, finish: str = "STOP") -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}], "role": "model"}, "finishReason": finish}]}


def error_body(code: int, message: str, retry: str | None = None) -> dict:
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry}] if retry else []
    return {"error": {"code": code, "message": message, "details": details}}


@pytest.fixture
def stub():
    """Replies are consumed in order; the last one repeats. Records every request."""
    state = {"replies": [(200, ok_body('{"queries": ["a"]}'))], "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            state["requests"].append({"path": self.path, "key": self.headers.get("x-goog-api-key"),
                                      "body": json.loads(self.rfile.read(length))})
            status, body = state["replies"].pop(0) if len(state["replies"]) > 1 else state["replies"][0]
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/v1beta"
    yield state
    server.shutdown()


def client(stub, **kw) -> GeminiClient:
    return GeminiClient("test-key", "gemini-test", base_url=stub["url"], **kw)


async def test_structured_sends_schema_system_and_key_and_parses_the_reply(stub) -> None:
    result = await client(stub).structured("find queries", Answer, system="be terse", max_tokens=100)

    assert result == Answer(queries=["a"])
    req = stub["requests"][0]
    assert req["path"] == "/v1beta/models/gemini-test:generateContent" and req["key"] == "test-key"
    config = req["body"]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"]["properties"]["queries"]["type"] == "array"
    assert "maxOutputTokens" not in config  # would truncate: Gemini counts thinking against it
    assert req["body"]["systemInstruction"]["parts"][0]["text"] == "be terse"
    assert req["body"]["contents"][0]["parts"][0]["text"] == "find queries"


async def test_thought_parts_are_ignored(stub) -> None:
    stub["replies"] = [(200, {"candidates": [{"content": {"parts": [
        {"text": "thinking...", "thought": True}, {"text": '{"queries": ["b"]}'}]}, "finishReason": "STOP"}]})]
    assert await client(stub).structured("x", Answer) == Answer(queries=["b"])


async def test_rate_limits_are_retried_after_the_suggested_delay(stub, monkeypatch) -> None:
    waits = []

    async def fake_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(gemini_client.asyncio, "sleep", fake_sleep)
    stub["replies"] = [(429, error_body(429, "quota exceeded", retry="7s")), (503, error_body(503, "overloaded")),
                       (200, ok_body('{"queries": ["c"]}'))]
    assert await client(stub).structured("x", Answer) == Answer(queries=["c"])
    assert waits[0] == 7.0 and len(stub["requests"]) == 3


async def test_suggested_delays_are_capped(stub, monkeypatch) -> None:
    waits = []

    async def fake_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(gemini_client.asyncio, "sleep", fake_sleep)
    stub["replies"] = [(429, error_body(429, "quota", retry="3600s")), (200, ok_body('{"queries": []}'))]
    await client(stub).structured("x", Answer)
    assert waits == [gemini_client.MAX_RETRY_WAIT_S] and gemini_client.MAX_RETRY_WAIT_S <= gemini_client.MAX_TOTAL_WAIT_S


async def test_persistent_rate_limiting_pauses_the_client_instead_of_retrying_every_step(stub, monkeypatch) -> None:
    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr(gemini_client.asyncio, "sleep", fake_sleep)
    stub["replies"] = [(429, error_body(429, "You exceeded your current quota"))]
    c = client(stub, max_retries=2)
    with pytest.raises(LLMError, match="429"):
        await c.structured("x", Answer)
    assert len(stub["requests"]) == 3
    # not an account problem, but later steps must not each repeat the same waits
    assert not c.available and "paused" in c.unavailable_reason
    with pytest.raises(LLMError, match="paused"):
        await c.structured("x", Answer)
    assert len(stub["requests"]) == 3  # no further request while paused


async def test_the_client_recovers_when_the_pause_ends(stub, monkeypatch) -> None:
    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr(gemini_client.asyncio, "sleep", fake_sleep)
    stub["replies"] = [(503, error_body(503, "overloaded")), (503, error_body(503, "overloaded")),
                       (200, ok_body('{"queries": ["back"]}'))]
    c = client(stub, max_retries=1)
    with pytest.raises(LLMError):
        await c.structured("x", Answer)
    assert not c.available
    now = gemini_client.time.monotonic()
    monkeypatch.setattr(gemini_client.time, "monotonic", lambda: now + gemini_client.COOLDOWN_S + 1)
    assert c.available and c.unavailable_reason is None
    assert await c.structured("x", Answer) == Answer(queries=["back"])


async def test_a_long_suggested_delay_gives_up_instead_of_stalling_the_run(stub, monkeypatch) -> None:
    waits = []

    async def fake_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(gemini_client.asyncio, "sleep", fake_sleep)
    stub["replies"] = [(429, error_body(429, "per minute", retry="25s"))]
    c = client(stub)
    with pytest.raises(LLMError):
        await c.structured("x", Answer)
    # a 25 s hint is capped to MAX_RETRY_WAIT_S; the next one would exceed the total budget
    assert sum(waits) <= gemini_client.MAX_TOTAL_WAIT_S and len(stub["requests"]) <= 3
    assert not c.available


@pytest.mark.parametrize("status, message", [(400, "API key not valid. [API_KEY_INVALID]"), (403, "permission denied")])
async def test_a_rejected_key_disables_the_client(stub, status, message) -> None:
    stub["replies"] = [(status, error_body(status, message))]
    c = client(stub)
    with pytest.raises(LLMError, match="rejected the API key"):
        await c.structured("x", Answer)
    assert not c.available
    with pytest.raises(LLMError):
        await c.structured("x", Answer)
    assert len(stub["requests"]) == 1  # no further calls


async def test_an_unavailable_model_disables_the_client_with_a_clear_reason(stub) -> None:
    stub["replies"] = [(404, error_body(404, "models/gemini-2.5-flash is no longer available to new users"))]
    c = client(stub)
    with pytest.raises(LLMError, match="not available to this key"):
        await c.structured("x", Answer)
    assert "gemini-test" in c.unavailable_reason


async def test_truncated_or_blocked_output_is_an_llm_error(stub) -> None:
    stub["replies"] = [(200, ok_body('{"queries": ["unterminated', finish="MAX_TOKENS"))]
    with pytest.raises(LLMError, match="MAX_TOKENS"):
        await client(stub).structured("x", Answer)
    stub["replies"] = [(200, {"promptFeedback": {"blockReason": "SAFETY"}})]
    with pytest.raises(LLMError, match="SAFETY"):
        await client(stub).structured("x", Answer)


async def test_no_key_means_unavailable_and_no_request() -> None:
    c = GeminiClient(None, "m")
    assert not c.available and "GEMINI_API_KEY" in c.unavailable_reason
    with pytest.raises(LLMError, match="not configured"):
        await c.structured("x", Answer)


def test_provider_selection(monkeypatch) -> None:
    from matsci_agent.api import dependencies as deps
    from matsci_agent.infrastructure.llm.anthropic_client import LLMClient

    def pick(provider, anthropic_key, gemini_key, groq_key=None):
        monkeypatch.setattr(deps.settings, "llm_provider", provider)
        monkeypatch.setattr(deps.settings, "anthropic_api_key", anthropic_key)
        monkeypatch.setattr(deps.settings, "gemini_api_key", gemini_key)
        monkeypatch.setattr(deps.settings, "groq_api_key", groq_key)
        deps.get_llm_client.cache_clear()
        try:
            return type(deps.build_provider_client())
        finally:
            deps.get_llm_client.cache_clear()

    assert pick("gemini", "a", "g") is GeminiClient
    assert pick("anthropic", "a", "g") is LLMClient
    assert pick("auto", "a", "g") is LLMClient  # both set: Anthropic wins
    assert pick("auto", None, "g") is GeminiClient
    assert pick("auto", None, None) is LLMClient  # unavailable either way; Anthropic is the default
    from matsci_agent.infrastructure.llm.groq_client import GroqClient

    assert pick("groq", "a", "g", "q") is GroqClient
    assert pick("auto", None, None, "q") is GroqClient  # only Groq is configured
    assert pick("auto", None, "g", "q") is GeminiClient  # Gemini comes before Groq
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        pick("openai", "a", None)


async def test_an_exhausted_daily_quota_disables_immediately_without_retrying(stub, monkeypatch) -> None:
    waits = []

    async def fake_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(gemini_client.asyncio, "sleep", fake_sleep)
    body = error_body(429, "You exceeded your current quota", retry="17s")
    body["error"]["details"].append({
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}],
    })
    stub["replies"] = [(429, body)]
    c = client(stub)
    with pytest.raises(LLMError, match="free-tier daily quota of 20 requests"):
        await c.structured("x", Answer)
    assert waits == [] and len(stub["requests"]) == 1  # no pointless retries
    assert not c.available and "resets daily" in c.unavailable_reason
