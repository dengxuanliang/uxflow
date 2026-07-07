import asyncio
import json

import httpx
import pytest

from llm_gateway import LLMGateway, GatewayConfig
from llm_gateway.outcomes import OutcomeClass

# Capture the real AsyncClient before any test patches
# ``llm_gateway.gateway.httpx.AsyncClient`` (which is the shared httpx module
# attribute). The mock factories below reference this so they build a genuine
# transport-backed client instead of recursing into the patched name.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _ok_response(content="hello"):
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        })
    return handler


def _fixed_status(code, body="error"):
    def handler(request):
        return httpx.Response(code, text=body)
    return handler


def _mock_client_factory(handler):
    """Return a factory callable that builds a real AsyncClient with a MockTransport.

    Used with monkeypatch to replace httpx.AsyncClient in the gateway module.
    The factory accepts **kw so it matches the AsyncClient(**kwargs) call site.
    """
    def factory(**kw):
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler))
    return factory


@pytest.fixture
def gateway_config():
    return GatewayConfig(
        litellm_base="http://test:9999/v1",
        litellm_key="test-key",
        concurrency=5,
        rps_limit=1000.0,
        request_timeout=5,
        request_timeout_escalation=[2, 3, 5],
        adaptive_window_requests=5,
        adaptive_window_seconds=60.0,
        adaptive_open_base_cooldown=0.5,
        adaptive_open_max_cooldown=1.0,
        adaptive_min_observations_degraded=2,
        adaptive_min_observations_open=4,
        adaptive_min_failures_degraded=1,
        adaptive_min_failures_open=2,
        fatal_abort_streak_limit=3,
        fatal_abort_global_rate_min_obs=100,
        transport_stuck_seconds=0,
    )


async def test_successful_call(gateway_config, monkeypatch):
    monkeypatch.setattr(
        "llm_gateway.gateway.httpx.AsyncClient",
        _mock_client_factory(_ok_response("hi")
    )
    )
    async with LLMGateway(gateway_config) as gw:
        text, usage = await gw.call(
            [{"role": "user", "content": "hello"}], "gpt-4o-mini",
        )
    assert text == "hi"
    assert usage["prompt_tokens"] == 5
    assert usage["status_code"] == 200


async def test_truncation_applied(gateway_config, monkeypatch):
    captured = {}

    def handler(request):
        body = json.loads(request.content)
        captured["messages"] = body["messages"]
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        })

    monkeypatch.setattr(
        "llm_gateway.gateway.httpx.AsyncClient",
        _mock_client_factory(handler)
    )
    gateway_config.max_request_tokens = 100
    long_msg = "x" * 50000

    async with LLMGateway(gateway_config) as gw:
        await gw.call([{"role": "user", "content": long_msg}], "gpt-4o-mini")
    sent_content = captured["messages"][0]["content"]
    assert len(sent_content) < 5000


async def test_circuit_breaker_degraded_on_429(gateway_config, monkeypatch):
    monkeypatch.setattr(
        "llm_gateway.gateway.httpx.AsyncClient",
        _mock_client_factory(_fixed_status(429)
    )
    )
    async with LLMGateway(gateway_config) as gw:
        for _ in range(6):
            await gw.call([{"role": "user", "content": "hi"}], "gpt-4o-mini")
        snap = gw.runtime_snapshot
        assert snap["state"] in ("degraded", "open", "probing")


async def test_should_abort_on_auth_streak(gateway_config, monkeypatch):
    monkeypatch.setattr(
        "llm_gateway.gateway.httpx.AsyncClient",
        _mock_client_factory(_fixed_status(401)
    )
    )
    async with LLMGateway(gateway_config) as gw:
        for _ in range(3):
            await gw.call([{"role": "user", "content": "hi"}], "gpt-4o-mini")
        assert gw.should_abort
        assert gw.abort_reason != ""


async def test_http_stats(gateway_config, monkeypatch):
    call_count = {"n": 0}

    def handler(request):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            })
        return httpx.Response(429, text="slow down")

    monkeypatch.setattr(
        "llm_gateway.gateway.httpx.AsyncClient",
        _mock_client_factory(handler)
    )
    async with LLMGateway(gateway_config) as gw:
        for _ in range(3):
            await gw.call([{"role": "user", "content": "hi"}], "gpt-4o-mini")
        stats = gw.http_stats
    assert stats["success"] == 2
    assert stats["errors"].get("429") == 1


async def test_non_adaptive_mode(monkeypatch):
    config = GatewayConfig(
        litellm_base="http://test:9999/v1",
        litellm_key="test",
        enable_adaptive_runtime=False,
        concurrency=2,
        rps_limit=1000.0,
        request_timeout=5,
        transport_stuck_seconds=0,
    )
    monkeypatch.setattr(
        "llm_gateway.gateway.httpx.AsyncClient",
        _mock_client_factory(_ok_response("hey")
    )
    )
    async with LLMGateway(config) as gw:
        text, usage = await gw.call([{"role": "user", "content": "hi"}], "gpt-4o-mini")
    assert text == "hey"
    assert usage["status_code"] == 200


async def test_exception_from_transport_does_not_propagate(gateway_config, monkeypatch):
    """When transport raises unexpectedly, gateway.call returns (None, usage)
    instead of letting the exception escape — and the permit is released."""

    def handler(request):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        "llm_gateway.gateway.httpx.AsyncClient",
        _mock_client_factory(handler),
    )
    async with LLMGateway(gateway_config) as gw:
        content, usage = await gw.call(
            [{"role": "user", "content": "hi"}], "gpt-4o-mini",
        )
        # No exception propagated; error captured in usage
        assert content is None
        assert "error" in usage
        # Permit was released — the gate is not stuck (a follow-up call works)
        content2, _ = await gw.call(
            [{"role": "user", "content": "again"}], "gpt-4o-mini",
        )
        assert content2 is None


async def test_exception_feeds_circuit_breaker(gateway_config, monkeypatch):
    """Repeated transport exceptions feed the circuit breaker window (no blind spot).

    ConnectError → TRANSIENT_ERROR, which is observed in the window but does not
    by itself trigger degradation (only timeout/overload/abnormal do). Verify the
    window received the observations.
    """

    def handler(request):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(
        "llm_gateway.gateway.httpx.AsyncClient",
        _mock_client_factory(handler),
    )
    async with LLMGateway(gateway_config) as gw:
        for _ in range(4):
            await gw.call([{"role": "user", "content": "hi"}], "gpt-4o-mini")
        # The breaker's sliding window should have observations
        snap = gw.runtime_snapshot
        assert snap["window_size"] == 4
