import asyncio
import json

import httpx
import pytest

from llm_gateway import LLMGateway, GatewayConfig

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
        rps_warmup=0.0,  # disable warmup for basic test
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


async def test_non_adaptive_warmup_ramps_rps(monkeypatch):
    """Non-adaptive mode with warmup starts at 1 rps and ramps to target."""
    config = GatewayConfig(
        litellm_base="http://test:9999/v1",
        litellm_key="test",
        enable_adaptive_runtime=False,
        concurrency=10,
        rps_limit=100.0,
        rps_warmup=1.0,  # 1 second warmup
        request_timeout=5,
        transport_stuck_seconds=0,
    )
    monkeypatch.setattr(
        "llm_gateway.gateway.httpx.AsyncClient",
        _mock_client_factory(_ok_response("ok")),
    )
    async with LLMGateway(config) as gw:
        # Immediately after init, limiter should be at low rps (started at 1.0)
        assert gw._limiter.target_rps < 50.0
        # Wait for warmup to complete
        await asyncio.sleep(1.5)
        # After warmup, should be at target
        assert gw._limiter.target_rps == 100.0


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


# ──── Integration tests (Task 18) ────────────────────────────────────────────


async def test_timeout_end_to_end(monkeypatch):
    """A slow handler triggers timeout, recorded in stats, breaker observes it."""
    config = GatewayConfig(
        litellm_base="http://test:9999/v1",
        litellm_key="test",
        concurrency=5,
        rps_limit=1000.0,
        request_timeout=1,  # 1 second timeout
        request_timeout_escalation=[1, 1, 1],
        adaptive_window_requests=5,
        adaptive_window_seconds=60.0,
        adaptive_open_base_cooldown=0.5,
        adaptive_open_max_cooldown=1.0,
        adaptive_min_observations_degraded=2,
        adaptive_min_observations_open=3,
        adaptive_min_failures_degraded=1,
        adaptive_min_failures_open=2,
        transport_stuck_seconds=0,
    )

    async def slow_handler(request):
        await asyncio.sleep(5)  # way longer than timeout
        return httpx.Response(200, json={"choices": [{"message": {"content": "late"}}], "usage": {}})

    monkeypatch.setattr(
        "llm_gateway.gateway.httpx.AsyncClient",
        _mock_client_factory(slow_handler),
    )
    async with LLMGateway(config) as gw:
        content, usage = await gw.call(
            [{"role": "user", "content": "hi"}], "gpt-4o-mini",
        )
        # Timeout should be captured, not propagated
        assert content is None
        assert "error" in usage
        # Breaker window should have observation
        snap = gw.runtime_snapshot
        assert snap["window_size"] >= 1


async def test_non_adaptive_retry_path(monkeypatch):
    """Non-adaptive mode retries on 429 with backoff and eventually succeeds."""
    config = GatewayConfig(
        litellm_base="http://test:9999/v1",
        litellm_key="test",
        enable_adaptive_runtime=False,
        concurrency=5,
        rps_limit=1000.0,
        rps_warmup=0.0,
        max_retries=3,
        request_timeout=10,
        request_timeout_escalation=[5, 5, 5, 5],
        transport_stuck_seconds=0,
    )

    call_count = {"n": 0}

    def handler(request):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "finally"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

    monkeypatch.setattr(
        "llm_gateway.gateway.httpx.AsyncClient",
        _mock_client_factory(handler),
    )
    async with LLMGateway(config) as gw:
        content, usage = await gw.call(
            [{"role": "user", "content": "hi"}], "gpt-4o-mini",
        )
    # Non-adaptive retries internally: first 2 calls are 429, 3rd succeeds
    assert call_count["n"] == 3
    assert content == "finally"
    assert usage["status_code"] == 200


async def test_recovery_loop_swaps_on_stuck_open(monkeypatch):
    """Recovery loop triggers a client swap when runtime is stuck in 'open'."""
    config = GatewayConfig(
        litellm_base="http://test:9999/v1",
        litellm_key="test",
        concurrency=5,
        rps_limit=1000.0,
        request_timeout=5,
        adaptive_window_requests=3,
        adaptive_window_seconds=60.0,
        adaptive_open_base_cooldown=60.0,   # long cooldown so it stays open
        adaptive_open_max_cooldown=60.0,
        adaptive_min_observations_degraded=1,
        adaptive_min_observations_open=2,
        adaptive_min_failures_degraded=1,
        adaptive_min_failures_open=2,
        adaptive_timeout_rate_open=0.10,
        transport_stuck_seconds=1,  # recovery kicks in after 1s open
    )

    monkeypatch.setattr(
        "llm_gateway.gateway.httpx.AsyncClient",
        _mock_client_factory(_fixed_status(429)),
    )
    async with LLMGateway(config) as gw:
        # Force circuit breaker into 'open' state
        for _ in range(6):
            await gw.call([{"role": "user", "content": "hi"}], "gpt-4o-mini")

        snap = gw.runtime_snapshot
        assert snap["state"] in ("degraded", "open", "probing")

        # If the state is open, wait for recovery loop to swap + enter probing
        if snap["state"] == "open":
            await asyncio.sleep(2.5)  # threshold=1s + some margin
            snap2 = gw.runtime_snapshot
            # Recovery loop should have pushed to probing (or beyond)
            assert snap2["state"] in ("probing", "degraded", "healthy")
