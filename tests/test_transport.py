import httpx
import pytest

from llm_gateway.config import GatewayConfig
from llm_gateway.transport import async_llm_call, RequestStats


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok_body(content):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


async def test_success_returns_content():
    def handler(request):
        return httpx.Response(200, json=_ok_body("hello world"))

    cfg = GatewayConfig()
    async with _client(handler) as client:
        content, usage = await async_llm_call(
            client, [{"role": "user", "content": "hi"}], "gpt-4o-mini",
            config=cfg, adaptive_mode=True,
        )
    assert content == "hello world"
    assert usage["status_code"] == 200
    assert usage["prompt_tokens"] == 10


async def test_strips_markdown_fence():
    def handler(request):
        return httpx.Response(200, json=_ok_body('```json\n{"a": 1}\n```'))

    cfg = GatewayConfig()
    async with _client(handler) as client:
        content, usage = await async_llm_call(
            client, [{"role": "user", "content": "hi"}], "gpt-4o-mini",
            config=cfg, adaptive_mode=True,
        )
    assert content.strip() == '{"a": 1}'


async def test_missing_choices_is_parse_error():
    def handler(request):
        return httpx.Response(200, json={"unexpected": "shape"})

    cfg = GatewayConfig()
    async with _client(handler) as client:
        content, usage = await async_llm_call(
            client, [{"role": "user", "content": "hi"}], "gpt-4o-mini",
            config=cfg, adaptive_mode=True,
        )
    assert content is None
    assert usage.get("parse_error") is True


async def test_429_adaptive_returns_immediately():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(429, text="rate limited")

    cfg = GatewayConfig()
    async with _client(handler) as client:
        content, usage = await async_llm_call(
            client, [{"role": "user", "content": "hi"}], "gpt-4o-mini",
            config=cfg, adaptive_mode=True,
        )
    assert content is None
    assert usage["status_code"] == 429
    assert usage["non_retryable"] is False
    assert calls["n"] == 1


async def test_401_non_retryable():
    def handler(request):
        return httpx.Response(401, text="unauthorized")

    cfg = GatewayConfig()
    async with _client(handler) as client:
        content, usage = await async_llm_call(
            client, [{"role": "user", "content": "hi"}], "gpt-4o-mini",
            config=cfg, adaptive_mode=True,
        )
    assert content is None
    assert usage["status_code"] == 401
    assert usage["non_retryable"] is True


async def test_400_content_filter_non_retryable():
    def handler(request):
        return httpx.Response(400, text="content_filter: blocked")

    cfg = GatewayConfig()
    async with _client(handler) as client:
        content, usage = await async_llm_call(
            client, [{"role": "user", "content": "hi"}], "gpt-4o-mini",
            config=cfg, adaptive_mode=True,
        )
    assert content is None
    assert usage.get("content_filtered") is True
    assert usage["non_retryable"] is True


async def test_reasoning_model_drops_temperature():
    captured = {}

    def handler(request):
        import json as _json
        captured.update(_json.loads(request.content))
        return httpx.Response(200, json=_ok_body("ok"))

    cfg = GatewayConfig()
    async with _client(handler) as client:
        await async_llm_call(
            client, [{"role": "user", "content": "hi"}], "o1-preview",
            config=cfg, adaptive_mode=True,
        )
    assert "temperature" not in captured
    assert "max_completion_tokens" in captured


def test_request_stats():
    stats = RequestStats()
    stats.record(200)
    stats.record(200)
    stats.record(429)
    stats.record_timeout()
    d = stats.to_dict()
    assert d["success"] == 2
    assert d["errors"]["429"] == 1
    assert d["timeouts"] == 1
