import asyncio

import httpx
import pytest

from llm_gateway.recovery import SwappableAsyncClient, transport_recovery_loop


def _factory():
    return httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"ok": True})
    ))


async def test_swappable_forwards_attributes():
    async with SwappableAsyncClient(_factory) as client:
        resp = await client.post("http://x/v1/chat/completions", json={})
        assert resp.status_code == 200


async def test_swap_replaces_underlying():
    async with SwappableAsyncClient(_factory) as client:
        old = client.underlying
        old_id, new_id = await client.swap()
        assert old_id != new_id
        assert client.underlying is not old


class _FakeRuntime:
    def __init__(self):
        self.state = "open"
        self.probing_calls = 0

    def _enter_probing(self, reason=""):
        self.probing_calls += 1
        self.state = "probing"


async def test_recovery_loop_swaps_when_stuck():
    runtime = _FakeRuntime()
    swapped = {"count": 0}

    async with SwappableAsyncClient(_factory) as client:
        orig_swap = client.swap

        async def counting_swap():
            swapped["count"] += 1
            return await orig_swap()

        client.swap = counting_swap

        loop_task = asyncio.create_task(transport_recovery_loop(
            runtime=runtime,
            swappable=client,
            threshold_seconds=1,
            cancel_in_flight=lambda: [],
            pprint=lambda *a, **k: None,
            label="test",
        ))
        await asyncio.sleep(2.5)
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

    assert swapped["count"] >= 1
    assert runtime.probing_calls >= 1
