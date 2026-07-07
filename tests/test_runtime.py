import asyncio
import gc
import time

import pytest

from llm_gateway.outcomes import OutcomeClass, RequestOutcome
from llm_gateway.runtime import (
    DynamicConcurrencyGate,
    AdaptiveRateLimiter,
    AdaptiveLLMRuntime,
    FatalFailureMonitor,
)


@pytest.mark.asyncio
async def test_gate_acquire_release():
    gate = DynamicConcurrencyGate(initial_limit=2)
    p1 = await gate.acquire()
    p2 = await gate.acquire()
    assert gate.in_flight == 2
    p1.release()
    await asyncio.sleep(0.01)
    assert gate.in_flight == 1
    p2.release()
    await asyncio.sleep(0.01)
    assert gate.in_flight == 0


@pytest.mark.asyncio
async def test_gate_blocks_at_limit():
    gate = DynamicConcurrencyGate(initial_limit=1)
    p1 = await gate.acquire()
    blocked = asyncio.create_task(gate.acquire())
    await asyncio.sleep(0.05)
    assert not blocked.done()
    p1.release()
    p2 = await asyncio.wait_for(blocked, timeout=1.0)
    p2.release()
    await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_gate_set_limit_wakes_waiters():
    gate = DynamicConcurrencyGate(initial_limit=1, max_limit=4)
    p1 = await gate.acquire()
    blocked = asyncio.create_task(gate.acquire())
    await asyncio.sleep(0.05)
    assert not blocked.done()
    gate.set_limit(2)
    p2 = await asyncio.wait_for(blocked, timeout=1.0)
    p1.release()
    p2.release()
    await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_gate_gc_stress():
    """Regression: release tasks must not be GC'd mid-flight."""
    gate = DynamicConcurrencyGate(initial_limit=50)
    for _ in range(200):
        p = await gate.acquire()
        p.release()
        gc.collect()
    await asyncio.sleep(0.1)
    assert gate.in_flight == 0


@pytest.mark.asyncio
async def test_rate_limiter_basic_acquire():
    limiter = AdaptiveRateLimiter(target_rps=1000.0)
    await asyncio.wait_for(limiter.acquire(), timeout=1.0)


@pytest.mark.asyncio
async def test_rate_limiter_pause_for():
    limiter = AdaptiveRateLimiter(target_rps=1000.0)
    limiter.pause_for(0.2)
    start = time.monotonic()
    await limiter.acquire()
    assert time.monotonic() - start >= 0.15


def test_fatal_monitor_streak():
    m = FatalFailureMonitor(fatal_streak_limit=3, global_rate_min_observations=1000)
    for _ in range(3):
        m.record(OutcomeClass.AUTH_ERROR, error="401")
    assert m.should_abort
    assert m.abort_reason != ""


def test_fatal_monitor_streak_resets_on_success():
    m = FatalFailureMonitor(fatal_streak_limit=3, global_rate_min_observations=1000)
    m.record(OutcomeClass.AUTH_ERROR, error="401")
    m.record(OutcomeClass.AUTH_ERROR, error="401")
    m.record(OutcomeClass.SUCCESS)
    m.record(OutcomeClass.AUTH_ERROR, error="401")
    assert not m.should_abort


def test_fatal_monitor_global_rate():
    m = FatalFailureMonitor(
        fatal_streak_limit=1000,
        global_failure_rate_limit=0.9,
        global_rate_min_observations=10,
    )
    for _ in range(10):
        m.record(OutcomeClass.TRANSIENT_ERROR, error="fail")
    assert m.should_abort


@pytest.mark.asyncio
async def test_runtime_healthy_to_degraded_to_open():
    rt = AdaptiveLLMRuntime(
        base_concurrency=20,
        base_rps=20.0,
        window_requests=10,
        window_seconds=100.0,
        min_observations_degraded=2,
        min_observations_open=4,
        min_failures_degraded=1,
        min_failures_open=2,
        open_overload_rate=0.15,
        degrade_overload_rate=0.05,
    )
    assert rt.state == "healthy"
    for _ in range(6):
        rt.observe(RequestOutcome(OutcomeClass.OVERLOAD, status_code=429))
    assert rt.state in ("degraded", "open")
    for _ in range(6):
        rt.observe(RequestOutcome(OutcomeClass.OVERLOAD, status_code=429))
    assert rt.state == "open"
    snap = rt.snapshot()
    assert snap["state"] == "open"
    assert snap["cooldown_remaining_seconds"] > 0
