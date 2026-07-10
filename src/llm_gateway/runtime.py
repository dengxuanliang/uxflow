"""Adaptive LLM runtime — circuit breaker, dynamic concurrency gate, rate limiter.

Ported from sft-label's llm/runtime.py. The state machine cycles through:
healthy → degraded → open → probing → (back to degraded or open).

Key invariants:
- observe() is synchronous (no await inside).
- _window is collections.deque for O(1) append/popleft.
- DynamicConcurrencyGate._background_tasks keeps strong refs to fire-and-forget
  tasks so GC cannot collect them mid-flight.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field

from llm_gateway.outcomes import OutcomeClass, RequestOutcome

__all__ = [
    "DynamicConcurrencyGate",
    "AdaptiveRateLimiter",
    "AdaptiveLLMRuntime",
    "FatalFailureMonitor",
    "RuntimePermit",
]


# ═══════════════════════════════════════════════════════════
# Gate Permit
# ═══════════════════════════════════════════════════════════


class _GatePermit:
    __slots__ = ("_gate", "_released")

    def __init__(self, gate: DynamicConcurrencyGate):
        self._gate = gate
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._gate.release()

    async def __aenter__(self) -> _GatePermit:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()


# ═══════════════════════════════════════════════════════════
# Dynamic Concurrency Gate
# ═══════════════════════════════════════════════════════════


class DynamicConcurrencyGate:
    def __init__(self, initial_limit: int, min_limit: int = 1, max_limit: int | None = None):
        if initial_limit <= 0:
            raise ValueError("initial_limit must be > 0")
        self._min_limit = max(min_limit, 1)
        self._max_limit = max_limit if max_limit is not None else max(initial_limit, self._min_limit)
        self._limit = max(self._min_limit, min(initial_limit, self._max_limit))
        self._in_flight = 0
        self._paused = False
        self._cond = asyncio.Condition()
        # Strong references to fire-and-forget release/notify tasks. Per
        # asyncio docs, ``loop.create_task`` only keeps a weak reference;
        # without retaining the returned Task it can be garbage-collected
        # mid-execution. Under heavy churn a GC'd _release task leaks
        # ``_in_flight``, eventually wedging the gate so all subsequent
        # acquire()s deadlock on ``_cond.wait()``.
        self._background_tasks: set[asyncio.Task] = set()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def paused(self) -> bool:
        return self._paused

    def set_limit(self, new_limit: int) -> None:
        clipped = max(self._min_limit, min(int(new_limit), self._max_limit))
        self._limit = clipped
        self._notify_waiters()

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
        self._notify_waiters()

    async def acquire(self) -> _GatePermit:
        async with self._cond:
            while self._paused or self._in_flight >= self._limit:
                await self._cond.wait()
            self._in_flight += 1
        return _GatePermit(self)

    def release(self) -> None:
        async def _release() -> None:
            async with self._cond:
                self._in_flight = max(0, self._in_flight - 1)
                self._cond.notify_all()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._in_flight = max(0, self._in_flight - 1)
            return
        task = loop.create_task(_release())
        # Strong-reference the task until it completes — see __init__ for
        # why. Without this, GC can collect the task mid-flight and
        # _in_flight is never decremented, eventually wedging the gate.
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _notify_waiters(self) -> None:
        async def _notify() -> None:
            async with self._cond:
                self._cond.notify_all()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(_notify())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)


# ═══════════════════════════════════════════════════════════
# Adaptive Rate Limiter
# ═══════════════════════════════════════════════════════════


class AdaptiveRateLimiter:
    def __init__(
        self,
        target_rps: float,
        *,
        burst: int | None = None,
        initial_tokens: float = 1.0,
    ):
        if target_rps <= 0:
            raise ValueError("target_rps must be > 0")
        self._target_rps = float(target_rps)
        self._burst_cap = burst if burst is not None else max(int(math.ceil(target_rps)), 1)
        self._burst = self._burst_cap
        self._tokens = max(0.0, min(float(initial_tokens), float(self._burst)))
        self._last_refill = time.monotonic()
        self._pause_until = 0.0
        self._lock = asyncio.Lock()

    @property
    def target_rps(self) -> float:
        return self._target_rps

    @property
    def pause_until(self) -> float:
        return self._pause_until

    def set_target_rps(self, target_rps: float) -> None:
        if target_rps <= 0:
            raise ValueError("target_rps must be > 0")
        self._target_rps = float(target_rps)
        self._burst = max(1, min(self._burst_cap, int(math.ceil(self._target_rps))))
        self._tokens = min(self._tokens, float(self._burst))

    def pause_for(self, seconds: float) -> None:
        if seconds <= 0:
            return
        self._pause_until = max(self._pause_until, time.monotonic() + seconds)

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                if now < self._pause_until:
                    wait_for = self._pause_until - now
                else:
                    elapsed = max(0.0, now - self._last_refill)
                    self._tokens = min(self._burst, self._tokens + elapsed * self._target_rps)
                    self._last_refill = now
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        return
                    wait_for = (1.0 - self._tokens) / self._target_rps
            await asyncio.sleep(max(wait_for, 0.0))


# ═══════════════════════════════════════════════════════════
# Runtime Permit
# ═══════════════════════════════════════════════════════════


@dataclass(slots=True)
class RuntimePermit:
    runtime: AdaptiveLLMRuntime
    gate_permit: _GatePermit
    stage: str
    sample_id: str
    queue_wait_ms: float
    state_at_acquire: str
    snapshot_at_acquire: dict
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.gate_permit.release()

    async def __aenter__(self) -> RuntimePermit:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()


# ═══════════════════════════════════════════════════════════
# Adaptive LLM Runtime — circuit breaker state machine
# ═══════════════════════════════════════════════════════════


class AdaptiveLLMRuntime:
    def __init__(
        self,
        *,
        base_concurrency: int,
        base_rps: float,
        min_concurrency: int = 4,
        min_rps: float = 2.0,
        window_requests: int = 50,
        window_seconds: float = 20.0,
        degrade_timeout_rate: float = 0.05,
        open_timeout_rate: float = 0.20,
        degrade_overload_rate: float = 0.05,
        open_overload_rate: float = 0.15,
        degrade_abnormal_rate: float = 0.04,
        open_abnormal_rate: float = 0.60,
        min_observations_degraded: int = 3,
        min_observations_open: int = 12,
        min_failures_degraded: int = 2,
        min_failures_open: int = 4,
        open_base_cooldown: float = 15.0,
        open_max_cooldown: float = 30.0,
        degrade_concurrency_factor: float = 0.5,
        degrade_rps_factor: float = 0.6,
        recovery_concurrency_step: int = 2,
        recovery_rps_step: float = 1.0,
        probe_success_required: int = 3,
        probe_failure_tolerance: int = 3,
    ):
        if base_concurrency <= 0:
            raise ValueError("base_concurrency must be > 0")
        if base_rps <= 0:
            raise ValueError("base_rps must be > 0")

        self.base_concurrency = int(base_concurrency)
        self.base_rps = float(base_rps)
        # Floor must remain clamped at or below base so the gate/rate-limiter
        # invariants (min_limit <= max_limit) stay consistent even when callers
        # configure a small base (e.g. smoke tests with concurrency=2).
        self.min_concurrency = max(1, min(int(min_concurrency), self.base_concurrency))
        self.min_rps = max(0.01, min(float(min_rps), self.base_rps))

        self.window_requests = max(int(window_requests), 1)
        self.window_seconds = max(float(window_seconds), 1.0)
        self.degrade_timeout_rate = max(float(degrade_timeout_rate), 0.0)
        self.open_timeout_rate = max(float(open_timeout_rate), 0.0)
        self.degrade_overload_rate = max(float(degrade_overload_rate), 0.0)
        self.open_overload_rate = max(float(open_overload_rate), 0.0)
        self.degrade_abnormal_rate = max(float(degrade_abnormal_rate), 0.0)
        self.open_abnormal_rate = max(float(open_abnormal_rate), 0.0)
        self.min_observations_degraded = max(int(min_observations_degraded), 1)
        self.min_observations_open = max(int(min_observations_open), self.min_observations_degraded)
        self.min_failures_degraded = max(int(min_failures_degraded), 1)
        self.min_failures_open = max(int(min_failures_open), self.min_failures_degraded)
        self.open_base_cooldown = max(float(open_base_cooldown), 0.0)
        self.open_max_cooldown = max(float(open_max_cooldown), self.open_base_cooldown)
        self.degrade_concurrency_factor = max(float(degrade_concurrency_factor), 0.01)
        self.degrade_rps_factor = max(float(degrade_rps_factor), 0.01)
        self.recovery_concurrency_step = max(int(recovery_concurrency_step), 1)
        self.recovery_rps_step = max(float(recovery_rps_step), 0.01)
        self.probe_success_required = max(int(probe_success_required), 1)
        self.probe_failure_tolerance = max(int(probe_failure_tolerance), 1)

        self.state = "healthy"
        self._cooldown_until = 0.0
        self._open_streak = 0
        self._probe_successes = 0
        self._probe_failures = 0
        self._healthy_streak = 0
        self._effective_concurrency = self.base_concurrency
        self._effective_rps = self.base_rps

        self.gate = DynamicConcurrencyGate(
            initial_limit=self.base_concurrency,
            min_limit=self.min_concurrency,
            max_limit=self.base_concurrency,
        )
        self.rate_limiter = AdaptiveRateLimiter(
            target_rps=self.base_rps,
            burst=max(int(math.ceil(self.base_rps)), 1),
            initial_tokens=1.0,
        )
        self._events: deque[dict] = deque(maxlen=500)
        self._window: deque[tuple[float, RequestOutcome]] = deque()

    async def acquire(self, *, stage: str = "", sample_id: str = "") -> RuntimePermit:
        start = time.perf_counter()
        while True:
            if self.state == "open":
                now = time.monotonic()
                if now >= self._cooldown_until:
                    self._enter_probing(reason="cooldown_elapsed")

            gate_permit = await self.gate.acquire()
            try:
                await self.rate_limiter.acquire()
            except Exception:
                gate_permit.release()
                raise

            if self.state == "open" and time.monotonic() >= self._cooldown_until:
                gate_permit.release()
                self._enter_probing(reason="cooldown_elapsed")
                continue

            wait_ms = (time.perf_counter() - start) * 1000.0
            snap = self.snapshot()
            return RuntimePermit(
                runtime=self,
                gate_permit=gate_permit,
                stage=stage,
                sample_id=sample_id,
                queue_wait_ms=wait_ms,
                state_at_acquire=self.state,
                snapshot_at_acquire=snap,
            )

    def observe(self, outcome: RequestOutcome) -> None:
        now = time.monotonic()
        self._window.append((now, outcome))
        self._prune_window(now)
        rates = self._window_rates()

        if self.state == "probing":
            if outcome.is_success:
                self._probe_successes += 1
                self._probe_failures = 0
                if self._probe_successes >= self.probe_success_required:
                    self._enter_degraded(reason="probe_success")
            elif outcome.is_infra_failure:
                self._probe_failures += 1
                if self._probe_failures >= self.probe_failure_tolerance:
                    self._enter_open(reason="probe_failure")
            else:
                self._enter_degraded(reason="probe_terminal_outcome")
            return

        if self.state in {"healthy", "degraded"}:
            should_open = (
                rates["total"] >= self.min_observations_open
                and (
                    (
                        rates["timeout_count"] >= self.min_failures_open
                        and rates["timeout_rate"] >= self.open_timeout_rate
                    )
                    or (
                        rates["overload_count"] >= self.min_failures_open
                        and rates["overload_rate"] >= self.open_overload_rate
                    )
                    or (
                        rates["abnormal_count"] >= self.min_failures_open
                        and rates["abnormal_rate"] >= self.open_abnormal_rate
                    )
                )
            )
            if should_open:
                self._enter_open(reason="open_threshold")
                return

        if self.state == "healthy":
            should_degrade = (
                rates["total"] >= self.min_observations_degraded
                and (
                    (
                        rates["timeout_count"] >= self.min_failures_degraded
                        and rates["timeout_rate"] >= self.degrade_timeout_rate
                    )
                    or (
                        rates["overload_count"] >= self.min_failures_degraded
                        and rates["overload_rate"] >= self.degrade_overload_rate
                    )
                    or (
                        rates["abnormal_count"] >= self.min_failures_degraded
                        and rates["abnormal_rate"] >= self.degrade_abnormal_rate
                    )
                )
            )
            if should_degrade:
                self._enter_degraded(reason="degrade_threshold")
            return

        if self.state == "degraded":
            is_healthy_window = (
                rates["timeout_rate"] <= self.degrade_timeout_rate * 0.5
                and rates["overload_rate"] <= self.degrade_overload_rate * 0.5
                and rates["abnormal_rate"] <= self.degrade_abnormal_rate * 0.5
            )
            if is_healthy_window:
                self._healthy_streak += 1
                self._increase_limits()
                if (
                    self._healthy_streak >= 3
                    and self._effective_concurrency >= self.base_concurrency
                    and self._effective_rps >= self.base_rps
                ):
                    self._enter_healthy(reason="recovered")
            else:
                self._healthy_streak = 0

    def snapshot(self) -> dict:
        rates = self._window_rates()
        cooldown_remaining = max(self._cooldown_until - time.monotonic(), 0.0)
        return {
            "state": self.state,
            "base_concurrency": self.base_concurrency,
            "base_rps": self.base_rps,
            "effective_concurrency": self._effective_concurrency,
            "effective_rps": self._effective_rps,
            "in_flight": self.gate.in_flight,
            "gate_paused": self.gate.paused,
            "cooldown_until": self._cooldown_until,
            "cooldown_remaining_seconds": cooldown_remaining,
            "rates": rates,
            "window_size": len(self._window),
            "events_recorded": len(self._events),
        }

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    def _prune_window(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()
        while len(self._window) > self.window_requests:
            self._window.popleft()

    def _window_rates(self) -> dict:
        total = len(self._window)
        if total == 0:
            return {
                "total": 0,
                "timeout_rate": 0.0,
                "timeout_count": 0,
                "overload_rate": 0.0,
                "overload_count": 0,
                "abnormal_rate": 0.0,
                "abnormal_count": 0,
                "infra_failure_rate": 0.0,
            }

        timeouts = 0
        overloads = 0
        abnormals = 0
        infra_failures = 0
        for _, outcome in self._window:
            if outcome.classification is OutcomeClass.TIMEOUT:
                timeouts += 1
            if outcome.classification in {OutcomeClass.OVERLOAD, OutcomeClass.SERVER_ERROR}:
                overloads += 1
            if outcome.classification is OutcomeClass.ABNORMAL_RESPONSE:
                abnormals += 1
            if outcome.is_infra_failure:
                infra_failures += 1

        return {
            "total": total,
            "timeout_rate": timeouts / total,
            "timeout_count": timeouts,
            "overload_rate": overloads / total,
            "overload_count": overloads,
            "abnormal_rate": abnormals / total,
            "abnormal_count": abnormals,
            "infra_failure_rate": infra_failures / total,
        }

    def _enter_healthy(self, *, reason: str) -> None:
        self.state = "healthy"
        self._probe_successes = 0
        self._probe_failures = 0
        self._healthy_streak = 0
        self._open_streak = 0
        self.gate.resume()
        self._set_effective_limits(self.base_concurrency, self.base_rps)
        self._record_event("state", reason, {"state": self.state})

    def _enter_degraded(self, *, reason: str) -> None:
        self.state = "degraded"
        self._probe_successes = 0
        self._probe_failures = 0
        self._healthy_streak = 0
        self.gate.resume()
        degraded_concurrency = max(
            self.min_concurrency,
            int(math.floor(self.base_concurrency * self.degrade_concurrency_factor)),
        )
        degraded_rps = max(self.min_rps, self.base_rps * self.degrade_rps_factor)
        self._set_effective_limits(degraded_concurrency, degraded_rps)
        self._record_event("state", reason, {"state": self.state})

    def _enter_open(self, *, reason: str) -> None:
        self.state = "open"
        self._probe_successes = 0
        self._probe_failures = 0
        self._healthy_streak = 0
        self._open_streak += 1
        cooldown = min(self.open_base_cooldown * (2 ** (self._open_streak - 1)), self.open_max_cooldown)
        self._cooldown_until = time.monotonic() + cooldown
        self.gate.resume()
        self._set_effective_limits(self.min_concurrency, self.min_rps)
        self._record_event("state", reason, {"state": self.state, "cooldown_seconds": cooldown})

    def _enter_probing(self, *, reason: str) -> None:
        self.state = "probing"
        self._probe_successes = 0
        self._probe_failures = 0
        self._healthy_streak = 0
        self._window.clear()
        self.gate.resume()
        self._set_effective_limits(self.min_concurrency, self.min_rps)
        self._record_event("state", reason, {"state": self.state})

    def _increase_limits(self) -> None:
        next_concurrency = min(
            self.base_concurrency,
            self._effective_concurrency + self.recovery_concurrency_step,
        )
        next_rps = min(self.base_rps, self._effective_rps + self.recovery_rps_step)
        self._set_effective_limits(next_concurrency, next_rps)

    def _set_effective_limits(self, concurrency: int, rps: float) -> None:
        self._effective_concurrency = int(max(self.min_concurrency, min(concurrency, self.base_concurrency)))
        self._effective_rps = float(max(self.min_rps, min(rps, self.base_rps)))
        self.gate.set_limit(self._effective_concurrency)
        self.rate_limiter.set_target_rps(self._effective_rps)

    def _record_event(self, kind: str, reason: str, payload: dict) -> None:
        self._events.append(
            {
                "ts": time.time(),
                "kind": kind,
                "reason": reason,
                "state": self.state,
                "effective_concurrency": self._effective_concurrency,
                "effective_rps": self._effective_rps,
                "payload": payload,
            }
        )


# ═══════════════════════════════════════════════════════════
# Fatal Failure Monitor — abort pipeline on sustained failures
# ═══════════════════════════════════════════════════════════


class FatalFailureMonitor:
    """Two-signal abort monitor: fatal error streak + global failure rate.

    Signal 1 — **Fatal streak**: N consecutive results whose OutcomeClass is in
    ``fatal_classes`` triggers immediate abort.  Any success *or* non-fatal
    failure resets the streak counter, so only truly consecutive fatal errors
    fire this signal (catches "billing exhausted" in seconds).

    Signal 2 — **Global failure rate**: once at least ``global_rate_min_observations``
    results have been recorded, if ``failed / total >= global_failure_rate_limit``
    the monitor fires (catches slow, persistent degradation).

    Either signal -> ``should_abort`` becomes True.
    """

    def __init__(
        self,
        *,
        fatal_streak_limit: int = 5,
        global_failure_rate_limit: float = 0.95,
        global_rate_min_observations: int = 20,
        fatal_classes: frozenset[OutcomeClass] = frozenset({OutcomeClass.AUTH_ERROR}),
        enabled: bool = True,
    ):
        self.fatal_streak_limit = fatal_streak_limit
        self.global_failure_rate_limit = global_failure_rate_limit
        self.global_rate_min_observations = global_rate_min_observations
        self.fatal_classes = fatal_classes
        self.enabled = enabled

        # Counters
        self._fatal_streak = 0
        self._total = 0
        self._total_failed = 0
        self._total_success = 0
        self._last_fatal_error = ""
        self._last_fatal_sample_id = ""

        # Abort state
        self._aborted = False
        self._abort_reason = ""

    # ── Recording ────────────────────────────────────────

    def record(
        self,
        outcome_class: OutcomeClass,
        *,
        sample_id: str = "",
        error: str = "",
        infra_only: bool = False,
        monitor: dict | None = None,
    ) -> None:
        """Record a sample outcome. Call after each sample completes.

        ``infra_only=True`` flags a sample whose failure was an infrastructure
        problem (TIMEOUT / OVERLOAD / SERVER_ERROR / ABNORMAL_RESPONSE /
        TRANSIENT_ERROR). Such samples are invisible to BOTH the global-rate
        signal and the fatal streak.
        """
        if not self.enabled:
            return

        if infra_only:
            return

        self._total += 1

        if outcome_class is OutcomeClass.SUCCESS:
            self._total_success += 1
            self._fatal_streak = 0
            return

        # Any non-success is a failure for global rate purposes
        self._total_failed += 1

        if outcome_class in self.fatal_classes:
            self._fatal_streak += 1
            self._last_fatal_error = error
            self._last_fatal_sample_id = sample_id
        else:
            # Non-fatal failure resets the streak
            self._fatal_streak = 0

        self._check_triggers()

    def _check_triggers(self) -> None:
        if self._aborted:
            return

        # Signal 1: fatal streak
        if self._fatal_streak >= self.fatal_streak_limit:
            self._aborted = True
            self._abort_reason = (
                f"连续 {self._fatal_streak} 个致命错误 "
                f"({', '.join(c.value for c in self.fatal_classes)}). "
                f"最后错误: {self._last_fatal_error[:200]}"
            )
            return

        # Signal 2: global failure rate
        if (
            self._total >= self.global_rate_min_observations
            and self._total_failed / self._total >= self.global_failure_rate_limit
        ):
            rate = self._total_failed / self._total
            self._aborted = True
            self._abort_reason = (
                f"全局失败率 {rate:.1%} ({self._total_failed}/{self._total}) "
                f"超过阈值 {self.global_failure_rate_limit:.0%}"
            )

    # ── Query ────────────────────────────────────────────

    @property
    def should_abort(self) -> bool:
        return self._aborted

    @property
    def abort_reason(self) -> str:
        return self._abort_reason

    def to_dict(self) -> dict:
        """Serialisable snapshot for checkpoint / diagnostics."""
        return {
            "enabled": self.enabled,
            "aborted": self._aborted,
            "abort_reason": self._abort_reason,
            "fatal_streak": self._fatal_streak,
            "fatal_streak_limit": self.fatal_streak_limit,
            "total": self._total,
            "total_failed": self._total_failed,
            "total_success": self._total_success,
            "global_failure_rate": (
                self._total_failed / self._total if self._total > 0 else 0.0
            ),
            "global_failure_rate_limit": self.global_failure_rate_limit,
            "global_rate_min_observations": self.global_rate_min_observations,
            "last_fatal_error": self._last_fatal_error[:300],
            "last_fatal_sample_id": self._last_fatal_sample_id,
        }
