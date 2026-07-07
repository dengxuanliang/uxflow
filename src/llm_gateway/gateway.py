"""LLMGateway facade — the single public entry point for LLM calls.

Orchestrates: truncation -> admission (gate+limiter) -> transport -> observe -> stats.
Rate limiting is always at this layer (both adaptive and non-adaptive modes).
"""

from __future__ import annotations

import asyncio

import httpx

from llm_gateway.config import GatewayConfig, resolve_httpx_connection_limits
from llm_gateway.outcomes import (
    OutcomeClass,
    RequestOutcome,
    classify_exception,
    classify_http_result,
)
from llm_gateway.recovery import SwappableAsyncClient, transport_recovery_loop
from llm_gateway.runtime import (
    AdaptiveLLMRuntime,
    AdaptiveRateLimiter,
    FatalFailureMonitor,
)
from llm_gateway.transport import RequestStats, async_llm_call
from llm_gateway.truncation import truncate_messages

__all__ = ["LLMGateway"]


def _outcome_from_usage(usage: dict) -> RequestOutcome:
    """Derive a RequestOutcome from the usage dict returned by async_llm_call."""
    status = usage.get("status_code")
    error_text = usage.get("error_response") or usage.get("error") or ""
    parse_error = bool(usage.get("parse_error"))
    content_filtered = bool(usage.get("content_filtered"))

    if status is not None:
        return classify_http_result(
            int(status),
            error_text=error_text,
            parse_error=parse_error,
            content_filtered=content_filtered,
        )
    # No status code — likely an exception during the request
    err = str(usage.get("error") or "")
    if "timeout" in err.lower():
        return RequestOutcome(OutcomeClass.TIMEOUT, error=err)
    return RequestOutcome(OutcomeClass.TRANSIENT_ERROR, error=err)


def _usage_from_exception(exc: Exception) -> tuple[None, dict, RequestOutcome]:
    """Build (content, usage_dict, outcome) for an exception that escaped transport.

    This ensures the circuit breaker/stats/fatal monitor always get feedback,
    even when async_llm_call raises rather than returning a usage dict.
    """
    outcome = classify_exception(exc)
    usage = {
        "error": outcome.error,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    return None, usage, outcome


class LLMGateway:
    """Async context-managed facade for making LLM calls with adaptive control.

    Usage::

        async with LLMGateway(config) as gw:
            text, usage = await gw.call(messages, model)
    """

    def __init__(self, config: GatewayConfig):
        self._config = config
        self._http_client: SwappableAsyncClient | None = None
        self._runtime: AdaptiveLLMRuntime | None = None
        self._fatal_monitor: FatalFailureMonitor | None = None
        self._stats = RequestStats()
        self._recovery_task: asyncio.Task | None = None
        # Non-adaptive mode uses a plain semaphore + rate limiter
        self._semaphore: asyncio.Semaphore | None = None
        self._limiter: AdaptiveRateLimiter | None = None

    async def __aenter__(self) -> "LLMGateway":
        max_conn, keepalive, _capped = resolve_httpx_connection_limits(
            requested_concurrency=self._config.concurrency,
        )
        limits = httpx.Limits(
            max_connections=max_conn,
            max_keepalive_connections=keepalive,
        )

        def client_factory():
            return httpx.AsyncClient(
                timeout=self._config.request_timeout,
                limits=limits,
            )

        self._http_client = SwappableAsyncClient(client_factory)
        await self._http_client.__aenter__()

        if self._config.enable_adaptive_runtime:
            self._runtime = AdaptiveLLMRuntime(
                base_concurrency=self._config.concurrency,
                base_rps=self._config.rps_limit,
                min_concurrency=self._config.adaptive_min_concurrency,
                min_rps=self._config.adaptive_min_rps,
                window_requests=self._config.adaptive_window_requests,
                window_seconds=self._config.adaptive_window_seconds,
                degrade_timeout_rate=self._config.adaptive_timeout_rate_degraded,
                open_timeout_rate=self._config.adaptive_timeout_rate_open,
                degrade_overload_rate=self._config.adaptive_overload_rate_degraded,
                open_overload_rate=self._config.adaptive_overload_rate_open,
                degrade_abnormal_rate=self._config.adaptive_abnormal_rate_degraded,
                open_abnormal_rate=self._config.adaptive_abnormal_rate_open,
                min_observations_degraded=self._config.adaptive_min_observations_degraded,
                min_observations_open=self._config.adaptive_min_observations_open,
                min_failures_degraded=self._config.adaptive_min_failures_degraded,
                min_failures_open=self._config.adaptive_min_failures_open,
                open_base_cooldown=self._config.adaptive_open_base_cooldown,
                open_max_cooldown=self._config.adaptive_open_max_cooldown,
                degrade_concurrency_factor=self._config.adaptive_degrade_concurrency_factor,
                degrade_rps_factor=self._config.adaptive_degrade_rps_factor,
                recovery_concurrency_step=self._config.adaptive_recovery_concurrency_step,
                recovery_rps_step=self._config.adaptive_recovery_rps_step,
                probe_success_required=self._config.adaptive_probe_success_required,
                probe_failure_tolerance=self._config.adaptive_probe_failure_tolerance,
            )
        else:
            self._semaphore = asyncio.Semaphore(self._config.concurrency)
            self._limiter = AdaptiveRateLimiter(target_rps=self._config.rps_limit)

        if self._config.fatal_abort_enabled:
            self._fatal_monitor = FatalFailureMonitor(
                fatal_streak_limit=self._config.fatal_abort_streak_limit,
                global_failure_rate_limit=self._config.fatal_abort_global_rate_limit,
                global_rate_min_observations=self._config.fatal_abort_global_rate_min_obs,
            )

        if self._config.transport_stuck_seconds > 0 and self._runtime is not None:
            self._recovery_task = asyncio.create_task(
                transport_recovery_loop(
                    runtime=self._runtime,
                    swappable=self._http_client,
                    threshold_seconds=self._config.transport_stuck_seconds,
                    cancel_in_flight=lambda: [],
                    pprint=lambda *a, **k: None,
                    label="LLMGateway",
                )
            )

        return self

    async def __aexit__(self, *exc) -> None:
        if self._recovery_task is not None:
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except (asyncio.CancelledError, Exception):
                pass
            self._recovery_task = None
        if self._http_client is not None:
            await self._http_client.__aexit__(None, None, None)
            self._http_client = None

    async def call(
        self,
        messages: list[dict],
        model: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> tuple[str | None, dict]:
        """Make a single LLM call through the full pipeline.

        Returns ``(content_str | None, usage_dict)``.
        """
        # Step 1: Truncation
        truncated = truncate_messages(
            messages,
            max_tokens=self._config.max_request_tokens,
            head_ratio=self._config.truncation_head_ratio,
            last_response_ratio=self._config.truncation_last_response_ratio,
            per_turn_ratio=self._config.truncation_per_turn_ratio,
        )

        # Step 2: Admission + Transport
        # If the transport raises an unexpected exception (e.g. an httpx
        # connection error that escapes async_llm_call's own handling), we
        # still classify it, feed the circuit breaker, and record stats
        # before returning — never let the breaker go blind on a failure.
        if self._runtime is not None:
            # Adaptive mode: runtime.acquire() handles gate + rate limiter
            permit = await self._runtime.acquire()
            try:
                content, usage = await async_llm_call(
                    self._http_client,
                    truncated,
                    model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    config=self._config,
                    adaptive_mode=True,
                )
                outcome = _outcome_from_usage(usage)
            except Exception as exc:
                content, usage, outcome = _usage_from_exception(exc)
            finally:
                permit.release()
            self._runtime.observe(outcome)
        else:
            # Non-adaptive mode: plain semaphore + rate limiter
            await self._limiter.acquire()
            async with self._semaphore:
                try:
                    content, usage = await async_llm_call(
                        self._http_client,
                        truncated,
                        model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        config=self._config,
                        adaptive_mode=False,
                    )
                    outcome = _outcome_from_usage(usage)
                except Exception as exc:
                    content, usage, outcome = _usage_from_exception(exc)

        # Step 3: Stats recording
        status = usage.get("status_code")
        if status is not None:
            self._stats.record(int(status))
        elif outcome.classification is OutcomeClass.TIMEOUT:
            self._stats.record_timeout()

        # Step 4: Fatal failure monitoring
        if self._fatal_monitor is not None:
            self._fatal_monitor.record(
                outcome.classification,
                error=str(usage.get("error") or ""),
            )

        return content, usage

    # ── Public properties ───────────────────────────────────

    @property
    def runtime_snapshot(self) -> dict:
        """Current adaptive runtime state snapshot."""
        if self._runtime is not None:
            return self._runtime.snapshot()
        return {"state": "non-adaptive", "concurrency": self._config.concurrency}

    @property
    def http_stats(self) -> dict:
        """Accumulated HTTP request statistics."""
        return self._stats.to_dict()

    @property
    def should_abort(self) -> bool:
        """Whether the fatal failure monitor recommends aborting."""
        if self._fatal_monitor is not None:
            return self._fatal_monitor.should_abort
        return False

    @property
    def abort_reason(self) -> str:
        """Human-readable reason for abort recommendation."""
        if self._fatal_monitor is not None:
            return self._fatal_monitor.abort_reason
        return ""
