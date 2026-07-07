"""Gateway configuration + httpx connection-limit derivation.

GatewayConfig field names and defaults align with sft-label's PipelineConfig
(the LLM-relevant subset). resolve_httpx_connection_limits is lifted from
sft-label's http_limits.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["GatewayConfig", "resolve_httpx_connection_limits"]


@dataclass
class GatewayConfig:
    # ── connection ──
    litellm_base: str = "http://localhost:4000/v1"
    litellm_key: str = ""

    # ── single-request control ──
    max_retries: int = 3
    request_timeout: int = 90
    request_timeout_escalation: list = field(default_factory=lambda: [60, 90, 120])

    # ── rate limiting (token bucket) ──
    rps_limit: float = 20.0
    rps_warmup: float = 30.0

    # ── concurrency (dynamic gate) ──
    concurrency: int = 200

    # ── adaptive circuit breaker ──
    enable_adaptive_runtime: bool = True
    adaptive_min_concurrency: int = 4
    adaptive_min_rps: float = 2.0
    adaptive_window_requests: int = 50
    adaptive_window_seconds: float = 20.0
    adaptive_timeout_rate_degraded: float = 0.05
    adaptive_overload_rate_degraded: float = 0.05
    adaptive_abnormal_rate_degraded: float = 0.04
    adaptive_min_observations_degraded: int = 3
    adaptive_min_failures_degraded: int = 2
    adaptive_timeout_rate_open: float = 0.20
    adaptive_overload_rate_open: float = 0.15
    adaptive_abnormal_rate_open: float = 0.60
    adaptive_min_observations_open: int = 12
    adaptive_min_failures_open: int = 4
    adaptive_open_base_cooldown: float = 15.0
    adaptive_open_max_cooldown: float = 30.0
    adaptive_degrade_concurrency_factor: float = 0.5
    adaptive_degrade_rps_factor: float = 0.6
    adaptive_recovery_concurrency_step: int = 2
    adaptive_recovery_rps_step: float = 1.0
    adaptive_probe_success_required: int = 3
    adaptive_probe_failure_tolerance: int = 3

    # ── truncation budget ──
    max_request_tokens: int = 10000
    truncation_head_ratio: float = 0.35
    truncation_last_response_ratio: float = 0.30
    truncation_per_turn_ratio: float = 0.35

    # ── transport recovery ──
    transport_stuck_seconds: int = 300

    # ── fatal failure monitor ──
    fatal_abort_enabled: bool = True
    fatal_abort_streak_limit: int = 5
    fatal_abort_global_rate_limit: float = 0.95
    fatal_abort_global_rate_min_obs: int = 200


def _soft_nofile_limit() -> int | None:
    try:
        import resource
    except Exception:
        return None
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception:
        return None
    if soft is None or soft <= 0:
        return None
    return int(soft)


def resolve_httpx_connection_limits(
    *,
    requested_concurrency: int,
    extra_connections: int = 10,
    reserve_fds: int = 64,
) -> tuple[int, int, bool]:
    """Return (max_connections, max_keepalive_connections, capped).

    max_connections must be >= concurrency, else the semaphore admits a
    request but httpx has no connection slot and it stalls internally.
    When the process FD soft limit is too low, cap and flag it.
    """
    concurrency = max(int(requested_concurrency or 0), 1)
    requested_max = max(concurrency + max(int(extra_connections or 0), 0), 1)
    requested_keepalive = min(concurrency, requested_max)

    soft = _soft_nofile_limit()
    if soft is None or soft >= 1_000_000:
        return requested_max, requested_keepalive, False

    available = max(soft - max(int(reserve_fds or 0), 0), 1)
    max_conn = min(requested_max, available)
    keepalive = min(requested_keepalive, max_conn)
    capped = max_conn < requested_max or keepalive < requested_keepalive
    return max_conn, keepalive, capped
