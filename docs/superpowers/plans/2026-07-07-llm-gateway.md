# LLM Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared async LLM call gateway that protects a self-hosted LiteLLM proxy from three interception modes: over-long requests, sliding-window rate limits, and excessive concurrency.

**Architecture:** A single `LLMGateway` facade orchestrates four orthogonal components per call — truncation (pre-send budget), circuit-breaker admission (adaptive concurrency gate + token-bucket rate limiter), HTTP transport (retry + status classification), and outcome observation (feeds the circuit-breaker state machine). Ported closely from the `sft-label` reference implementation, with JSON parsing removed (callers parse) and rate limiting unified at the gateway layer.

**Tech Stack:** Python 3.11+, httpx (async HTTP), pytest + pytest-asyncio (testing via `httpx.MockTransport`), hatchling (build).

**Design Spec:** [`../specs/2026-07-07-llm-gateway-design.md`](../specs/2026-07-07-llm-gateway-design.md)
**Reference implementation:** `sft-label` repo, `src/sft_label/llm/` and neighbors (cloned at `/tmp/sft-label`).

---

## File Structure

Package lives at `src/llm_gateway/`. Each file has one responsibility; dependency order is leaves-first.

| File | Responsibility | Depends on |
|------|---------------|------------|
| `src/llm_gateway/outcomes.py` | Classify HTTP results / exceptions into `OutcomeClass` | (none) |
| `src/llm_gateway/config.py` | `GatewayConfig` dataclass + httpx connection-limit calc | (none) |
| `src/llm_gateway/truncation.py` | Pre-send message budget truncation (OpenAI format) | (none) |
| `src/llm_gateway/runtime.py` | Circuit breaker: gate + rate limiter + state machine + fatal monitor | outcomes |
| `src/llm_gateway/transport.py` | Single HTTP call + retry + markdown strip | outcomes, config |
| `src/llm_gateway/recovery.py` | Swappable httpx client + zombie-connection recovery loop | runtime (type refs) |
| `src/llm_gateway/gateway.py` | `LLMGateway` facade wiring all components | ALL |
| `src/llm_gateway/__init__.py` | Public API exports | gateway, config, outcomes |

Tests mirror one-to-one under `tests/`.

**Environment note:** The venv already exists at `.venv` (Python 3.11). All commands assume it is active: `source .venv/bin/activate`.

---

## Task 0: Test fixtures scaffold

**Files:**
- Create: `tests/conftest.py`
- Test: (none — fixtures only, exercised by later tasks)

- [ ] **Step 1: Write conftest.py with the shared config fixture**

```python
import pytest

from llm_gateway.config import GatewayConfig


@pytest.fixture
def default_config():
    """Test-friendly config: low timeouts, small windows, fast cooldown."""
    return GatewayConfig(
        litellm_base="http://localhost:9999/v1",
        litellm_key="test-key",
        request_timeout=5,
        request_timeout_escalation=[3, 5, 8],
        concurrency=10,
        rps_limit=100.0,
        adaptive_window_requests=5,
        adaptive_window_seconds=2.0,
        adaptive_open_base_cooldown=1.0,
        adaptive_open_max_cooldown=3.0,
        adaptive_min_observations_degraded=2,
        adaptive_min_observations_open=4,
        adaptive_min_failures_degraded=1,
        adaptive_min_failures_open=2,
        fatal_abort_global_rate_min_obs=10,
    )
```

Note: this fixture imports `GatewayConfig`, which does not exist until Task 2. That is fine — Task 0 is committed together with Task 2, or the import error is expected until then. To keep tasks independently runnable, **commit Task 0 together with Task 2** (config is the first thing the fixture needs).

- [ ] **Step 2: Commit later with Task 2** (no standalone commit — see note above)

---

## Task 1: outcomes.py — result classification

**Files:**
- Create: `src/llm_gateway/outcomes.py`
- Test: `tests/test_outcomes.py`

Reference: sft-label `src/sft_label/llm/runtime.py` lines 23-202. Differences: drop `validation_error` param, keep `parse_error`, drop `monitor_to_outcome_class` and `PipelineAbortError`.

- [ ] **Step 1: Write the failing test**

```python
import asyncio

import pytest

from llm_gateway.outcomes import (
    OutcomeClass,
    RequestOutcome,
    classify_http_result,
    classify_exception,
)


def test_2xx_success():
    o = classify_http_result(200)
    assert o.classification is OutcomeClass.SUCCESS
    assert o.is_success
    assert not o.is_retryable


def test_2xx_parse_error_is_abnormal():
    o = classify_http_result(200, parse_error=True)
    assert o.classification is OutcomeClass.ABNORMAL_RESPONSE
    assert o.is_abnormal
    assert o.is_retryable


def test_429_is_overload():
    o = classify_http_result(429)
    assert o.classification is OutcomeClass.OVERLOAD
    assert o.is_overload
    assert o.is_infra_failure


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_5xx_is_server_error(code):
    o = classify_http_result(code)
    assert o.classification is OutcomeClass.SERVER_ERROR
    assert o.is_overload


@pytest.mark.parametrize("code", [401, 402])
def test_auth_errors(code):
    o = classify_http_result(code)
    assert o.classification is OutcomeClass.AUTH_ERROR
    assert not o.is_retryable


def test_400_content_filter_keyword():
    o = classify_http_result(400, error_text="Azure content_filter triggered")
    assert o.classification is OutcomeClass.CONTENT_FILTERED


def test_400_content_filter_explicit_flag():
    o = classify_http_result(400, content_filtered=True)
    assert o.classification is OutcomeClass.CONTENT_FILTERED


def test_400_context_length_is_input_error():
    o = classify_http_result(400, error_text="context_length_exceeded: too long")
    assert o.classification is OutcomeClass.INPUT_ERROR


def test_400_other_is_transient():
    o = classify_http_result(400, error_text="Unknown model foo")
    assert o.classification is OutcomeClass.TRANSIENT_ERROR
    assert o.is_retryable


def test_403_is_transient():
    o = classify_http_result(403)
    assert o.classification is OutcomeClass.TRANSIENT_ERROR


def test_other_4xx_is_input_error():
    o = classify_http_result(404)
    assert o.classification is OutcomeClass.INPUT_ERROR


def test_classify_timeout_exception():
    o = classify_exception(asyncio.TimeoutError())
    assert o.classification is OutcomeClass.TIMEOUT


def test_classify_generic_exception():
    o = classify_exception(ValueError("boom"))
    assert o.classification is OutcomeClass.TRANSIENT_ERROR
    assert "ValueError" in o.error


def test_request_outcome_default_extra_is_isolated():
    a = RequestOutcome(classification=OutcomeClass.SUCCESS)
    b = RequestOutcome(classification=OutcomeClass.SUCCESS)
    a.extra["k"] = 1
    assert b.extra == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_outcomes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_gateway.outcomes'`

- [ ] **Step 3: Write the implementation**

Create `src/llm_gateway/outcomes.py`:

```python
"""Request outcome classification.

Maps HTTP status codes and exceptions to a small set of semantic outcome
classes that the circuit breaker (runtime.py) uses to decide when to
degrade / open / recover.

Ported from sft-label's llm/runtime.py, with pipeline glue removed. The
gateway does not parse business JSON, so validation_error is dropped;
parse_error is kept to signal a structurally-broken 200 response
(missing choices[0].message.content).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "OutcomeClass",
    "RequestOutcome",
    "classify_http_result",
    "classify_exception",
]


class OutcomeClass(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    OVERLOAD = "overload"                    # 429
    SERVER_ERROR = "server_error"            # 500/502/503/504
    ABNORMAL_RESPONSE = "abnormal_response"  # 200 but no choices[0].message.content
    CONTENT_FILTERED = "content_filtered"    # 400 + content filter keyword
    AUTH_ERROR = "auth_error"                # 401/402
    INPUT_ERROR = "input_error"              # 400 + context_length_exceeded
    TRANSIENT_ERROR = "transient_error"      # other retryable errors


@dataclass(slots=True)
class RequestOutcome:
    classification: OutcomeClass
    status_code: int | None = None
    error: str = ""
    latency_ms: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.classification is OutcomeClass.SUCCESS

    @property
    def is_retryable(self) -> bool:
        return self.classification in {
            OutcomeClass.TIMEOUT,
            OutcomeClass.OVERLOAD,
            OutcomeClass.SERVER_ERROR,
            OutcomeClass.ABNORMAL_RESPONSE,
            OutcomeClass.TRANSIENT_ERROR,
        }

    @property
    def is_infra_failure(self) -> bool:
        return self.classification in {
            OutcomeClass.TIMEOUT,
            OutcomeClass.OVERLOAD,
            OutcomeClass.SERVER_ERROR,
            OutcomeClass.ABNORMAL_RESPONSE,
            OutcomeClass.TRANSIENT_ERROR,
        }

    @property
    def is_overload(self) -> bool:
        return self.classification in {OutcomeClass.OVERLOAD, OutcomeClass.SERVER_ERROR}

    @property
    def is_abnormal(self) -> bool:
        return self.classification is OutcomeClass.ABNORMAL_RESPONSE


_CONTENT_FILTER_KEYWORDS = (
    "content_filter",
    "content policy",
    "content_policy",
    "responsibleaipolicyviolation",
    "content_management_policy",
    "moderation",
)

_INPUT_ERROR_KEYWORDS = (
    "context_length_exceeded",
    "maximum context length",
    "invalid_request_error",
)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(k in low for k in keywords)


def classify_http_result(
    status_code: int,
    *,
    error_text: str = "",
    parse_error: bool = False,
    content_filtered: bool = False,
    latency_ms: float | None = None,
) -> RequestOutcome:
    if 200 <= status_code < 300:
        if parse_error:
            return RequestOutcome(OutcomeClass.ABNORMAL_RESPONSE, status_code, error_text, latency_ms)
        return RequestOutcome(OutcomeClass.SUCCESS, status_code, latency_ms=latency_ms)

    if status_code in (401, 402):
        return RequestOutcome(OutcomeClass.AUTH_ERROR, status_code, error_text, latency_ms)

    if status_code == 429:
        return RequestOutcome(OutcomeClass.OVERLOAD, status_code, error_text, latency_ms)

    if status_code in (500, 502, 503, 504):
        return RequestOutcome(OutcomeClass.SERVER_ERROR, status_code, error_text, latency_ms)

    if status_code == 400:
        if content_filtered or _contains_any(error_text, _CONTENT_FILTER_KEYWORDS):
            return RequestOutcome(OutcomeClass.CONTENT_FILTERED, status_code, error_text, latency_ms)
        if _contains_any(error_text, _INPUT_ERROR_KEYWORDS):
            return RequestOutcome(OutcomeClass.INPUT_ERROR, status_code, error_text, latency_ms)
        return RequestOutcome(OutcomeClass.TRANSIENT_ERROR, status_code, error_text, latency_ms)

    if status_code == 403:
        return RequestOutcome(OutcomeClass.TRANSIENT_ERROR, status_code, error_text, latency_ms)

    if 400 <= status_code < 500:
        return RequestOutcome(OutcomeClass.INPUT_ERROR, status_code, error_text, latency_ms)

    return RequestOutcome(OutcomeClass.TRANSIENT_ERROR, status_code, error_text, latency_ms)


def classify_exception(exc: Exception, *, latency_ms: float | None = None) -> RequestOutcome:
    message = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, asyncio.TimeoutError):
        return RequestOutcome(OutcomeClass.TIMEOUT, error=message, latency_ms=latency_ms)
    return RequestOutcome(OutcomeClass.TRANSIENT_ERROR, error=message, latency_ms=latency_ms)
```

Note: `RequestOutcome`'s positional args are `(classification, status_code, error, latency_ms, extra)`. The calls above pass `latency_ms` positionally only where `error` is also passed; otherwise use the `latency_ms=` keyword. Verify against the dataclass field order.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_outcomes.py -v`
Expected: PASS (all cases)

- [ ] **Step 5: Commit**

```bash
git add src/llm_gateway/outcomes.py tests/test_outcomes.py
git commit -m "feat(outcomes): OutcomeClass enum + classify functions"
```

---

## Task 2: config.py — GatewayConfig + connection limits

**Files:**
- Create: `src/llm_gateway/config.py`
- Test: `tests/test_config.py`
- Also commit: `tests/conftest.py` (from Task 0)

Reference: spec §6 + sft-label `src/sft_label/http_limits.py`. Merge http_limits into config.py (single use site). Add two probe fields missing from spec §6.

- [ ] **Step 1: Write the failing test**

```python
import llm_gateway.config as cfg
from llm_gateway.config import GatewayConfig, resolve_httpx_connection_limits


def test_defaults():
    c = GatewayConfig()
    assert c.litellm_base == "http://localhost:4000/v1"
    assert c.concurrency == 200
    assert c.rps_limit == 20.0
    assert c.max_request_tokens == 10000
    assert c.enable_adaptive_runtime is True
    assert c.adaptive_open_base_cooldown == 15.0
    assert c.adaptive_probe_success_required == 3
    assert c.adaptive_probe_failure_tolerance == 3
    assert c.fatal_abort_global_rate_min_obs == 200


def test_timeout_escalation_default_factory_is_isolated():
    a = GatewayConfig()
    b = GatewayConfig()
    assert a.request_timeout_escalation == [60, 90, 120]
    a.request_timeout_escalation.append(999)
    assert b.request_timeout_escalation == [60, 90, 120]


def test_override_fields():
    c = GatewayConfig(concurrency=500, rps_limit=30.0)
    assert c.concurrency == 500
    assert c.rps_limit == 30.0


def test_resolve_limits_normal(monkeypatch):
    # soft FD limit huge -> no capping
    monkeypatch.setattr(cfg, "_soft_nofile_limit", lambda: 1_000_000)
    max_conn, keepalive, capped = resolve_httpx_connection_limits(
        requested_concurrency=200, extra_connections=10
    )
    assert max_conn == 210
    assert keepalive == 200
    assert capped is False


def test_resolve_limits_capped_by_fds(monkeypatch):
    # tiny FD budget forces a cap
    monkeypatch.setattr(cfg, "_soft_nofile_limit", lambda: 128)
    max_conn, keepalive, capped = resolve_httpx_connection_limits(
        requested_concurrency=500, extra_connections=10, reserve_fds=64
    )
    assert max_conn == 64  # 128 - 64 reserve
    assert keepalive <= max_conn
    assert capped is True


def test_resolve_limits_none_soft_limit(monkeypatch):
    monkeypatch.setattr(cfg, "_soft_nofile_limit", lambda: None)
    max_conn, keepalive, capped = resolve_httpx_connection_limits(
        requested_concurrency=50
    )
    assert capped is False
    assert max_conn >= 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_gateway.config'`

- [ ] **Step 3: Write the implementation**

Create `src/llm_gateway/config.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py tests/test_outcomes.py -v`
Expected: PASS (config + outcomes together; conftest now imports cleanly)

- [ ] **Step 5: Commit**

```bash
git add src/llm_gateway/config.py tests/test_config.py tests/conftest.py
git commit -m "feat(config): GatewayConfig dataclass + connection limit calculation"
```

---

## Task 3: truncation.py — message budget truncation

**Files:**
- Create: `src/llm_gateway/truncation.py`
- Test: `tests/test_truncation.py`

Reference: sft-label `src/sft_label/preprocessing.py` truncation logic, **rewritten** for OpenAI message format (`role/content` instead of `from/value`). System messages are preserved intact; budget distributed by ratio among the remaining turns.

- [ ] **Step 1: Write the failing test**

```python
from llm_gateway.truncation import (
    TRUNCATION_MARKER,
    estimate_tokens,
    truncate_messages,
)


def test_estimate_tokens():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 100) == 25


def test_within_budget_unchanged():
    msgs = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    result = truncate_messages(msgs, max_tokens=10000)
    assert result == msgs
    # must be a copy, not mutate original
    assert result is not msgs


def test_system_message_never_truncated():
    system_content = "x" * 5000  # would blow budget if counted
    msgs = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "short"},
    ]
    # budget is tiny, but system must survive intact
    result = truncate_messages(msgs, max_tokens=100)
    assert result[0]["content"] == system_content


def test_long_user_message_truncated():
    long_content = "word " * 20000  # ~20k tokens
    msgs = [
        {"role": "user", "content": long_content},
    ]
    result = truncate_messages(msgs, max_tokens=500)
    assert len(result[0]["content"]) < len(long_content)
    assert TRUNCATION_MARKER in result[0]["content"]


def test_multi_turn_respects_ratios():
    msgs = [
        {"role": "user", "content": "a" * 4000},      # first user
        {"role": "assistant", "content": "b" * 4000},  # middle
        {"role": "user", "content": "c" * 4000},       # middle
        {"role": "assistant", "content": "d" * 4000},  # last assistant
    ]
    # budget = 1000 tokens = 4000 chars
    result = truncate_messages(msgs, max_tokens=1000)
    total_chars = sum(len(m["content"]) for m in result)
    assert total_chars <= 4000 + 200  # small tolerance for markers


def test_empty_messages():
    result = truncate_messages([], max_tokens=1000)
    assert result == []


def test_only_system_messages():
    msgs = [{"role": "system", "content": "sys"}]
    result = truncate_messages(msgs, max_tokens=10)
    assert result == msgs


def test_preserves_head_and_tail():
    long_content = "HEAD " + ("middle " * 5000) + " TAIL"
    msgs = [{"role": "user", "content": long_content}]
    result = truncate_messages(msgs, max_tokens=100)
    content = result[0]["content"]
    assert content.startswith("HEAD")
    assert content.endswith("TAIL")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_truncation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_gateway.truncation'`

- [ ] **Step 3: Write the implementation**

Create `src/llm_gateway/truncation.py`:

```python
"""Pre-send message budget truncation (OpenAI format).

Ensures the total input token estimate stays under max_request_tokens
before hitting the network. System messages are never truncated.
Budget is distributed by ratio: first-user (head), last-assistant (tail),
middle turns share the remainder.

Token estimation: len(text) // 4 (fast, no external dependency).
"""

from __future__ import annotations

import copy

__all__ = [
    "TRUNCATION_MARKER",
    "estimate_tokens",
    "truncate_messages",
]

TRUNCATION_MARKER = "\n\n[... content truncated ...]\n\n"


def estimate_tokens(text: str) -> int:
    """Approximate token count. 1 token ≈ 4 chars for English/code."""
    return max(len(text) // 4, 0)


def _truncate_text(text: str, max_chars: int, keep_head_ratio: float = 0.3) -> str:
    """Truncate a single text blob, keeping head + tail with marker in between."""
    if len(text) <= max_chars:
        return text
    marker_len = len(TRUNCATION_MARKER)
    usable = max(max_chars - marker_len, 0)
    head_len = max(int(usable * keep_head_ratio), 1)
    tail_len = max(usable - head_len, 1)
    return text[:head_len] + TRUNCATION_MARKER + text[-tail_len:]


def truncate_messages(
    messages: list[dict],
    *,
    max_tokens: int = 10000,
    head_ratio: float = 0.35,
    last_response_ratio: float = 0.30,
    per_turn_ratio: float = 0.35,
) -> list[dict]:
    """Truncate messages to fit within token budget.

    System messages (role=='system') are preserved intact and don't count
    toward the budget ratios. Returns a new list (never mutates input).
    """
    if not messages:
        return []

    result = copy.deepcopy(messages)

    # Separate system messages (preserve intact)
    system_indices = [i for i, m in enumerate(result) if m.get("role") == "system"]
    non_system = [(i, m) for i, m in enumerate(result) if m.get("role") != "system"]

    if not non_system:
        return result

    # Estimate total tokens for non-system messages
    total_chars = sum(len(m["content"]) for _, m in non_system)
    budget_chars = max_tokens * 4  # convert tokens to chars

    # Subtract system message chars from budget
    system_chars = sum(len(result[i]["content"]) for i in system_indices)
    available_chars = budget_chars - system_chars

    if total_chars <= available_chars:
        return result

    # Need to truncate. Allocate budgets.
    available_chars = max(available_chars, 0)

    # First non-system message gets head_ratio
    first_budget = int(available_chars * head_ratio)
    # Last non-system message gets last_response_ratio
    last_budget = int(available_chars * last_response_ratio)
    # Middle messages share the rest
    middle_count = max(len(non_system) - 2, 0)
    middle_total = max(available_chars - first_budget - last_budget, 0)
    per_turn_cap = int(available_chars * per_turn_ratio)
    per_middle = min(middle_total // max(middle_count, 1), per_turn_cap) if middle_count > 0 else 0

    # Apply truncation
    for idx_in_list, (orig_idx, _msg) in enumerate(non_system):
        content = result[orig_idx]["content"]
        if idx_in_list == 0:
            budget = first_budget
        elif idx_in_list == len(non_system) - 1:
            budget = last_budget
        else:
            budget = per_middle

        if len(content) > budget:
            result[orig_idx]["content"] = _truncate_text(content, budget)

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_truncation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_gateway/truncation.py tests/test_truncation.py
git commit -m "feat(truncation): message budget truncation for OpenAI format"
```

---

## Task 4: runtime.py — circuit breaker, gate, rate limiter, fatal monitor

**Files:**
- Create: `src/llm_gateway/runtime.py`
- Test: `tests/test_runtime.py`

Reference: sft-label `src/sft_label/llm/runtime.py` lines 222-989 — **near-verbatim port**. Do NOT rewrite the async concurrency logic; it contains subtle GC and Condition-race fixes. Differences: drop `probe` param from `AdaptiveRateLimiter.acquire`; drop file I/O from `FatalFailureMonitor`; `RuntimePermit` keeps optional `stage`/`sample_id`.

**Critical porting notes:**
1. `_window` MUST be `collections.deque`; `observe()` MUST stay synchronous (no `await`).
2. `DynamicConcurrencyGate` MUST keep the `_background_tasks` strong-reference set — without it, GC collects the release task, `_in_flight` never decrements, and the gate deadlocks.
3. `FatalFailureMonitor.global_rate_min_observations` default comes from config (200), not the class default.

- [ ] **Step 1: Write the failing test**

```python
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


async def test_gate_acquire_release():
    gate = DynamicConcurrencyGate(initial_limit=2)
    p1 = await gate.acquire()
    p2 = await gate.acquire()
    assert gate.in_flight == 2
    p1.release()
    await asyncio.sleep(0.01)  # release runs as a task
    assert gate.in_flight == 1
    p2.release()
    await asyncio.sleep(0.01)
    assert gate.in_flight == 0


async def test_gate_blocks_at_limit():
    gate = DynamicConcurrencyGate(initial_limit=1)
    p1 = await gate.acquire()
    blocked = asyncio.create_task(gate.acquire())
    await asyncio.sleep(0.05)
    assert not blocked.done()  # second acquire waits
    p1.release()
    p2 = await asyncio.wait_for(blocked, timeout=1.0)
    p2.release()
    await asyncio.sleep(0.01)


async def test_gate_set_limit_wakes_waiters():
    gate = DynamicConcurrencyGate(initial_limit=1)
    p1 = await gate.acquire()
    blocked = asyncio.create_task(gate.acquire())
    await asyncio.sleep(0.05)
    assert not blocked.done()
    gate.set_limit(2)  # raise limit -> waiter proceeds
    p2 = await asyncio.wait_for(blocked, timeout=1.0)
    p1.release()
    p2.release()
    await asyncio.sleep(0.01)


async def test_gate_gc_stress():
    """Regression: release tasks must not be GC'd mid-flight."""
    gate = DynamicConcurrencyGate(initial_limit=50)
    for _ in range(200):
        p = await gate.acquire()
        p.release()
        gc.collect()  # force GC between releases
    await asyncio.sleep(0.1)
    assert gate.in_flight == 0


async def test_rate_limiter_basic_acquire():
    limiter = AdaptiveRateLimiter(target_rps=1000.0)
    await asyncio.wait_for(limiter.acquire(), timeout=1.0)


async def test_rate_limiter_pause_for():
    limiter = AdaptiveRateLimiter(target_rps=1000.0)
    limiter.pause_for(0.2)
    start = time.monotonic()
    await limiter.acquire()
    # first acquire after pause should wait roughly the pause duration
    assert time.monotonic() - start >= 0.15


def test_fatal_monitor_streak():
    m = FatalFailureMonitor(fatal_streak_limit=3, global_rate_min_observations=1000)
    for _ in range(3):
        m.record(OutcomeClass.AUTH_ERROR, error="401")
    assert m.should_abort
    assert "致命" in m.abort_reason or "fatal" in m.abort_reason.lower() or m.abort_reason


def test_fatal_monitor_streak_resets_on_success():
    m = FatalFailureMonitor(fatal_streak_limit=3, global_rate_min_observations=1000)
    m.record(OutcomeClass.AUTH_ERROR, error="401")
    m.record(OutcomeClass.AUTH_ERROR, error="401")
    m.record(OutcomeClass.SUCCESS)  # resets streak
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
    # feed overload outcomes until open
    for _ in range(6):
        rt.observe(RequestOutcome(OutcomeClass.OVERLOAD, status_code=429))
    assert rt.state in ("degraded", "open")
    # enough to open
    for _ in range(6):
        rt.observe(RequestOutcome(OutcomeClass.OVERLOAD, status_code=429))
    assert rt.state == "open"
    snap = rt.snapshot()
    assert snap["state"] == "open"
    assert snap["cooldown_remaining_seconds"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runtime.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_gateway.runtime'`

- [ ] **Step 3: Write the implementation (port from sft-label)**

Create `src/llm_gateway/runtime.py`. Port these classes **verbatim** from `/tmp/sft-label/src/sft_label/llm/runtime.py`, applying only the documented changes:

- `_GatePermit` (lines 222-239) — verbatim
- `DynamicConcurrencyGate` (lines 242-321) — verbatim, keep `_background_tasks`
- `AdaptiveRateLimiter` (lines 324-377) — verbatim EXCEPT remove the `probe` parameter from `acquire()` (delete the `del probe` line and the `*, probe: bool = False` from the signature)
- `RuntimePermit` (lines 380-401) — verbatim (keep `stage`/`sample_id`)
- `AdaptiveLLMRuntime` (lines 404-745) — verbatim
- `FatalFailureMonitor` (lines 753-988) — port, but DELETE: `failure_log_path`/`_failure_log_fp`/`_failure_log_open_failed` attributes, `close()`, `_write_failure_record()`, and all calls to `_write_failure_record`. Keep `record()`, `_check_triggers()`, `should_abort`, `abort_reason`, `to_dict()`.

Import at top:
```python
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
```

Do NOT copy `classify_http_result` / `classify_exception` / `OutcomeClass` / `RequestOutcome` into runtime.py — import them from `outcomes.py`. Do NOT copy `PipelineAbortError` or `monitor_to_outcome_class`.

For `FatalFailureMonitor.__init__`, keep the signature parameter `global_rate_min_observations: int = 20` but the gateway will pass 200 from config. Remove `failure_log_path` from the signature.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_runtime.py -v`
Expected: PASS (all cases including GC stress and state transitions)

- [ ] **Step 5: Commit**

```bash
git add src/llm_gateway/runtime.py tests/test_runtime.py
git commit -m "feat(runtime): circuit breaker, dynamic gate, rate limiter, fatal monitor"
```

---

## Task 5: transport.py — async HTTP transport

**Files:**
- Create: `src/llm_gateway/transport.py`
- Test: `tests/test_transport.py`

Reference: sft-label `src/sft_label/llm/transport.py`. Changes: return `(str | None, dict)` not the 3-tuple; no `json.loads` (return raw `content`); keep the markdown `in_block` stripper as pure text op; drop `rate_limiter` param; `config` required; `adaptive_mode` explicit bool.

**Critical implementation notes:**
1. **Double timeout:** `asyncio.wait_for(http_client.post(..., timeout=T), timeout=T)` — httpx timeout can hang on DNS/connect; `wait_for` is the hard wall-clock cap.
2. **Content extraction:** wrap `data["choices"][0]["message"]["content"]` in `try/except (KeyError, IndexError, TypeError)`; on failure return `(None, {..., "parse_error": True})`.
3. **Markdown strip:** use the `in_block` state machine (sft-label lines 317-329), not regex.

- [ ] **Step 1: Write the failing test**

```python
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
    assert calls["n"] == 1  # no local retry in adaptive mode


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transport.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_gateway.transport'`

- [ ] **Step 3: Write the implementation (port from sft-label)**

Create `src/llm_gateway/transport.py`. Port `RequestStats` verbatim from `/tmp/sft-label/src/sft_label/llm/transport.py` lines 19-62. Then port `async_llm_call` (lines 115-379) with these exact changes:

Signature:
```python
async def async_llm_call(
    http_client,
    messages,
    model,
    *,
    temperature: float = 0.1,
    max_tokens: int = 1000,
    config,               # GatewayConfig, required
    adaptive_mode: bool = True,
) -> tuple[str | None, dict]:
```

Changes from sft-label:
1. Read `_base`/`_key`/`_timeout`/`_escalation` directly from `config` (no LITELLM_* fallback constants, no `getattr` chains).
2. Delete the `rate_limiter` parameter and every `rate_limiter.*` call. `RequestStats` recording happens in the gateway, not here.
3. Replace `runtime = getattr(config, "_adaptive_runtime", None); adaptive_mode = runtime is not None` with the explicit `adaptive_mode` parameter.
4. On success: extract content via
   ```python
   try:
       content = data["choices"][0]["message"]["content"].strip()
   except (KeyError, IndexError, TypeError) as e:
       return None, {"prompt_tokens": 0, "completion_tokens": 0,
                     "status_code": resp.status_code, "parse_error": True,
                     "error": f"malformed response: {e}"}
   ```
5. Keep the markdown `in_block` stripper (lines 317-329) applied to `content`, but **do NOT** `json.loads`. Return `(stripped_content, usage_dict)`.
6. Keep the double-timeout `asyncio.wait_for(http_client.post(..., timeout=attempt_timeout), timeout=float(attempt_timeout))` pattern.
7. Keep: reasoning-model detection, 403 diagnostic headers, 429 "No deployments available"/provider_cooldown, adaptive-mode immediate-return on 429/5xx/400-transient, non-adaptive exponential backoff `min(2**attempt*3+2, 60)` + jitter, 400 non-retryable keywords, 401/402 immediate return, timeout escalation.

Header + imports:
```python
from __future__ import annotations

import asyncio
import random

__all__ = ["async_llm_call", "RequestStats"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transport.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_gateway/transport.py tests/test_transport.py
git commit -m "feat(transport): async HTTP transport with adaptive retry logic"
```

---

## Task 6: recovery.py — swappable client + recovery loop

**Files:**
- Create: `src/llm_gateway/recovery.py`
- Test: `tests/test_recovery.py`

Reference: sft-label `src/sft_label/transport_recovery.py` — **verbatim port** (clean abstraction, no pipeline coupling).

- [ ] **Step 1: Write the failing test**

```python
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
        # forwards .post to underlying client
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
        # runtime stays "open" long enough to exceed threshold
        await asyncio.sleep(2.5)
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

    assert swapped["count"] >= 1
    assert runtime.probing_calls >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_recovery.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_gateway.recovery'`

- [ ] **Step 3: Write the implementation (verbatim port)**

Create `src/llm_gateway/recovery.py` by copying `/tmp/sft-label/src/sft_label/transport_recovery.py` verbatim (both `SwappableAsyncClient` and `transport_recovery_loop`). Only change: add module header + `__all__`:

```python
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Iterable

import httpx

__all__ = ["SwappableAsyncClient", "transport_recovery_loop"]
```

The rest (both classes/functions) is identical to sft-label. No pipeline-specific edits needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_recovery.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_gateway/recovery.py tests/test_recovery.py
git commit -m "feat(recovery): swappable httpx client + transport recovery loop"
```

---

## Task 7: gateway.py + \_\_init\_\_.py — LLMGateway facade

**Files:**
- Create: `src/llm_gateway/gateway.py`
- Modify: `src/llm_gateway/__init__.py`
- Test: `tests/test_gateway.py`

Reference: spec §3-4, sft-label pipeline acquire→call→observe pattern.

**Key design decisions baked in:**
- Rate limiting unified at gateway layer (both adaptive and non-adaptive). Transport never holds a rate_limiter.
- `RequestStats.record()` called in gateway after observe.
- Non-adaptive mode: gateway holds a standalone `AdaptiveRateLimiter` (with warmup) + `asyncio.Semaphore(concurrency)`.
- `should_abort` / `abort_reason` polled externally by caller.

- [ ] **Step 1: Write the failing test**

```python
import asyncio
import json

import httpx
import pytest

from llm_gateway import LLMGateway, GatewayConfig
from llm_gateway.outcomes import OutcomeClass


def _mock_handler(response_fn):
    """Create a handler function for httpx.MockTransport."""
    def handler(request):
        return response_fn(request)
    return handler


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


@pytest.fixture
def gateway_config():
    """Fast-failing config for integration tests."""
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
        transport_stuck_seconds=0,  # disable recovery loop for tests
    )


async def test_successful_call(gateway_config, monkeypatch):
    monkeypatch.setattr(
        "llm_gateway.gateway.httpx.AsyncClient",
        lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(_ok_response("hi")))
    )
    async with LLMGateway(gateway_config) as gw:
        text, usage = await gw.call(
            [{"role": "user", "content": "hello"}],
            "gpt-4o-mini",
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
        lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    # max_request_tokens is 10000 by default; send a message way over budget
    gateway_config.max_request_tokens = 100
    long_msg = "x" * 50000

    async with LLMGateway(gateway_config) as gw:
        await gw.call(
            [{"role": "user", "content": long_msg}],
            "gpt-4o-mini",
        )
    # The sent message should be much shorter than original
    sent_content = captured["messages"][0]["content"]
    assert len(sent_content) < 5000


async def test_circuit_breaker_degraded_on_429(gateway_config, monkeypatch):
    monkeypatch.setattr(
        "llm_gateway.gateway.httpx.AsyncClient",
        lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(_fixed_status(429)))
    )
    async with LLMGateway(gateway_config) as gw:
        # Feed enough 429s to trigger degradation
        for _ in range(6):
            await gw.call([{"role": "user", "content": "hi"}], "gpt-4o-mini")
        snap = gw.runtime_snapshot
        assert snap["state"] in ("degraded", "open")


async def test_should_abort_on_auth_streak(gateway_config, monkeypatch):
    monkeypatch.setattr(
        "llm_gateway.gateway.httpx.AsyncClient",
        lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(_fixed_status(401)))
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
        lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(handler))
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
        lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(_ok_response("hey")))
    )
    async with LLMGateway(config) as gw:
        text, usage = await gw.call(
            [{"role": "user", "content": "hi"}], "gpt-4o-mini",
        )
    assert text == "hey"
    assert usage["status_code"] == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gateway.py -v`
Expected: FAIL with `ImportError: cannot import name 'LLMGateway' from 'llm_gateway'`

- [ ] **Step 3: Write the implementation**

Create `src/llm_gateway/gateway.py`:

```python
"""LLMGateway facade — the single public entry point for LLM calls.

Orchestrates: truncation → admission (gate+limiter) → transport → observe → stats.
Rate limiting is always at this layer (both adaptive and non-adaptive modes).
"""

from __future__ import annotations

import asyncio

import httpx

from llm_gateway.config import GatewayConfig, resolve_httpx_connection_limits
from llm_gateway.outcomes import OutcomeClass, RequestOutcome, classify_http_result
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
    """Build an outcome from transport's usage dict for the circuit breaker."""
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
    # Exception path (timeout, etc)
    err = str(usage.get("error") or "")
    if "timeout" in err.lower():
        return RequestOutcome(OutcomeClass.TIMEOUT, error=err)
    return RequestOutcome(OutcomeClass.TRANSIENT_ERROR, error=err)


class LLMGateway:
    """Async LLM call gateway with circuit breaker, rate limiting, truncation."""

    def __init__(self, config: GatewayConfig):
        self._config = config
        self._http_client: SwappableAsyncClient | None = None
        self._runtime: AdaptiveLLMRuntime | None = None
        self._fatal_monitor: FatalFailureMonitor | None = None
        self._stats = RequestStats()
        self._recovery_task: asyncio.Task | None = None
        # Non-adaptive mode primitives
        self._semaphore: asyncio.Semaphore | None = None
        self._limiter: AdaptiveRateLimiter | None = None

    async def __aenter__(self) -> LLMGateway:
        # httpx connection pool limits
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

        # Adaptive runtime (circuit breaker + gate + rate limiter)
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
            # Non-adaptive: standalone semaphore + rate limiter
            self._semaphore = asyncio.Semaphore(self._config.concurrency)
            self._limiter = AdaptiveRateLimiter(
                target_rps=self._config.rps_limit,
                initial_tokens=1.0,
            )

        # Fatal failure monitor
        if self._config.fatal_abort_enabled:
            self._fatal_monitor = FatalFailureMonitor(
                fatal_streak_limit=self._config.fatal_abort_streak_limit,
                global_failure_rate_limit=self._config.fatal_abort_global_rate_limit,
                global_rate_min_observations=self._config.fatal_abort_global_rate_min_obs,
            )

        # Transport recovery loop
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
        """Make one LLM call. Returns (content_text | None, usage_dict)."""
        # 1. Truncation
        truncated = truncate_messages(
            messages,
            max_tokens=self._config.max_request_tokens,
            head_ratio=self._config.truncation_head_ratio,
            last_response_ratio=self._config.truncation_last_response_ratio,
            per_turn_ratio=self._config.truncation_per_turn_ratio,
        )

        # 2. Admission + 3. Transport + 4. Observe
        if self._runtime is not None:
            # Adaptive mode
            permit = await self._runtime.acquire()
            try:
                content, usage = await async_llm_call(
                    self._http_client, truncated, model,
                    temperature=temperature, max_tokens=max_tokens,
                    config=self._config, adaptive_mode=True,
                )
            finally:
                permit.release()
            # Observe
            outcome = _outcome_from_usage(usage)
            self._runtime.observe(outcome)
        else:
            # Non-adaptive mode
            await self._limiter.acquire()
            async with self._semaphore:
                content, usage = await async_llm_call(
                    self._http_client, truncated, model,
                    temperature=temperature, max_tokens=max_tokens,
                    config=self._config, adaptive_mode=False,
                )
            outcome = _outcome_from_usage(usage)

        # Stats
        status = usage.get("status_code")
        if status:
            self._stats.record(int(status))
        elif outcome.classification is OutcomeClass.TIMEOUT:
            self._stats.record_timeout()

        # Fatal monitor
        if self._fatal_monitor is not None:
            self._fatal_monitor.record(
                outcome.classification,
                error=usage.get("error", ""),
            )

        return content, usage

    @property
    def runtime_snapshot(self) -> dict:
        if self._runtime is not None:
            return self._runtime.snapshot()
        return {"state": "non-adaptive", "concurrency": self._config.concurrency}

    @property
    def http_stats(self) -> dict:
        return self._stats.to_dict()

    @property
    def should_abort(self) -> bool:
        if self._fatal_monitor is None:
            return False
        return self._fatal_monitor.should_abort

    @property
    def abort_reason(self) -> str:
        if self._fatal_monitor is None:
            return ""
        return self._fatal_monitor.abort_reason
```

- [ ] **Step 4: Write `__init__.py` exports**

Update `src/llm_gateway/__init__.py`:

```python
"""LLM Gateway — adaptive async LLM call gateway."""

from llm_gateway.config import GatewayConfig
from llm_gateway.gateway import LLMGateway
from llm_gateway.outcomes import OutcomeClass, RequestOutcome

__all__ = ["LLMGateway", "GatewayConfig", "OutcomeClass", "RequestOutcome"]
```

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/llm_gateway/gateway.py src/llm_gateway/__init__.py tests/test_gateway.py
git commit -m "feat(gateway): LLMGateway facade + public API exports"
```

---

## Self-Review Checklist

1. **Spec coverage:** §1–§10 all addressed. §7 fatal monitor = poll-based `should_abort` ✓. §8 monitoring = `runtime_snapshot` + `http_stats` ✓. §9 module differences = config overrides ✓.
2. **Placeholder scan:** No TBD/TODO/implement-later. All code blocks complete.
3. **Type consistency:** `classify_http_result` signature matches between outcomes.py (Task 1) and gateway.py `_outcome_from_usage` (Task 7). `GatewayConfig` field names used in gateway.py match Task 2 definitions. `async_llm_call` signature in Task 5 matches the call site in Task 7.

---

## Execution Readiness

**All tasks are independently testable:** Each task produces passing tests before the next begins. The dependency order ensures imports resolve at each step.

**Total production files:** 7 + `__init__.py` = 8
**Total test files:** 7 + `conftest.py` = 8
**Estimated implementation time:** 3-4 hours for a skilled developer following TDD steps.
