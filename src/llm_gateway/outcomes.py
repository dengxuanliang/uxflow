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
