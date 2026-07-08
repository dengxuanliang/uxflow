"""Async HTTP transport for the LiteLLM /chat/completions endpoint.

Ported from sft-label's ``async_llm_call`` with these deliberate changes:

* Returns a 2-tuple ``(content_str | None, usage_dict)`` — the raw assistant
  content string, NOT parsed business JSON. The caller is responsible for
  parsing. On any content-extraction failure the content is ``None`` and the
  usage dict carries ``parse_error=True``.
* No ``rate_limiter`` parameter and no stats recording here — rate limiting and
  stats live in the gateway layer.
* ``config`` (GatewayConfig) is required; ``adaptive_mode`` is an explicit bool.

The markdown-fence stripper is retained as a pure text operation applied to the
content string (state-machine, no regex).
"""

from __future__ import annotations

import asyncio
import random

__all__ = ["async_llm_call", "RequestStats"]


class RequestStats:
    """Track HTTP request outcomes for real-time display and summary."""
    __slots__ = ('success', 'errors', 'timeouts')

    def __init__(self):
        self.success = 0
        self.errors = {}   # status_code (int) -> count
        self.timeouts = 0

    def record(self, status_code: int):
        if 200 <= status_code < 300:
            self.success += 1
        else:
            self.errors[status_code] = self.errors.get(status_code, 0) + 1

    def record_timeout(self):
        self.timeouts += 1

    @property
    def total(self):
        return self.success + sum(self.errors.values()) + self.timeouts

    def summary_line(self):
        """One-line summary for progress display."""
        t = self.total
        if t == 0:
            return ""
        parts = [f"✓{self.success}"]
        for code in sorted(self.errors):
            parts.append(f"{code}×{self.errors[code]}")
        if self.timeouts:
            parts.append(f"timeout×{self.timeouts}")
        rate = self.success / t * 100
        return f"http({' '.join(parts)} {rate:.0f}%)"

    def to_dict(self):
        t = self.total
        return {
            "total_http_requests": t,
            "success": self.success,
            "errors": {str(k): v for k, v in sorted(self.errors.items())},
            "timeouts": self.timeouts,
            "success_rate": round(self.success / t * 100, 1) if t > 0 else 0,
        }


# ─────────────────────────────────────────────────────────
# Async LLM call
# ─────────────────────────────────────────────────────────

def _strip_markdown_fence(text: str) -> str:
    """Strip a leading/trailing markdown code fence, pure text state-machine.

    Mirrors sft-label's stripper: if the content starts with ```, keep the
    lines between the opening fence and the next closing fence. No regex.
    """
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    inner = []
    in_block = False
    for line in lines:
        if line.startswith("```") and not in_block:
            in_block = True
            continue
        elif line.startswith("```") and in_block:
            break
        elif in_block:
            inner.append(line)
    return "\n".join(inner)


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
    """Single HTTP call to LiteLLM /chat/completions with retry handling.

    Returns ``(content_str | None, usage_dict)``. The content is the raw
    ``choices[0].message.content`` string (markdown fence stripped); it is
    ``None`` on any error or extraction failure. The usage dict always carries
    ``status_code`` when an HTTP response was received, plus diagnostic fields
    (``error``, ``error_response``, ``non_retryable``, ``content_filtered``,
    ``provider_cooldown``, ``parse_error``) where relevant.
    """
    _base = config.litellm_base
    _key = config.litellm_key
    _timeout = config.request_timeout
    _escalation = config.request_timeout_escalation
    max_retries = config.max_retries

    url = f"{_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {_key}",
        "Content-Type": "application/json",
        "User-Agent": "llm-gateway/0.1.0",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Reasoning models (o1, o3, gpt-5*) don't support temperature or max_tokens
    _model_lower = model.lower()
    if any(p in _model_lower for p in ("o1", "o3", "gpt-5")):
        payload.pop("temperature", None)
        payload["max_completion_tokens"] = payload.pop("max_tokens")

    last_error = None
    last_error_response = None

    for attempt in range(max_retries + 1):
        # Adaptive timeout: escalate on retries
        attempt_timeout = (_escalation[attempt] if _escalation and attempt < len(_escalation)
                           else _timeout)
        try:
            if float(attempt_timeout or 0) > 0:
                resp = await asyncio.wait_for(
                    http_client.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=attempt_timeout,
                    ),
                    timeout=float(attempt_timeout),
                )
            else:
                resp = await http_client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=attempt_timeout,
                )

            if resp.status_code == 403:
                # Content filtered by upstream WAF/provider — retry once to rule out transient proxy issues
                resp_body = resp.text
                # Capture diagnostic headers to identify which layer returned the 403:
                #   x-litellm-* → LiteLLM or upstream provider
                #   via/x-squid/x-cache → corporate proxy
                #   server → nginx/uvicorn/squid/etc
                diag_keys = ("server", "via", "x-squid-error", "x-cache",
                             "content-type", "x-litellm-version",
                             "x-litellm-model-group", "x-litellm-call-id")
                diag_headers = {k: resp.headers[k] for k in diag_keys if k in resp.headers}
                is_html = "text/html" in resp.headers.get("content-type", "")
                diag_src = ("litellm/upstream" if "x-litellm-version" in diag_headers
                            else "proxy/WAF" if any(k in diag_headers for k in ("via", "x-squid-error", "x-cache"))
                            else f"server={diag_headers.get('server', '?')}")
                diag_hint = f" [source={diag_src}]"
                if is_html:
                    diag_hint += " [HTML response — likely WAF/proxy block page]"
                last_error = f"HTTP 403: {resp_body[:300]}{diag_hint}"
                if attempt < 1:
                    await asyncio.sleep(3 + random.uniform(0, 2))
                    continue
                return None, {
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "status_code": resp.status_code,
                    "error": last_error,
                    "error_response": resp_body,
                    "error_diag_headers": diag_headers,
                    "error_is_html": is_html,
                    "non_retryable": True,
                }

            if resp.status_code in (429, 500, 502, 503, 504):
                resp_text = resp.text[:500]
                resp_lower = resp_text.lower()
                # LiteLLM "No deployments available" — all deployments are in cooldown.
                # In adaptive mode the outer controller opens the circuit and pauses;
                # in legacy mode treat as non-retryable to avoid local retry storms.
                if "no deployments available" in resp_lower or "cooldown_list" in resp_lower:
                    return None, {
                        "prompt_tokens": 0, "completion_tokens": 0,
                        "status_code": resp.status_code,
                        "error": f"HTTP {resp.status_code} (deployment cooldown): {resp_text[:300]}",
                        "error_response": resp.text,
                        "provider_cooldown": True,
                        "non_retryable": False if adaptive_mode else True,
                    }
                # Rate limited or server error.
                if adaptive_mode:
                    return None, {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "status_code": resp.status_code,
                        "error": f"HTTP {resp.status_code}: {resp_text[:300]}",
                        "error_response": resp.text,
                        "non_retryable": False,
                    }
                # Non-adaptive: exponential backoff with jitter
                base_wait = min(2 ** attempt * 3 + 2, 60)
                wait = base_wait + random.uniform(0, base_wait * 0.5)
                last_error = f"HTTP {resp.status_code}: {resp_text[:200]}"
                last_error_response = resp.text
                if attempt < max_retries:
                    await asyncio.sleep(wait)
                    continue
                return None, {
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "status_code": resp.status_code,
                    "error": last_error,
                    "error_response": resp.text,
                    "non_retryable": False,
                }

            if resp.status_code == 400:
                error_text = resp.text[:500]
                error_lower = error_text.lower()
                # Genuinely non-retryable: the request content itself is invalid.
                # Keep this list narrow — broad keywords can match transient routing errors.
                _NON_RETRYABLE_400_KEYWORDS = (
                    "context_length_exceeded", "maximum context length",
                    "content_policy", "content_filter",
                    "responsibleaipolicyviolation", "content_management_policy",
                    "moderation",
                )
                if any(kw in error_lower for kw in _NON_RETRYABLE_400_KEYWORDS):
                    return None, {
                        "prompt_tokens": 0, "completion_tokens": 0,
                        "status_code": resp.status_code,
                        "error": f"HTTP 400 (content filtered): {error_text[:300]}",
                        "error_response": resp.text,
                        "content_filtered": True,
                        "non_retryable": True,
                    }
                # Likely a transient proxy/supplier error.
                if adaptive_mode:
                    return None, {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "status_code": resp.status_code,
                        "error": f"HTTP 400 (transient): {error_text[:300]}",
                        "error_response": resp.text,
                        "non_retryable": False,
                    }
                last_error = f"HTTP 400 (transient): {error_text[:200]}"
                last_error_response = resp.text
                if attempt < max_retries:
                    base_wait = min(2 ** attempt * 3 + 2, 60)
                    await asyncio.sleep(base_wait + random.uniform(0, base_wait * 0.5))
                    continue
                return None, {
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "status_code": resp.status_code,
                    "error": last_error,
                    "error_response": resp.text,
                    "non_retryable": False,
                }

            if resp.status_code == 401:
                # Auth failure — not retryable
                error_text = resp.text[:300]
                return None, {
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "status_code": resp.status_code,
                    "error": f"HTTP 401: {error_text}",
                    "error_response": resp.text,
                    "non_retryable": True,
                }

            if resp.status_code == 402:
                # Billing exhausted — not retryable (same as 401)
                error_text = resp.text[:300]
                return None, {
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "status_code": resp.status_code,
                    "error": f"HTTP 402 (billing): {error_text}",
                    "error_response": resp.text,
                    "non_retryable": True,
                }

            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {}) if isinstance(data, dict) else {}

            # Content extraction — return raw string, caller parses.
            try:
                content = data["choices"][0]["message"]["content"]
                content = _strip_markdown_fence(content)
            except (KeyError, IndexError, TypeError) as e:
                return None, {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "status_code": resp.status_code,
                    "error": f"ParseError: {e}",
                    "parse_error": True,
                    "non_retryable": False,
                }

            return content, {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "status_code": resp.status_code,
            }

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if adaptive_mode:
                return None, {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "error": last_error,
                    "exception_type": type(e).__name__,
                    "non_retryable": False,
                }
            if attempt < max_retries:
                base_wait = min(2 ** attempt * 3 + 2, 60)
                wait = base_wait + random.uniform(0, base_wait * 0.5)
                await asyncio.sleep(wait)
                continue
            return None, {
                "prompt_tokens": 0, "completion_tokens": 0,
                "error": last_error,
                "exception_type": type(e).__name__,
            }

    return None, {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": last_error or "max_retries",
        "error_response": last_error_response,
    }
