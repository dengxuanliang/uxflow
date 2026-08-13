"""Runtime wiring shared by the entry-point scripts: embedder choice + config guards.

Scripts stay thin: they call make_embedder() / require_llm_config() / resolve_models()
instead of each re-implementing backend selection and env validation.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import os

__all__ = [
    "make_embedder",
    "make_gateway",
    "backend_banner",
    "require_llm_config",
    "resolve_models",
    "DEFAULT_COMPILE_MODEL",
    "DEFAULT_JUDGE_MODEL",
]

# Compile demands strict JSON (module0 Call 1/2); weak models measured 0/3 vs
# gpt-5.5 3/3 and surface as "Call 1 failed after retries". Judge is the
# highest-volume call, so it defaults to a cheaper model.
DEFAULT_COMPILE_MODEL = "gpt-5.5"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"

_EMBED_ENV = "UXFLOW_EMBED_BACKEND"
_VALID_BACKENDS = ("fake", "local", "api")

_LLM_ENV = "UXFLOW_LLM_BACKEND"
_VALID_LLM_BACKENDS = ("real", "fake")


def make_embedder(backend: str | None = None):
    """Build an Embedder from UXFLOW_EMBED_BACKEND (fake | local | api).

    Defaults to 'fake' so a fresh clone runs with zero ML dependencies.
    Fake vectors are deterministic but semantically meaningless — use 'local'
    or 'api' for results you intend to act on.
    """
    choice = (backend or os.environ.get(_EMBED_ENV) or "fake").strip().lower()

    if choice == "fake":
        from uxflow_embed import FakeEmbedder
        return FakeEmbedder()

    if choice == "local":
        from uxflow_embed import LocalEmbedder
        try:
            return LocalEmbedder()
        except ImportError as e:
            raise SystemExit(
                f"{_EMBED_ENV}=local needs the local-embed extra.\n"
                '  uv sync --extra dev --extra service --extra local-embed   (or: pip install -e ".[local-embed]")\n'
                f"Or set {_EMBED_ENV}=fake to run without ML dependencies.\n"
                f"  underlying error: {e}"
            ) from e

    if choice == "api":
        from uxflow_embed import ApiEmbedder

        # LiteLLM first: this project already routes every LLM call through that
        # proxy, and it serves embeddings too, so reusing its credentials means
        # switching to the api backend needs no new .env entries at all.
        api_key = os.environ.get("LITELLM_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = (
            os.environ.get("LITELLM_BASE")
            or os.environ.get("UXFLOW_EMBED_API_BASE")
            or "https://api.openai.com/v1"
        )

        model = os.environ.get("UXFLOW_EMBED_API_MODEL", "text-embedding-3-large")
        dimension = int(os.environ.get("UXFLOW_EMBED_API_DIM", "3072"))

        # Restricted-network knobs. Batch 32 keeps a 3072-dim base64 response
        # near 0.5MB, about half the ~1MB ceiling on the target gateway; 64
        # would be 1.03MB and over the line. Raise it only where the gateway
        # is known to allow more.
        timeout = float(os.environ.get("UXFLOW_EMBED_TIMEOUT", "25"))
        max_tries = int(os.environ.get("UXFLOW_EMBED_MAX_TRIES", "5"))
        batch_size = int(os.environ.get("UXFLOW_EMBED_BATCH", "32"))

        try:
            return ApiEmbedder(
                model=model,
                dimension=dimension,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_tries=max_tries,
                preferred_batch_size=batch_size,
            )
        except ValueError as e:
            raise SystemExit(
                f"{_EMBED_ENV}=api needs LITELLM_KEY or OPENAI_API_KEY set "
                f"(see .env.example).\n"
                f"  underlying error: {e}"
            ) from e

    raise SystemExit(
        f"unknown {_EMBED_ENV}={choice!r}; expected one of {_VALID_BACKENDS}"
    )


def require_llm_config() -> tuple[str, str]:
    """Return (base, key), exiting with actionable guidance if unconfigured.

    Without this, an empty key fails deep inside module0 as the misleading
    "Call 1 failed after retries / empty response".
    """
    base = os.environ.get("LITELLM_BASE", "").strip()
    key = os.environ.get("LITELLM_KEY", "").strip()
    missing = [n for n, v in (("LITELLM_BASE", base), ("LITELLM_KEY", key)) if not v]
    if missing:
        raise SystemExit(
            f"LLM gateway not configured — missing: {', '.join(missing)}\n"
            "  1. cp .env.example .env\n"
            "  2. edit .env and set LITELLM_BASE + LITELLM_KEY\n"
            "  3. verify with: uv run python scripts/check_model_split.py\n"
            "See the Quickstart in README.md."
        )
    return base, key


def resolve_models() -> tuple[str, str]:
    """Return (compile_model, judge_model) — one policy for every entry point.

    MODULE0_TEST_MODEL stays honoured as a legacy alias for the judge slot only;
    it must not drive compile, which needs the strict-JSON-capable model.
    """
    compile_model = os.environ.get("UXFLOW_COMPILE_MODEL", DEFAULT_COMPILE_MODEL)
    judge_model = os.environ.get(
        "UXFLOW_JUDGE_MODEL",
        os.environ.get("MODULE0_TEST_MODEL", DEFAULT_JUDGE_MODEL),
    )
    return compile_model, judge_model


def llm_backend() -> str:
    """Return the selected LLM backend ('real' | 'fake'). Defaults to 'real'.

    Defaulting to 'real' is deliberate: a missing config then fails loudly via
    require_llm_config() instead of silently degrading to scripted replies that
    look like a successful run.
    """
    choice = (os.environ.get(_LLM_ENV) or "real").strip().lower()
    if choice not in _VALID_LLM_BACKENDS:
        raise SystemExit(
            f"unknown {_LLM_ENV}={choice!r}; expected one of {_VALID_LLM_BACKENDS}"
        )
    return choice


def backend_banner() -> str:
    """Render the startup disclosure for the active backends.

    Always reports BOTH backends, even when only one is fake: the easy mistake
    is switching the LLM to real and forgetting embedding (or vice versa), which
    a single-backend message would hide. On an all-real run this prints a plain
    confirmation, so "no warning shown" is itself a meaningful signal rather
    than an absence of information.
    """
    llm = llm_backend()
    embed = (os.environ.get(_EMBED_ENV) or "fake").strip().lower()

    if llm == "real" and embed not in ("fake",):
        return f"✓ LLM: real   ✓ Embedding: {embed}"

    fake_parts = []
    if llm == "fake":
        fake_parts.append("LLM 为固定回放，不是真实模型输出")
    if embed == "fake":
        fake_parts.append("Embedding 向量为确定性哈希，语义无意义")

    def _pad(s: str, width=62) -> str:
        """Pad `s` to `width` visible width, accounting for wide CJK chars."""
        vis = sum(2 if ord(c) > 0x3000 else 1 for c in s)
        return s + " " * (width - vis)

    lines = [
        "╔════════════════════════════════════════════════════════════════╗",
        "║  ⚠️  FAKE 后端 — 本次运行的结果不可用于真实数据筛选             ║",
        "╠════════════════════════════════════════════════════════════════╣",
        f"║  {_pad(f'{_LLM_ENV}={llm}')} ║",
        f"║  {_pad(f'{_EMBED_ENV}={embed}')} ║",
    ]
    for part in fake_parts:
        lines.append(f"║  {_pad('· ' + part)} ║")
    lines.extend([
        f"║  {_pad('仅验证流水线能跑通，不代表选出的轨迹有意义。')} ║",
        f"║  {_pad('')} ║",
        f"║  {_pad('切真实：.env 设 LITELLM_BASE/LITELLM_KEY，并')} ║",
        f"║  {_pad(f'         {_LLM_ENV}=real {_EMBED_ENV}=local')} ║",
        "╚════════════════════════════════════════════════════════════════╝",
    ])
    return "\n".join(lines)


def make_gateway(config_factory=None):
    """Build the LLM gateway from UXFLOW_LLM_BACKEND (real | fake).

    'real' (the default) validates config and returns a live LLMGateway;
    'fake' returns a scripted FakeGateway that needs no proxy. Both are async
    context managers, so callers treat them identically.

    `config_factory` receives (base, key) and returns a GatewayConfig, letting
    each entry point keep its own tuning (e.g. transport_stuck_seconds=0).
    """
    if llm_backend() == "fake":
        from llm_gateway.fake import FakeGateway
        return FakeGateway()

    from llm_gateway import GatewayConfig, LLMGateway
    base, key = require_llm_config()
    if config_factory is not None:
        return LLMGateway(config_factory(base, key))
    return LLMGateway(GatewayConfig(litellm_base=base, litellm_key=key))
