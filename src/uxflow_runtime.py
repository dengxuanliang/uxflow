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
                '  uv sync --extra local-embed   (or: pip install -e ".[local-embed]")\n'
                f"Or set {_EMBED_ENV}=fake to run without ML dependencies.\n"
                f"  underlying error: {e}"
            ) from e

    if choice == "api":
        from uxflow_embed import ApiEmbedder
        model = os.environ.get("UXFLOW_EMBED_API_MODEL", "text-embedding-3-small")
        dimension = int(os.environ.get("UXFLOW_EMBED_API_DIM", "1024"))
        base_url = os.environ.get("UXFLOW_EMBED_API_BASE", "https://api.openai.com/v1")
        try:
            return ApiEmbedder(model=model, dimension=dimension, base_url=base_url)
        except ValueError as e:
            raise SystemExit(
                f"{_EMBED_ENV}=api needs OPENAI_API_KEY set (see .env.example).\n"
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
