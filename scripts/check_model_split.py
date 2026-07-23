"""Verify the compile/judge model split actually reaches both models.

Isolates ONE question from the full pipeline: do the two model names in
UXFLOW_COMPILE_MODEL / UXFLOW_JUDGE_MODEL each route through the LiteLLM
gateway and return a live response?

This is a connectivity + naming check, NOT a business-logic check. It sends a
trivial prompt to each model and reports pass/fail per model with the failure
cause surfaced, so a red result tells you *which* model and *why* (404 → not
registered in LiteLLM config.yaml; timeout → gateway unreachable; etc.).

Usage:
    .venv/bin/python scripts/check_model_split.py

Exit code 0 = both models answered; 1 = at least one failed.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

import asyncio
import os
import pathlib
import sys

from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent.parent / ".env")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from llm_gateway import GatewayConfig, LLMGateway  # noqa: E402

# Same defaults as inspector_serve.build_app — keep these in sync so this
# check exercises exactly what production wiring resolves.
_COMPILE_DEFAULT = "gpt-5.5"
_JUDGE_DEFAULT = "gpt-4o-mini"

_PING_MESSAGES = [
    {"role": "user", "content": "Reply with the single word: pong"},
]


async def _probe(gateway: LLMGateway, role: str, model: str) -> bool:
    """Send one trivial call; return True if the model produced content."""
    print(f"\n[{role}] model = {model!r}")
    try:
        content, usage = await gateway.call(_PING_MESSAGES, model, max_tokens=16)
    except Exception as exc:  # noqa: BLE001 — this probe must report, not raise
        print(f"  ✗ call raised: {type(exc).__name__}: {exc}")
        return False

    status = usage.get("status_code")
    if content and content.strip():
        print(f"  ✓ answered (status={status}): {content.strip()[:80]!r}")
        return True

    # No content: surface everything useful for diagnosis.
    err = usage.get("error")
    print(f"  ✗ empty response (status={status}, error={err!r})")
    return False


async def _run() -> int:
    compile_model = os.environ.get("UXFLOW_COMPILE_MODEL", _COMPILE_DEFAULT)
    judge_model = os.environ.get("UXFLOW_JUDGE_MODEL", _JUDGE_DEFAULT)

    base = os.environ.get("LITELLM_BASE", "http://localhost:4000/v1")
    key = os.environ.get("LITELLM_KEY", "")
    print("=== model split connectivity check ===")
    print(f"LITELLM_BASE      = {base}")
    print(f"LITELLM_KEY       = {'<set>' if key else '<empty>'}")
    print(f"UXFLOW_COMPILE_MODEL = {compile_model!r}")
    print(f"UXFLOW_JUDGE_MODEL   = {judge_model!r}")

    if compile_model == judge_model:
        print("\n⚠  compile and judge resolve to the SAME model — split is a no-op.")

    config = GatewayConfig(
        litellm_base=base,
        litellm_key=key,
        transport_stuck_seconds=0,
    )
    gateway = LLMGateway(config)
    await gateway.__aenter__()
    try:
        results = []
        # Probe compile and judge concurrently — both go through the one gateway.
        results = await asyncio.gather(
            _probe(gateway, "compile", compile_model),
            _probe(gateway, "judge", judge_model),
        )
    finally:
        await gateway.__aexit__(None, None, None)

    compile_ok, judge_ok = results
    print("\n=== result ===")
    print(f"  compile ({compile_model}): {'PASS' if compile_ok else 'FAIL'}")
    print(f"  judge   ({judge_model}): {'PASS' if judge_ok else 'FAIL'}")

    if compile_ok and judge_ok:
        print("\n✓ both models answered — split is live.")
        return 0
    print("\n✗ split NOT verified — see failures above.")
    print("  404-ish / empty → model name not registered in LiteLLM config.yaml")
    print("  timeout / conn  → gateway at LITELLM_BASE not reachable")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
