# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: the LiteLLM example config exposes exactly the model names UXFlow requests.

A mismatch here fails deep inside module0 as the misleading
"Call 1 failed after retries", so it is worth catching at test time.

Parsed with a regex rather than PyYAML: yaml is only a transitive dependency
here, and this file is simple enough that adding a direct one isn't warranted.
"""
import pathlib
import re

from uxflow_runtime import DEFAULT_COMPILE_MODEL, DEFAULT_JUDGE_MODEL

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG = ROOT / "litellm.config.example.yaml"
COMPOSE = ROOT / "docker-compose.litellm.yml"

_MODEL_NAME_RE = re.compile(r"^\s*-\s*model_name:\s*(\S+)\s*$", re.MULTILINE)


def _declared_model_names() -> list[str]:
    return _MODEL_NAME_RE.findall(CONFIG.read_text(encoding="utf-8"))


def test_example_config_exists():
    assert CONFIG.is_file(), "litellm.config.example.yaml missing — README references it"
    assert COMPOSE.is_file(), "docker-compose.litellm.yml missing — README references it"


def test_config_serves_uxflow_default_models():
    names = _declared_model_names()
    for default in (DEFAULT_COMPILE_MODEL, DEFAULT_JUDGE_MODEL):
        assert default in names, (
            f"litellm.config.example.yaml does not serve {default!r} "
            f"(declares {names}) — a fresh clone would fail with "
            f'"Call 1 failed after retries"'
        )


def test_compose_publishes_the_port_env_example_points_at():
    """.env.example defaults LITELLM_BASE to localhost:4000; compose must match."""
    compose = COMPOSE.read_text(encoding="utf-8")
    assert '"4000:4000"' in compose, "compose must publish 4000 to match LITELLM_BASE"

    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "localhost:4000" in env_example, (
        ".env.example no longer points at port 4000 — compose mapping is now stale")


def test_compose_pins_an_image_version():
    """Unpinned :latest silently rolls; LiteLLM had a supply-chain incident."""
    compose = COMPOSE.read_text(encoding="utf-8")
    m = re.search(r"^\s*image:\s*(\S+)\s*$", compose, re.MULTILINE)
    assert m, "no image declared in docker-compose.litellm.yml"
    image = m.group(1)
    assert ":" in image and not image.endswith(":latest"), (
        f"pin the LiteLLM image to a version instead of {image!r}")
