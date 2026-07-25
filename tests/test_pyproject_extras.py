# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: pyproject.toml advertises only extras that actually have backends."""
import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"

LIVE_EXTRAS = {"dev", "service", "local-embed"}


def _optional_dependencies() -> dict:
    with PYPROJECT.open("rb") as f:
        data = tomllib.load(f)
    return data["project"]["optional-dependencies"]


def test_no_dead_extras():
    extras = _optional_dependencies()
    assert "onnx-embed" not in extras, (
        "onnx-embed extra has zero backends — drop it or implement it")


def test_live_extras_present():
    extras = _optional_dependencies()
    for extra in LIVE_EXTRAS:
        assert extra in extras, f"pyproject.toml missing live extra: {extra}"
