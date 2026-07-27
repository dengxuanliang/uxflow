# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: pyproject.toml advertises only extras that actually have backends."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"

LIVE_EXTRAS = {"dev", "service", "local-embed"}


def test_no_dead_extras():
    text = PYPROJECT.read_text(encoding="utf-8")
    assert "onnx-embed" not in text, (
        "onnx-embed extra has zero backends — drop it or implement it")


def test_live_extras_present():
    text = PYPROJECT.read_text(encoding="utf-8")
    for extra in LIVE_EXTRAS:
        assert extra in text, f"pyproject.toml missing live extra: {extra}"
