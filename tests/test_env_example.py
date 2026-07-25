# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: .env.example exists, is tracked, lists required keys, has no real secrets."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV_EXAMPLE = ROOT / ".env.example"

REQUIRED_ENV_KEYS = {"LITELLM_BASE", "LITELLM_KEY"}


def test_env_example_exists():
    assert ENV_EXAMPLE.is_file(), ".env.example missing — cloners can't configure"


def test_env_example_is_tracked_by_git():
    import subprocess
    result = subprocess.run(
        ["git", "ls-files", ".env.example"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == ".env.example", (
        ".env.example is not git-tracked — cloners won't receive it")


def test_env_example_lists_required_keys():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    for key in REQUIRED_ENV_KEYS:
        assert re.search(rf"^{key}\s*=", text, re.MULTILINE), (
            f".env.example missing {key}")


def test_env_example_has_no_real_secrets():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert not re.search(r"sk-[A-Za-z0-9]{20,}", text), (
        ".env.example must not contain a real-looking API key")
