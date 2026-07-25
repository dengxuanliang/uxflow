# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: .env.example exists, is tracked, lists required keys, has no real secrets, and every active var is read by code."""
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


def test_env_example_active_vars_are_read_by_code():
    # Every active (uncommented) KEY= documented in .env.example must be
    # read by some src/scripts/tests file. Guards against documenting env
    # vars that no code reads.
    import subprocess
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    active = set(re.findall(r"^([A-Z_][A-Z0-9_]*)=", text, re.MULTILINE))
    assert active, "expected at least one active env var in .env.example"
    for var in active:
        result = subprocess.run(
            ["git", "grep", "-l", var, "--", "src/", "scripts/", "tests/"],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert result.stdout.strip(), (
            f".env.example documents '{var}' but no src/scripts/tests "
            f"file on this branch reads it")
