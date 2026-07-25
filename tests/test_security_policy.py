# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: SECURITY.md exists and documents private reporting + the .env rule."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SECURITY = ROOT / "SECURITY.md"


def test_security_policy_exists():
    assert SECURITY.is_file(), "SECURITY.md missing"


def test_security_policy_is_tracked_by_git():
    import subprocess
    result = subprocess.run(
        ["git", "ls-files", "SECURITY.md"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "SECURITY.md", (
        "SECURITY.md is not git-tracked — GitHub's Security tab won't surface it")


def test_security_policy_documents_private_reporting():
    text = SECURITY.read_text(encoding="utf-8")
    assert "GitHub Security Advisories" in text or "security advisory" in text.lower()


def test_security_policy_mentions_env_rule():
    text = SECURITY.read_text(encoding="utf-8")
    assert ".env" in text  # must remind that real keys never get committed
