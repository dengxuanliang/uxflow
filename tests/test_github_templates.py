# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: issue templates have valid YAML frontmatter; PR template exists + tracked."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
ISSUE_DIR = ROOT / ".github" / "ISSUE_TEMPLATE"

EXPECTED_ISSUE_TEMPLATES = {
    "bug_report.md",
    "bug_report.zh-CN.md",
    "feature_request.md",
    "feature_request.zh-CN.md",
}


def _frontmatter(text: str) -> str:
    m = re.match(r"^---\n(.*?\n)---\n", text, re.DOTALL)
    assert m, "missing YAML frontmatter"
    return m.group(1)


def test_issue_templates_present():
    files = {p.name for p in ISSUE_DIR.glob("*.md")}
    missing = EXPECTED_ISSUE_TEMPLATES - files
    assert not missing, f"missing issue templates: {missing}"


def test_issue_templates_are_tracked_by_git():
    import subprocess
    expected_tracked = {
        f".github/ISSUE_TEMPLATE/{name}"
        for name in EXPECTED_ISSUE_TEMPLATES
    } | {".github/ISSUE_TEMPLATE/config.yml"}
    result = subprocess.run(
        ["git", "ls-files", ".github/ISSUE_TEMPLATE/"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    tracked = {line for line in result.stdout.splitlines() if line}
    missing = expected_tracked - tracked
    assert not missing, (
        f"these ISSUE_TEMPLATE files are not git-tracked (invisible in chooser): {missing}")


def test_issue_templates_have_valid_frontmatter():
    for name in EXPECTED_ISSUE_TEMPLATES:
        text = (ISSUE_DIR / name).read_text(encoding="utf-8")
        fm = _frontmatter(text)
        assert re.search(r"^name:\s*\S", fm, re.MULTILINE), (
            f"{name} frontmatter missing 'name:'")


def test_issue_template_config_yaml_exists():
    cfg = ISSUE_DIR / "config.yml"
    assert cfg.is_file(), ".github/ISSUE_TEMPLATE/config.yml missing"


def test_pull_request_template_exists():
    prt = ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"
    assert prt.is_file(), "PULL_REQUEST_TEMPLATE.md missing"
    text = prt.read_text(encoding="utf-8")
    assert "pytest" in text.lower() and "ruff" in text.lower()
    assert text.lower().count("ruff") >= 2, (
        "ruff should appear in both the EN and zh-CN checklist items")


def test_pull_request_template_is_tracked_by_git():
    import subprocess
    result = subprocess.run(
        ["git", "ls-files", ".github/PULL_REQUEST_TEMPLATE.md"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == ".github/PULL_REQUEST_TEMPLATE.md", (
        "PULL_REQUEST_TEMPLATE.md is not git-tracked — GitHub would silently ignore it")
