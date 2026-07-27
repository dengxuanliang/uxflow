# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: issue templates have valid YAML frontmatter; PR template exists."""
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
    assert "ruff" in text  # both EN checklist and zh-CN checklist reference it
