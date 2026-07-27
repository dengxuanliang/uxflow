# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: README surfaces the service extra, Quickstart, and module0_5 stage."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
README_CN = ROOT / "README.zh-CN.md"


def test_readme_mentions_service_extra_and_inspector():
    text = README.read_text(encoding="utf-8")
    assert "--extra service" in text, "README Install section must list the service extra"
    assert "inspector_serve" in text, "README must show how to start the Inspector"
    assert "Quickstart" in text or "quickstart" in text.lower()


def test_readme_stage_list_includes_module0_5():
    text = README.read_text(encoding="utf-8")
    assert "module0_5" in text or "module 0.5" in text.lower()
    assert "uxflow-evolve" in text


def test_readme_zh_cn_mirrors_install_and_quickstart():
    text = README_CN.read_text(encoding="utf-8")
    assert "--extra service" in text
    assert "inspector_serve" in text


def test_readme_docs_link_targets_architecture_md():
    for readme in (README, README_CN):
        text = readme.read_text(encoding="utf-8")
        assert "docs/architecture.md" in text, (
            f"{readme.name} should link docs/architecture.md, not bare docs/")
