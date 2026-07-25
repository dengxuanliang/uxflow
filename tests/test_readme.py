# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: README surfaces the service extra, Quickstart, module0_5, and architecture link."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
README_CN = ROOT / "README.zh-CN.md"


def test_readme_mentions_service_extra_and_inspector():
    for readme in (README, README_CN):
        text = readme.read_text(encoding="utf-8")
        assert "--extra service" in text, (
            f"{readme.name} Install section must list the service extra")
        assert "inspector_serve" in text, (
            f"{readme.name} must show how to start the Inspector")
        assert "Quickstart" in text or "快速上手" in text, (
            f"{readme.name} missing Quickstart section")


def test_readme_stage_list_includes_module0_5():
    for readme in (README, README_CN):
        text = readme.read_text(encoding="utf-8")
        assert "module0_5" in text or "module 0.5" in text.lower(), (
            f"{readme.name} missing module0_5 in stage list")
        assert "uxflow-evolve" in text, (
            f"{readme.name} missing uxflow-evolve CLI mention")


def test_readme_docs_link_targets_architecture_md():
    for readme in (README, README_CN):
        text = readme.read_text(encoding="utf-8")
        assert "docs/architecture.md" in text, (
            f"{readme.name} should link docs/architecture.md, not bare docs/")
