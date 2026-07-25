# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: docs/architecture.md exists, covers every module, links the interface contract."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARCH_DOC = ROOT / "docs" / "architecture.md"

EXPECTED_MODULES = (
    "module0", "module1", "module2", "module3",
    "module0_5", "llm_gateway", "uxflow_embed", "service", "uxflow_paths",
)


def test_public_architecture_doc_exists():
    assert ARCH_DOC.is_file(), "docs/architecture.md missing — README links it"


def test_architecture_doc_is_tracked_by_git():
    import subprocess
    result = subprocess.run(
        ["git", "ls-files", "docs/architecture.md"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "docs/architecture.md", (
        "docs/architecture.md is not git-tracked — cloners' links would break")


def test_architecture_doc_covers_all_modules():
    text = ARCH_DOC.read_text(encoding="utf-8")
    for name in EXPECTED_MODULES:
        assert re.search(rf"\b{name}\b", text), (
            f"architecture.md missing module {name}")


def test_architecture_doc_links_interface_contract():
    text = ARCH_DOC.read_text(encoding="utf-8")
    assert "interface-contract.md" in text
