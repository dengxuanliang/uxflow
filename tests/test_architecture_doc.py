# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: docs/architecture.md exists, covers every module, links the interface contract."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARCH_DOC = ROOT / "docs" / "architecture.md"

EXPECTED_MODULES = (
    "module0", "module1", "module2", "module3",
    "module0_5", "llm_gateway", "uxflow_embed", "service",
)


def test_public_architecture_doc_exists():
    assert ARCH_DOC.is_file(), "docs/architecture.md missing — README links it"


def test_architecture_doc_covers_all_modules():
    text = ARCH_DOC.read_text(encoding="utf-8")
    for name in EXPECTED_MODULES:
        assert name in text, f"architecture.md missing module {name}"


def test_architecture_doc_links_interface_contract():
    text = ARCH_DOC.read_text(encoding="utf-8")
    assert "interface-contract.md" in text
