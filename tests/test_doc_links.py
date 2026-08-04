# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
"""Guard: all relative links in public markdown documents resolve on disk."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Public-facing documents that cloners depend on.
_PUBLIC_DOCS = [
    ROOT / "README.md",
    ROOT / "README.zh-CN.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "CONTRIBUTING.zh-CN.md",
    ROOT / "SECURITY.md",
    ROOT / "docs" / "architecture.md",
]

# Matches markdown links like [text](relative/path) — skips http(s):// and # anchors.
_LINK_RE = re.compile(r"\]\((?!https?://|#)([^)]+)\)")


def _strip_anchor(href: str) -> str:
    """Remove a trailing #fragment before resolving."""
    return href.split("#")[0]


def test_all_relative_links_resolve():
    broken = []
    for doc in _PUBLIC_DOCS:
        if not doc.is_file():
            broken.append(f"MISSING DOC: {doc.relative_to(ROOT)}")
            continue
        text = doc.read_text(encoding="utf-8")
        base_dir = doc.parent
        for m in _LINK_RE.finditer(text):
            href = _strip_anchor(m.group(1))
            if not href:
                continue
            target = (base_dir / href).resolve()
            if not target.exists():
                broken.append(
                    f"{doc.relative_to(ROOT)} -> {href}"
                )
    assert not broken, "Broken relative links:\n  " + "\n  ".join(broken)
