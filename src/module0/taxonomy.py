"""Taxonomy read-only layer.

Loads a taxonomy JSON file (contract §5 schema), provides:
- Label lookup by name
- Leaf vs parent classification
- Prompt injection text generation (indented tree for LLM context)

Does NOT write back or evolve taxonomy (that's Plan B / module 0.5).
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

__all__ = ["Taxonomy", "TaxonomyLabel"]


@dataclass(frozen=True)
class TaxonomyLabel:
    """契约 §5.1 TaxonomyLabel（只读视图）。"""
    label: str
    parent: str | None
    new_root: bool
    description: str
    keywords: list[str]
    description_embedding: list[float]
    taxonomy_extension: bool
    created_at: str


class Taxonomy:
    """Read-only taxonomy store."""

    def __init__(self, version: str, updated_at: str, labels: list[TaxonomyLabel]):
        self.version = version
        self.updated_at = updated_at
        self.labels = labels
        self._by_name: dict[str, TaxonomyLabel] = {l.label: l for l in labels}

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Taxonomy":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Taxonomy":
        labels = []
        for entry in data.get("labels", []):
            labels.append(TaxonomyLabel(
                label=entry["label"],
                parent=entry.get("parent"),
                new_root=entry.get("new_root", False),
                description=entry["description"],
                keywords=entry.get("keywords", []),
                description_embedding=entry.get("description_embedding", []),
                taxonomy_extension=entry.get("taxonomy_extension", False),
                created_at=entry.get("created_at", ""),
            ))
        return cls(
            version=data.get("version", "0.0.0"),
            updated_at=data.get("updated_at", ""),
            labels=labels,
        )

    @property
    def is_empty(self) -> bool:
        return len(self.labels) == 0

    def get(self, label_name: str) -> TaxonomyLabel | None:
        return self._by_name.get(label_name)

    def leaf_labels(self) -> list[TaxonomyLabel]:
        """Labels that are not parent of any other label."""
        parents = {l.parent for l in self.labels if l.parent}
        return [l for l in self.labels if l.label not in parents]

    def to_prompt_text(self) -> str:
        """Generate indented tree text for LLM prompt injection.

        Format (per spec Part 2):
          parent_label  (中文description)
            ├── child_label  中文description
        """
        if not self.labels:
            return ""

        # Group children by parent
        children: dict[str | None, list[TaxonomyLabel]] = {}
        for l in self.labels:
            children.setdefault(l.parent, []).append(l)

        lines = []
        for root in children.get(None, []):
            lines.append(f"{root.label}  ({root.description})")
            for child in children.get(root.label, []):
                lines.append(f"  ├── {child.label}  {child.description}")
        return "\n".join(lines)
