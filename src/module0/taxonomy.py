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

__all__ = ["Taxonomy", "TaxonomyLabel", "TaxonomyStore"]


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
        self._by_name: dict[str, TaxonomyLabel] = {lbl.label: lbl for lbl in labels}

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
        parents = {lbl.parent for lbl in self.labels if lbl.parent}
        return [lbl for lbl in self.labels if lbl.label not in parents]

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
        for lbl in self.labels:
            children.setdefault(lbl.parent, []).append(lbl)

        lines = []
        for root in children.get(None, []):
            lines.append(f"{root.label}  ({root.description})")
            for child in children.get(root.label, []):
                lines.append(f"  ├── {child.label}  {child.description}")
        return "\n".join(lines)


class TaxonomyStore:
    """可变演化层。Taxonomy 保持只读；本类为唯一写入点。"""

    def __init__(self, taxonomy: Taxonomy):
        self._version = taxonomy.version
        self._updated_at = taxonomy.updated_at
        self._labels: list[TaxonomyLabel] = list(taxonomy.labels)

    def snapshot(self) -> Taxonomy:
        """产出只读快照供 prompt 注入 / 查询。"""
        return Taxonomy(
            version=self._version,
            updated_at=self._updated_at,
            labels=list(self._labels),
        )

    def existing_labels(self) -> list[TaxonomyLabel]:
        """当前全部标签（供 ② 算余弦）。"""
        return list(self._labels)

    def add_label(self, label: TaxonomyLabel) -> None:
        """追加新标签 + version patch +1。不影响已有条目。"""
        self._labels.append(label)
        self._version = _bump_patch(self._version)

    def save(self, path) -> None:
        """写回契约 §5.2 结构的 JSON。"""
        data = {
            "version": self._version,
            "updated_at": self._updated_at,
            "labels": [
                {
                    "label": l.label,
                    "parent": l.parent,
                    "new_root": l.new_root,
                    "description": l.description,
                    "keywords": l.keywords,
                    "description_embedding": l.description_embedding,
                    "taxonomy_extension": l.taxonomy_extension,
                    "created_at": l.created_at,
                }
                for l in self._labels
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _bump_patch(version: str) -> str:
    """Increment the patch component of a semver-ish 'a.b.c' string."""
    parts = version.split(".")
    if len(parts) != 3 or not parts[2].isdigit():
        return version  # non-standard version left as-is
    parts[2] = str(int(parts[2]) + 1)
    return ".".join(parts)
