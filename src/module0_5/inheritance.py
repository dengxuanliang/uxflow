"""Module 0.5 ③ — inheritance-decayed rerank (query-time, zero LLM, pure)."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

from dataclasses import replace

from module0.taxonomy import Taxonomy
from module1.store import RecallHit

__all__ = ["rerank_with_inheritance"]


def _family(target_label: str, taxonomy: Taxonomy) -> set[str]:
    """target 的父 + 同父兄弟叶子（不含 target 自身）集合，用于继承判定。"""
    target = taxonomy.get(target_label)
    if target is None or target.parent is None:
        return set()
    parent = target.parent
    fam = {parent}
    # NOTE: current taxonomy is two-level, so same-parent labels ARE leaves.
    # For deeper trees, filter siblings to leaves (Taxonomy.leaf_labels()).
    for lbl in taxonomy.labels:
        if lbl.parent == parent:
            fam.add(lbl.label)
    fam.discard(target_label)
    return fam


def rerank_with_inheritance(
    hits: list[RecallHit],
    *,
    target_label: str,
    taxonomy: Taxonomy,
    exact_weight: float = 1.0,
    inherited_weight: float = 0.3,   # 契约 §5.3
) -> list[RecallHit]:
    """对已召回候选按 capability_labels + taxonomy 树加权重排。零 LLM。

    切片含 target_label 本身 → ×exact_weight；
    含 target 的父/兄弟叶子 → ×inherited_weight；
    都不含 → ×1.0（不加成不惩罚，仅靠召回分）。
    权重作用在 hit.rrf_score 上，重排后按加权分降序返回。
    纯函数：不修改入参，返回 replace() 出的新 hit。
    """
    family = _family(target_label, taxonomy)
    rescored = []
    for hit in hits:
        labels = hit.signature.capability_labels or []
        if target_label in labels:
            weight = exact_weight
        elif family & set(labels):
            weight = inherited_weight
        else:
            weight = 1.0
        rescored.append(replace(hit, rrf_score=hit.rrf_score * weight))
    rescored.sort(key=lambda h: h.rrf_score, reverse=True)
    return rescored
