"""Module 0.5 ② — dedup + mount decision for label proposals (pure, zero I/O)."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from module0.taxonomy import TaxonomyLabel, TaxonomyStore
from module0_5.models import LabelProposal

__all__ = ["resolve_proposal", "ProposalResolution", "ingest_proposal"]


@dataclass
class ProposalResolution:
    """② 判定结果。kind ∈ {duplicate, new_leaf, new_root}。"""
    kind: Literal["duplicate", "new_leaf", "new_root"]
    maps_to: str | None = None      # duplicate: 映射到的已有标签名
    parent: str | None = None       # new_leaf: 挂载父节点；new_root: None
    new_root: bool = False


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(va @ vb / (na * nb))


def resolve_proposal(
    proposal: LabelProposal,
    existing: list[TaxonomyLabel],
    *,
    dedup_threshold: float = 0.85,   # 契约 §5.3
    mount_threshold: float = 0.60,   # 契约 §5.3
) -> ProposalResolution:
    """判定：重复→映射已有 / 新叶子→挂父 / 新顶层→new_root。零写入。

    去重：与任一已有标签 cosine > dedup_threshold → duplicate。
    挂载：与顶层节点（parent is None）cosine >= mount_threshold → new_leaf（挂最相似）。
    否则 → new_root。embedding 相似度为权威；proposal.parent 仅在 cosine 接近时作参考。
    """
    emb = proposal.description_embedding

    best_dup, best_dup_sim = None, -1.0
    for lbl in existing:
        sim = _cosine(emb, lbl.description_embedding)
        if sim > best_dup_sim:
            best_dup, best_dup_sim = lbl, sim
    if best_dup is not None and best_dup_sim > dedup_threshold:
        return ProposalResolution(kind="duplicate", maps_to=best_dup.label)

    roots = [lbl for lbl in existing if lbl.parent is None]
    best_root, best_root_sim = None, -1.0
    for root in roots:
        sim = _cosine(emb, root.description_embedding)
        if sim > best_root_sim:
            best_root, best_root_sim = root, sim
    if best_root is not None and best_root_sim >= mount_threshold:
        return ProposalResolution(kind="new_leaf", parent=best_root.label, new_root=False)

    return ProposalResolution(kind="new_root", parent=None, new_root=True)


def ingest_proposal(
    proposal: LabelProposal,
    store: TaxonomyStore,
    *,
    created_at: str,
    dedup_threshold: float = 0.85,
    mount_threshold: float = 0.60,
) -> ProposalResolution:
    """判定 + 写入。duplicate 丢弃不入库；new_leaf/new_root 转 TaxonomyLabel 入库。"""
    res = resolve_proposal(
        proposal, store.existing_labels(),
        dedup_threshold=dedup_threshold, mount_threshold=mount_threshold,
    )
    if res.kind == "duplicate":
        return res  # 丢弃不回填（决策）
    store.add_label(TaxonomyLabel(
        label=proposal.label,
        parent=res.parent,
        new_root=res.new_root,
        description=proposal.description,
        keywords=proposal.keywords,
        description_embedding=proposal.description_embedding,
        taxonomy_extension=True,
        created_at=created_at,
    ))
    return res
