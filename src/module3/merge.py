"""Module 3 跨子问题合并：把共享 (trajectory_id, slice_index) 的候选塌缩为多归属候选。"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["MergedCandidate", "merge_by_slice", "absorb"]


@dataclass
class MergedCandidate:
    """多归属候选：一个 (trajectory_id, slice_index) 唯一样本，可覆盖多个子问题。

    字段名与 ScoredCandidate 尽量兼容，供 selection/dedup/compose duck-typing 读。
    """
    trajectory_id: str
    slice_index: int
    trajectory_path: str
    sub_problem_ids: list[str]
    relevance_by_problem: dict[str, float]
    relevance_score: float
    loss_mask_spans: list[dict]
    embedding: list[float]
    bm25_tokens: list[str]
    capability: list[str] = field(default_factory=list)
    judge_match: bool = True


def _union_spans(spans_lists: list[list[dict]]) -> list[dict]:
    """并集去重（按 start_step/end_step）后升序。"""
    seen: dict[tuple, dict] = {}
    for spans in spans_lists:
        for s in spans or []:
            seen[(s["start_step"], s["end_step"])] = s
    return [seen[k] for k in sorted(seen)]


def _new_from(c: Any) -> MergedCandidate:
    """从一个 ScoredCandidate-like 起一个 MergedCandidate（单归属）。"""
    sub = c.sub_problem_id
    return MergedCandidate(
        trajectory_id=c.trajectory_id,
        slice_index=c.slice_index,
        trajectory_path=getattr(c, "trajectory_path", ""),
        sub_problem_ids=[sub],
        relevance_by_problem={sub: c.relevance_score},
        relevance_score=c.relevance_score,
        loss_mask_spans=list(c.loss_mask_spans or []),
        embedding=list(getattr(c, "embedding", []) or []),
        bm25_tokens=list(getattr(c, "bm25_tokens", []) or []),
        capability=list(getattr(c, "capability", []) or []),
        judge_match=getattr(c, "judge_match", True),
    )


def _add_member(m: MergedCandidate, c: Any) -> None:
    """把一个 ScoredCandidate-like 并入已有 MergedCandidate。"""
    sub = c.sub_problem_id
    if sub not in m.relevance_by_problem:
        m.sub_problem_ids.append(sub)
        m.relevance_by_problem[sub] = c.relevance_score
    else:
        m.relevance_by_problem[sub] = max(m.relevance_by_problem[sub], c.relevance_score)
    m.relevance_score = max(m.relevance_by_problem.values())
    m.loss_mask_spans = _union_spans([m.loss_mask_spans, list(c.loss_mask_spans or [])])
    for cap in getattr(c, "capability", []) or []:
        if cap not in m.capability:
            m.capability.append(cap)


def merge_by_slice(candidates: list[Any]) -> list[MergedCandidate]:
    """按 (trajectory_id, slice_index) 塌缩为多归属候选。保持首次出现顺序。"""
    merged: dict[tuple, MergedCandidate] = {}
    order: list[tuple] = []
    for c in candidates:
        key = (c.trajectory_id, c.slice_index)
        if key not in merged:
            merged[key] = _new_from(c)
            order.append(key)
        else:
            _add_member(merged[key], c)
    return [merged[k] for k in order]


def absorb(survivor: MergedCandidate, dropped: MergedCandidate) -> None:
    """dedup 近重塌缩：把 dropped 的归属/相关度/掩码并入 survivor（保覆盖）。"""
    for sub in dropped.sub_problem_ids:
        r = dropped.relevance_by_problem[sub]
        if sub not in survivor.relevance_by_problem:
            survivor.sub_problem_ids.append(sub)
            survivor.relevance_by_problem[sub] = r
        else:
            survivor.relevance_by_problem[sub] = max(survivor.relevance_by_problem[sub], r)
    survivor.relevance_score = max(survivor.relevance_by_problem.values())
    survivor.loss_mask_spans = _union_spans([survivor.loss_mask_spans, dropped.loss_mask_spans])
    for cap in dropped.capability:
        if cap not in survivor.capability:
            survivor.capability.append(cap)
