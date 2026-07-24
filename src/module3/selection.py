"""Module 3 submodular-style coverage selection."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["SelectionConfig", "select_set"]


@dataclass
class SelectionConfig:
    n: int
    cap_per_problem: int | None = None
    coverage_cap_per_problem: float | None = None
    min_per_problem: int = 0
    lam: float = 0.1


def _vector(candidate: Any) -> np.ndarray:
    return np.asarray(candidate.embedding, dtype=np.float32)


def _diversity_gain(vec: np.ndarray, chosen_vecs: list[np.ndarray]) -> float:
    if not chosen_vecs:
        return 0.0
    return float(sum(np.linalg.norm(vec - chosen) for chosen in chosen_vecs))


def select_set(
    candidates: list[Any],
    *,
    sub_problem_ids: list[str],
    config: SelectionConfig,
) -> list[Any]:
    """Select candidates under budget, minimum coverage, and optional caps.

    候选为多归属（MergedCandidate）：以 sub_problem_ids/relevance_by_problem 记账。
    一个候选被选中即同时计入它覆盖的所有子问题的 counts/cover。
    """
    if not candidates or config.n <= 0:
        return []

    budget = min(config.n, len(candidates))
    remaining = list(candidates)
    chosen: list[Any] = []
    chosen_vecs: list[np.ndarray] = []
    chosen_keys: set[tuple[str, int]] = set()   # (traj, slice) 去重，防多归属重复 append
    counts = {sub_id: 0 for sub_id in sub_problem_ids}
    cover = {sub_id: 0.0 for sub_id in sub_problem_ids}

    def _rel(c, sub_id):
        return c.relevance_by_problem.get(sub_id, 0.0)

    def _capped(c):
        # 任一归属子问题达 cap → 挡下（保守）
        if config.cap_per_problem is None:
            return False
        return any(counts.get(sid, 0) >= config.cap_per_problem
                   for sid in c.sub_problem_ids)

    def _commit(c):
        chosen.append(c)
        chosen_vecs.append(_vector(c))
        chosen_keys.add((c.trajectory_id, c.slice_index))
        for sid in c.sub_problem_ids:
            counts[sid] = counts.get(sid, 0) + 1
            cover[sid] = cover.get(sid, 0.0) + _rel(c, sid)
        remaining.remove(c)

    if config.min_per_problem > 0:
        for sub_id in sub_problem_ids:
            pool = sorted(
                [c for c in remaining if sub_id in c.sub_problem_ids],
                key=lambda c: _rel(c, sub_id),
                reverse=True,
            )
            taken = 0
            for candidate in pool:
                if taken >= config.min_per_problem or len(chosen) >= budget:
                    break
                if (candidate.trajectory_id, candidate.slice_index) in chosen_keys:
                    taken += 1          # 已被选（可能因别的子问题），计入该子问题保底
                    continue
                if _capped(candidate):
                    continue
                _commit(candidate)
                taken += 1

    while len(chosen) < budget and remaining:
        best = None
        best_gain = float("-inf")
        for candidate in remaining:
            if _capped(candidate):
                continue
            vec = _vector(candidate)
            gain = 0.0
            for sid in candidate.sub_problem_ids:
                cur = cover.get(sid, 0.0)
                r = _rel(candidate, sid)
                if config.coverage_cap_per_problem is None:
                    gain += r
                else:
                    gain += max(0.0, min(cur + r, config.coverage_cap_per_problem) - cur)
            gain += config.lam * _diversity_gain(vec, chosen_vecs)
            if gain > best_gain:
                best = candidate
                best_gain = gain

        if best is None:
            break
        _commit(best)

    return chosen
