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
    """Select candidates under budget, minimum coverage, and optional caps."""
    if not candidates or config.n <= 0:
        return []

    budget = min(config.n, len(candidates))
    remaining = list(candidates)
    chosen: list[Any] = []
    chosen_vecs: list[np.ndarray] = []
    counts = {sub_id: 0 for sub_id in sub_problem_ids}
    cover = {sub_id: 0.0 for sub_id in sub_problem_ids}

    if config.min_per_problem > 0:
        for sub_id in sub_problem_ids:
            pool = sorted(
                [c for c in remaining if c.sub_problem_id == sub_id],
                key=lambda c: c.relevance_score,
                reverse=True,
            )
            for candidate in pool[: config.min_per_problem]:
                if len(chosen) >= budget:
                    break
                if (
                    config.cap_per_problem is not None
                    and counts.get(candidate.sub_problem_id, 0) >= config.cap_per_problem
                ):
                    continue
                chosen.append(candidate)
                chosen_vecs.append(_vector(candidate))
                counts[candidate.sub_problem_id] = counts.get(candidate.sub_problem_id, 0) + 1
                cover[candidate.sub_problem_id] = (
                    cover.get(candidate.sub_problem_id, 0.0)
                    + candidate.relevance_score
                )
                remaining.remove(candidate)

    while len(chosen) < budget and remaining:
        best = None
        best_gain = float("-inf")
        for candidate in remaining:
            if (
                config.cap_per_problem is not None
                and counts.get(candidate.sub_problem_id, 0) >= config.cap_per_problem
            ):
                continue
            vec = _vector(candidate)
            current_cover = cover.get(candidate.sub_problem_id, 0.0)
            if config.coverage_cap_per_problem is None:
                coverage_gain = candidate.relevance_score
            else:
                coverage_gain = max(
                    0.0,
                    min(
                        current_cover + candidate.relevance_score,
                        config.coverage_cap_per_problem,
                    )
                    - current_cover,
                )
            gain = coverage_gain + config.lam * _diversity_gain(vec, chosen_vecs)
            if gain > best_gain:
                best = candidate
                best_gain = gain

        if best is None:
            break

        chosen.append(best)
        chosen_vecs.append(_vector(best))
        counts[best.sub_problem_id] = counts.get(best.sub_problem_id, 0) + 1
        cover[best.sub_problem_id] = (
            cover.get(best.sub_problem_id, 0.0) + best.relevance_score
        )
        remaining.remove(best)

    return chosen
