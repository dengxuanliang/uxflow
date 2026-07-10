"""Module 2: relevance re-ranking with soft label scoring."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

from typing import TYPE_CHECKING

from module2.models import ScoredCandidate

if TYPE_CHECKING:
    from module1.models import JudgeResult
    from module1.store import RecallHit

__all__ = ["rerank"]

_MISS_DECAY = 0.3


def rerank(
    hits: list["RecallHit"],
    judge_results: list["JudgeResult"],
    sub_problem: dict,
    *,
    trajectory_path: str,
) -> list[ScoredCandidate]:
    """Score recalled hits against judge verdicts and sort by relevance."""
    if len(hits) != len(judge_results):
        raise ValueError("hits and judge_results must have the same length")

    sub_id = sub_problem.get("id", "unknown")
    capability = sub_problem.get("target_capability", [])

    scored: list[ScoredCandidate] = []
    for hit, jr in zip(hits, judge_results):
        multiplier = 1.0 if jr.match else _MISS_DECAY
        scored.append(
            ScoredCandidate(
                trajectory_id=hit.signature.trajectory_id,
                slice_index=hit.signature.slice_index,
                trajectory_path=trajectory_path,
                sub_problem_id=sub_id,
                capability=capability,
                relevance_score=hit.rrf_score * multiplier,
                judge_confidence=jr.confidence,
                loss_mask_spans=jr.spans,
                judge_match=jr.match,
                embedding=hit.signature.embedding,
                bm25_tokens=hit.signature.bm25_tokens,
            )
        )
    scored.sort(key=lambda c: c.relevance_score, reverse=True)
    return scored
