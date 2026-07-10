"""Module 2 output model: slice-level scored candidate."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ScoredCandidate"]


@dataclass
class ScoredCandidate:
    """One (slice, sub_problem) scored record."""

    trajectory_id: str
    slice_index: int
    trajectory_path: str
    sub_problem_id: str
    capability: list[str]
    relevance_score: float
    judge_confidence: float
    loss_mask_spans: list[dict]
    judge_match: bool = False
    embedding: list[float] = field(default_factory=list)
    bm25_tokens: list[str] = field(default_factory=list)
