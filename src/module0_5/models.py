"""Module 0.5 data models — label proposals + backfill results."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["LabelProposal", "BackfillResult"]


@dataclass
class LabelProposal:
    """模块0 side-channel 产出，0.5 ② 的输入。仅来自 passed 子问题。"""
    label: str
    description: str
    parent: str | None
    description_embedding: list[float]
    keywords: list[str]
    source_sub_problem_id: str


@dataclass
class BackfillResult:
    """④ 回填执行体的显式返回（不抛穿，失败可读）。"""
    label: str
    candidates_screened: int
    judged_true: int
    slices_written: int
    errors: list[str] = field(default_factory=list)
