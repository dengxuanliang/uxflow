"""Assign stable colors to taxonomy labels (spec §5.3)."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

__all__ = ["PALETTE", "assign_colors"]

# 继承 brainstorm mockup 的配色，扩展到 10 色调色板
PALETTE = (
    "#f5b800",  # 黄 (mockup valid_syntax)
    "#1c7ed6",  # 蓝 (mockup self_verification)
    "#37b24d",  # 绿
    "#e8590c",  # 橙
    "#ae3ec9",  # 紫
    "#0ca678",  # 青
    "#e64980",  # 粉
    "#4263eb",  # 靛
    "#f76707",  # 深橙
    "#66a80f",  # 橄榄
)


def assign_colors(labels: list[str]) -> dict[str, str]:
    """Deterministically map each label to a stable palette color.

    Assignment is order-independent: labels are sorted before indexing, so the
    same set of labels always yields the same label→color mapping regardless of
    input order. Palette wraps (mod) when labels exceed palette size.
    """
    unique_sorted = sorted(set(labels))
    return {
        label: PALETTE[i % len(PALETTE)]
        for i, label in enumerate(unique_sorted)
    }
