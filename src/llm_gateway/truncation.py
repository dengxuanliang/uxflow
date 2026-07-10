"""Pre-send message budget truncation (OpenAI format).

Ensures the total input token estimate stays under max_request_tokens
before hitting the network. System messages are never truncated.
Budget is distributed by ratio: first-user (head), last-assistant (tail),
middle turns share the remainder.

Token estimation: len(text) // 4 (fast, no external dependency).
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import copy

__all__ = [
    "TRUNCATION_MARKER",
    "estimate_tokens",
    "truncate_messages",
]

TRUNCATION_MARKER = "\n\n[... content truncated ...]\n\n"


def estimate_tokens(text: str) -> int:
    """Approximate token count. 1 token ~= 4 chars for English/code."""
    return max(len(text) // 4, 0)


def _truncate_text(text: str, max_chars: int, keep_head_ratio: float = 0.3) -> str:
    """Truncate a single text blob, keeping head + tail with marker in between."""
    if len(text) <= max_chars:
        return text
    marker_len = len(TRUNCATION_MARKER)
    usable = max(max_chars - marker_len, 0)
    head_len = max(int(usable * keep_head_ratio), 1)
    tail_len = max(usable - head_len, 1)
    return text[:head_len] + TRUNCATION_MARKER + text[-tail_len:]


def truncate_messages(
    messages: list[dict],
    *,
    max_tokens: int = 10000,
    head_ratio: float = 0.35,
    last_response_ratio: float = 0.30,
    per_turn_ratio: float = 0.35,
) -> list[dict]:
    """Truncate messages to fit within token budget.

    System messages (role=='system') are preserved intact and don't count
    toward the budget ratios. Returns a new list (never mutates input).
    """
    if not messages:
        return []

    result = copy.deepcopy(messages)

    system_indices = [i for i, m in enumerate(result) if m.get("role") == "system"]
    non_system = [(i, m) for i, m in enumerate(result) if m.get("role") != "system"]

    if not non_system:
        return result

    total_chars = sum(len(m["content"]) for _, m in non_system)
    budget_chars = max_tokens * 4

    system_chars = sum(len(result[i]["content"]) for i in system_indices)
    available_chars = budget_chars - system_chars

    if total_chars <= available_chars:
        return result

    available_chars = max(available_chars, 0)

    first_budget = int(available_chars * head_ratio)
    last_budget = int(available_chars * last_response_ratio)
    middle_count = max(len(non_system) - 2, 0)
    middle_total = max(available_chars - first_budget - last_budget, 0)
    per_turn_cap = int(available_chars * per_turn_ratio)
    per_middle = min(middle_total // max(middle_count, 1), per_turn_cap) if middle_count > 0 else 0

    for idx_in_list, (orig_idx, _msg) in enumerate(non_system):
        content = result[orig_idx]["content"]
        if idx_in_list == 0:
            budget = first_budget
        elif idx_in_list == len(non_system) - 1:
            budget = last_budget
        else:
            budget = per_middle

        if len(content) > budget:
            result[orig_idx]["content"] = _truncate_text(content, budget)

    return result
