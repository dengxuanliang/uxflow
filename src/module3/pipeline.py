"""Module 3 orchestration: dedup -> select -> compose."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

from typing import Any

from module3.compose import GeneralDataConfig, compose_dataset
from module3.dedup import deduplicate
from module3.selection import SelectionConfig, select_set

__all__ = ["select_final_dataset"]


def select_final_dataset(
    candidates: list[Any],
    *,
    sub_problem_ids: list[str],
    selection: SelectionConfig,
    general: GeneralDataConfig,
    cosine_threshold: float = 0.95,
    minhash_threshold: float = 0.9,
) -> dict:
    """Run module3's full final dataset selection flow."""
    trainable = [
        c for c in candidates
        if getattr(c, "judge_match", True) and getattr(c, "loss_mask_spans", [])
    ]
    deduped = deduplicate(
        trainable,
        cosine_threshold=cosine_threshold,
        minhash_threshold=minhash_threshold,
    )
    selected = select_set(deduped, sub_problem_ids=sub_problem_ids, config=selection)
    return compose_dataset(selected, general_config=general)
