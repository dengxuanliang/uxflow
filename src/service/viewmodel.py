"""Aggregate module0/2/3 outputs into the frontend InspectorView (spec §5.2, §6)."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

from typing import Any

from service.colors import assign_colors

__all__ = ["build_inspector_view", "build_trajectory_index"]


def _all_labels(spec: dict) -> list[str]:
    labels: list[str] = []
    for sp in spec.get("sub_problems", []):
        labels.extend(sp.get("target_capability", []))
    return labels


def build_inspector_view(
    *,
    run_id: str,
    spec: dict,
    scored: list[Any],
    select_result: dict,
) -> dict:
    """Join spec (module0) ⋈ scored (module2) ⋈ select_result (module3).

    Trajectory full text is NOT inlined here; the frontend lazy-loads it via
    /trajectory/{id}. See build_trajectory_index for that side channel.

    `spec` must be the dict-serialized ProblemSpec (contract §1 shape), not the
    module0 ProblemSpec dataclass — sub_problems are accessed as dicts here.
    """
    colors = assign_colors(_all_labels(spec))

    # (traj_id, slice_index) of module3-selected candidates（MergedCandidate 多归属，
    # 无单一 sub_problem_id；同一 slice 在其覆盖的所有子问题卡片下都算已入选）
    selected_keys = {
        (c.trajectory_id, c.slice_index)
        for c in select_result.get("targeted", [])
    }

    # Index scored candidates by sub_problem_id for quick lookup
    by_problem: dict[str, list] = {}
    for c in scored:
        by_problem.setdefault(c.sub_problem_id, []).append(c)

    problems = []
    for sp in spec.get("sub_problems", []):
        sp_id = sp.get("id")
        if sp_id is None:
            continue  # defensive: skip malformed sub_problem lacking an id (issue 11)
        target_caps = sp.get("target_capability", [])
        cands = by_problem.get(sp_id, [])

        # Deduplicate hits per (trajectory_id, slice_index) for this sub_problem,
        # keeping only genuine judge matches with a non-empty span set. The judge
        # verdict is per-slice-per-sub_problem (not per individual capability), so
        # every capability of this sub_problem shares the SAME hit set — we build
        # it once (issue 5) instead of once-per-label, and drop misses (issue 7).
        seen: dict[tuple, dict] = {}
        for c in cands:
            if not getattr(c, "judge_match", False):
                continue
            spans = list(c.loss_mask_spans or [])
            if not spans:
                continue
            key = (c.trajectory_id, c.slice_index)
            if key not in seen:
                seen[key] = {
                    "trajectory_id": c.trajectory_id,
                    "slice_index": c.slice_index,
                    "relevance_score": c.relevance_score,
                    "judge_confidence": c.judge_confidence,
                    "selected": (c.trajectory_id, c.slice_index) in selected_keys,
                    "loss_mask_spans": spans,
                    # 可回溯字段：judge 定位的决定性步 + 命中的 rubric 判据，供人审跳到证据。
                    # 老 ScoredCandidate（无此二字段）用 getattr 降级，向后兼容。
                    "evidence_step": getattr(c, "evidence_step", None),
                    "criteria_hit": list(getattr(c, "criteria_hit", []) or []),
                }
        sub_hits = list(seen.values())

        capabilities = [
            {
                "label": label,
                "parent": None,  # taxonomy parent enrichment reserved (spec §10)
                "color": colors.get(label, "#888888"),
                "hit_count": len(sub_hits),
                "hit_trajectories": sub_hits,
            }
            for label in target_caps
        ]

        problems.append({
            "id": sp_id,
            "failure_summary": sp.get("failure_summary", ""),
            "confidence": sp.get("confidence", 0.0),
            "capabilities": capabilities,
            # Full module0 compilation output, surfaced for the "编译详情" modal.
            # All fields already ride along in the serialized spec (orchestrator
            # _spec_to_dict) — we just expose them instead of dropping them.
            "compile": {
                "raw_text": sp.get("raw_text", ""),
                "origin": sp.get("origin", ""),
                "target_capability": target_caps,
                "trajectory_signal": sp.get("trajectory_signal", ""),
                "keywords": sp.get("keywords", []),
                "hyde_positive": sp.get("hyde_positive", []),
                "structured_filters": sp.get("structured_filters", {}),
            },
        })

    return {
        "run_id": run_id,
        "problems": problems,
        "manifest": select_result.get("manifest", {}),
    }


def _step_to_dict(step: Any) -> dict:
    return {
        "index": step.index,
        "role": step.role,
        "content": step.content,
        "tool_call_name": step.tool_call_name,
        "tool_call_args": step.tool_call_args,
        "tool_result": step.tool_result,
    }


def build_trajectory_index(trajectories: list[Any]) -> dict[str, dict]:
    """Index Trajectory objects by id for lazy /trajectory/{id} serving."""
    return {
        t.id: {
            "trajectory_id": t.id,
            "steps": [_step_to_dict(s) for s in t.steps],
            # 原封不动的 OpenAI messages，供前端 Raw JSON 视图。steps 是扁平化
            # 产物，丢了 tool_call_id 等关联信息。getattr 兜底：鸭子类型的调用方
            # （测试 fake、下游自定义轨迹对象）未必有这个属性。
            "raw": getattr(t, "raw_messages", None) or None,
        }
        for t in trajectories
    }
