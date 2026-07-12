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
    """
    colors = assign_colors(_all_labels(spec))

    # (traj_id, slice_index, sub_problem_id) of module3-selected candidates
    selected_keys = {
        (c.trajectory_id, c.slice_index, c.sub_problem_id)
        for c in select_result.get("targeted", [])
    }

    # Index scored candidates by sub_problem_id for quick lookup
    by_problem: dict[str, list] = {}
    for c in scored:
        by_problem.setdefault(c.sub_problem_id, []).append(c)

    problems = []
    for sp in spec.get("sub_problems", []):
        sp_id = sp["id"]
        target_caps = sp.get("target_capability", [])
        cands = by_problem.get(sp_id, [])

        capabilities = []
        for label in target_caps:
            hits = []
            for c in cands:
                if label not in (c.capability or []):
                    continue
                hits.append({
                    "trajectory_id": c.trajectory_id,
                    "slice_index": c.slice_index,
                    "relevance_score": c.relevance_score,
                    "judge_confidence": c.judge_confidence,
                    "selected": (c.trajectory_id, c.slice_index, sp_id) in selected_keys,
                    "loss_mask_spans": list(c.loss_mask_spans or []),
                })
            capabilities.append({
                "label": label,
                "parent": None,  # taxonomy parent enrichment reserved (spec §10)
                "color": colors.get(label, "#888888"),
                "hit_count": len(hits),
                "hit_trajectories": hits,
            })

        problems.append({
            "id": sp_id,
            "failure_summary": sp.get("failure_summary", ""),
            "confidence": sp.get("confidence", 0.0),
            "capabilities": capabilities,
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
        }
        for t in trajectories
    }
