"""Pipeline orchestration extracted from e2e_smoke (spec §7).

Dependency-injected so tests can run with fakes (no real LLM/embedding).
Emits progress events keyed by pipeline stage.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Callable

from service.viewmodel import build_inspector_view, build_trajectory_index

__all__ = ["PipelineDeps", "run_pipeline"]


@dataclass
class PipelineDeps:
    """Injected collaborators. Production wiring builds real ones; tests fake."""
    compiler: Any                 # has async compile(raw_input) -> ProblemSpec
    pipeline: Any                 # has async run_scored(*, trajectory_paths, problem_specs)
    select_fn: Callable          # select_final_dataset signature
    load_trajectories_fn: Callable  # load_trajectories(path) -> list[Trajectory]


def _spec_to_dict(spec: Any, *, id_prefix: str) -> dict:
    """Serialize ProblemSpec to contract §1 dict, prefixing sub_problem ids.

    id_prefix keeps ids unique across manifest lines (spec §12 open point 3).
    """
    return {
        "raw_input": spec.raw_input,
        "domain": spec.domain,
        "sub_problems": [
            {
                "id": f"{id_prefix}{sp.id}",
                "origin": sp.origin,
                "parent_id": sp.parent_id,
                "raw_text": sp.raw_text,
                "failure_summary": sp.failure_summary,
                "target_capability": sp.target_capability,
                "trajectory_signal": sp.trajectory_signal,
                "hyde_positive": sp.hyde_positive,
                "keywords": sp.keywords,
                "structured_filters": {
                    "languages": sp.structured_filters.languages,
                    "tools_used": sp.structured_filters.tools_used,
                    "has_verification_step": sp.structured_filters.has_verification_step,
                },
                "confidence": sp.confidence,
                "route": sp.route,
            }
            for sp in spec.sub_problems
        ],
    }


async def run_pipeline(
    *,
    manifest_lines: list[str],
    trajectory_path: str | pathlib.Path,
    deps: PipelineDeps,
    emit: Callable[[dict], None],
    run_id: str = "run",
    selection_config: Any = None,
    general_config: Any = None,
) -> tuple[dict, dict]:
    """Run module0→1→2→3, emit progress, return (InspectorView, trajectory_index).

    Returns view dict (spec §5.2) and trajectory index (id → {steps}).
    """
    trajectory_path = pathlib.Path(trajectory_path)

    # ── Module 0: compile each manifest line ────────────────────────────
    emit({"stage": "module0", "status": "running", "msg": "编译问题清单..."})
    specs: list[dict] = []
    for i, line in enumerate(manifest_lines):
        line = line.strip()
        if not line:
            continue
        emit({"stage": "module0", "status": "running",
              "msg": f"编译第 {i + 1} 条: {line[:30]}"})
        spec_obj = await deps.compiler.compile(line)
        specs.append(_spec_to_dict(spec_obj, id_prefix=f"L{i + 1}."))

    all_sub_ids = [sp["id"] for s in specs for sp in s["sub_problems"]]

    # ── Module 1+2: recall + judge + soft-score ─────────────────────────
    emit({"stage": "module1", "status": "running", "msg": "切片+召回+精判..."})
    scored = await deps.pipeline.run_scored(
        trajectory_paths=[trajectory_path], problem_specs=specs
    )
    emit({"stage": "module2", "status": "running",
          "msg": f"软加分完成: {len(scored)} 个候选"})

    # ── Module 3: dedup + select ────────────────────────────────────────
    emit({"stage": "module3", "status": "running", "msg": "去重+集合优选..."})
    if selection_config is None or general_config is None:
        from module3.selection import SelectionConfig
        from module3.compose import GeneralDataConfig
        selection_config = selection_config or SelectionConfig(
            n=min(10, max(1, len(scored))), min_per_problem=1
        )
        general_config = general_config or GeneralDataConfig(ratio=0.3, source_path=None)

    select_result = deps.select_fn(
        scored,
        sub_problem_ids=all_sub_ids,
        selection=selection_config,
        general=general_config,
    )

    # ── Aggregate view + trajectory index ───────────────────────────────
    merged_spec = {
        "raw_input": "\n".join(manifest_lines),
        "domain": "agentic_swe",
        "sub_problems": [sp for s in specs for sp in s["sub_problems"]],
    }
    view = build_inspector_view(
        run_id=run_id, spec=merged_spec, scored=scored, select_result=select_result
    )
    trajectories = build_trajectory_index(
        deps.load_trajectories_fn(trajectory_path)
    )
    return view, trajectories
