"""Pipeline orchestration extracted from e2e_smoke (spec §7).

Dependency-injected so tests can run with fakes (no real LLM/embedding).
Emits progress events keyed by pipeline stage.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Callable

from service.viewmodel import build_inspector_view, build_trajectory_index

__all__ = ["PipelineDeps", "run_pipeline", "run_search", "run_ingest"]


def _parse_manifest_line(line: str) -> tuple[str, Any]:
    """Parse one manifest line into (question, failure_trajectory).

    Upgrade path (PR-1): a line that is a JSON object carrying a "question" key
    whose value is a non-empty string is read structurally — question drives
    compile, failure_trajectory (may be null) is parsed out but NOT consumed
    here (PR-2 wires it into compile()). Anything else — non-JSON, JSON that is
    not a dict-with-"question", or a "question" that is not a usable string —
    falls back to treating the WHOLE raw line as the question (legacy plain-text
    behavior, byte-for-byte unchanged). The string guard keeps a malformed
    structured line (e.g. {"question": 123}) from injecting a non-str into
    compile()/logging, which run_pipeline's compile loop would not isolate.

    `line` is expected already-stripped by callers.
    """
    if line[:1] in ("{", "["):
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return line, None
        if isinstance(obj, dict) and isinstance(obj.get("question"), str) and obj["question"]:
            return obj["question"], obj.get("failure_trajectory")
    return line, None


@dataclass
class PipelineDeps:
    """Injected collaborators. Production wiring builds real ones; tests fake."""
    compiler: Any                 # has async compile(raw_input) -> ProblemSpec
    pipeline: Any                 # has async run_scored(*, trajectory_paths, problem_specs)
    select_fn: Callable          # select_final_dataset signature
    load_trajectories_fn: Callable  # load_trajectories(path) -> list[Trajectory]
    problem_store: Any = None     # read/write problem manifest (search/ingest)
    trajectory_store: Any = None  # persist trajectory steps (ingest)
    judge_cache: Any = None       # pairwise judge verdict cache (search)
    embedder: Any = None          # embed(text) -> list[float] for dedup


def _dc_to_dict(obj: Any) -> dict | None:
    """Serialize an optional dataclass field (rubric / failure_evidence) to a
    plain dict, or None when absent. asdict() handles the nested structure;
    a plain dict passes through unchanged (defensive, in case already serialized).
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    return asdict(obj)


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
                # 可选增强字段（契约 §1.5/§1.6）：None → null；非 None → 嵌套 dict。
                # getattr 默认 None 保证旧 spec 对象/测试 fake 无此属性时不炸。
                "rubric": _dc_to_dict(getattr(sp, "rubric", None)),
                "failure_evidence": _dc_to_dict(getattr(sp, "failure_evidence", None)),
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
    # `total` counts non-blank lines so the frontend can show real "i/N"
    # determinate progress for the compile stage (index/total are additive,
    # back-compatible SSE fields — older consumers just ignore them).
    non_blank = [ln for ln in manifest_lines if ln.strip()]
    total = len(non_blank)
    emit({"stage": "module0", "status": "running", "msg": "编译问题清单...",
          "index": 0, "total": total})
    specs: list[dict] = []
    done_n = 0
    for i, line in enumerate(manifest_lines):
        line = line.strip()
        if not line:
            continue
        done_n += 1
        # PR-1: 结构化行取 question 作编译输入；failure_trajectory 解析出但暂不消费（PR-2 接入）。
        # 纯文本行 question 即整行，行为逐字不变。
        question, _failure_trajectory = _parse_manifest_line(line)
        emit({"stage": "module0", "status": "running",
              "msg": f"编译第 {done_n}/{total} 条: {question[:30]}",
              "index": done_n, "total": total})
        spec_obj = await deps.compiler.compile(question)
        specs.append(_spec_to_dict(spec_obj, id_prefix=f"L{i + 1}."))

    all_sub_problems = [sp for s in specs for sp in s["sub_problems"]]
    all_sub_ids = [sp["id"] for sp in all_sub_problems]

    # ── Module 1+2: recall + judge + soft-score ─────────────────────────
    # index/total start at 0 (indeterminate) until run_scored reports the first
    # sub_problem; on_progress then drives a real "精判 i/N" determinate bar
    # through the judge-heavy phase (the slowest, previously feedback-less step).
    emit({"stage": "module1", "status": "running", "msg": "切片+召回+精判...",
          "index": 0, "total": len(all_sub_problems)})
    scored = await deps.pipeline.run_scored(
        trajectory_paths=[trajectory_path],
        problem_specs=specs,
        on_progress=lambda done, total: emit({
            "stage": "module1", "status": "running",
            "msg": f"精判 {done}/{total} 个子问题",
            "index": done, "total": total,
        }),
    )
    emit({"stage": "module2", "status": "running",
          "msg": f"软加分完成: {len(scored)} 个候选"})

    # ── Module 3: dedup + select ────────────────────────────────────────
    emit({"stage": "module3", "status": "running", "msg": "去重+集合优选..."})
    if selection_config is None or general_config is None:
        # Lazy: keep the orchestrator free of a hard top-level module3 dependency
        # (the DI seam decouples from concrete pipeline modules). Only the
        # default-config convenience branch needs these.
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
    # viewmodel wants one flat spec; run_scored wants per-line specs — hence two shapes
    merged_spec = {
        "raw_input": "\n".join(manifest_lines),
        # domain is view-cosmetic (build_inspector_view never reads it); fixed intentionally
        "domain": "agentic_swe",
        "sub_problems": all_sub_problems,
    }
    view = build_inspector_view(
        run_id=run_id, spec=merged_spec, scored=scored, select_result=select_result
    )
    trajectories = build_trajectory_index(
        deps.load_trajectories_fn(trajectory_path)
    )
    return view, trajectories


async def run_search(
    question,
    *,
    deps: PipelineDeps,
    emit: Callable[[dict], None],
    run_id: str = "search",
    tau_q: float = 0.90,
    selection_config: Any = None,
    general_config: Any = None,
) -> tuple[dict, dict]:
    """Read path (spec §7): dedup-aware compile → search → select → view.

    只读铁律：全程绝不调 deps.problem_store.add — 预览不写库 (spec 决策1)。
    Detail text is served lazily via TrajectoryStore, so the trajectory index
    returned here is intentionally empty (batch4 app layer resolves it).
    """
    from service.dedup import ProblemCompiler, problem_id_of

    emit({"stage": "search0", "status": "running", "msg": "分析问题...",
          "index": 0, "total": 1})
    pc = ProblemCompiler(
        deps.compiler, deps.problem_store, deps.embedder,
        tau_q=tau_q,
        serialize_fn=lambda so, q: _spec_to_dict(so, id_prefix=f"{problem_id_of(q)}."),
    )
    spec, dedup = await pc.get_or_compile(question)
    if dedup is not None:
        emit({"stage": "search0", "status": "running",
              "msg": f"命中已有问题 (相似度 {dedup['similarity']:.2f})"})
    spec_lines = [spec]
    all_sub = [sp for s in spec_lines for sp in s["sub_problems"]]
    all_sub_ids = [sp["id"] for sp in all_sub]

    scored = await deps.pipeline.search(
        problem_specs=spec_lines, judge_cache=deps.judge_cache,
        on_progress=lambda done, total: emit({
            "stage": "search1", "status": "running",
            "msg": f"检索 {done}/{total}", "index": done, "total": total}))

    if selection_config is None or general_config is None:
        from module3.selection import SelectionConfig
        from module3.compose import GeneralDataConfig
        selection_config = selection_config or SelectionConfig(
            n=min(10, max(1, len(scored))), min_per_problem=1)
        general_config = general_config or GeneralDataConfig(ratio=0.3, source_path=None)
    select_result = deps.select_fn(
        scored, sub_problem_ids=all_sub_ids,
        selection=selection_config, general=general_config)

    merged_spec = {"raw_input": question, "domain": "agentic_swe",
                   "sub_problems": all_sub}
    view = build_inspector_view(
        run_id=run_id, spec=merged_spec, scored=scored, select_result=select_result)
    view["mode"] = "search"
    view["dedup"] = dedup
    return view, {}     # 详情走 TrajectoryStore（不内联），批4 的 app 层处理


async def run_ingest(
    *,
    manifest_lines: list[str] | None = None,
    trajectory_path: str | pathlib.Path | None = None,
    deps: PipelineDeps,
    emit: Callable[[dict], None],
    run_id: str = "ingest",
    tau_q: float = 0.90,
) -> tuple[dict, dict]:
    """Write path (spec §7, D4 逐行隔离): persist trajectories + dedup-gated manifest.

    Manifest lines are ingested one-by-one; a single line's failure is isolated
    (counted, emitted) without aborting the rest of the batch.
    """
    from service.dedup import problem_id_of
    from service.viewmodel import _step_to_dict

    added = skipped = failed = 0

    if trajectory_path is not None:
        emit({"stage": "ingest_traj", "status": "running", "msg": "切片+签名+写库..."})

        def on_traj(traj, src):
            if deps.trajectory_store is not None:
                deps.trajectory_store.upsert(
                    traj.id, [_step_to_dict(s) for s in traj.steps], source_path=src)

        deps.pipeline.ingest_trajectories([trajectory_path], on_trajectory=on_traj)

    if manifest_lines:
        for i, raw in enumerate(manifest_lines):
            line = raw.strip()
            if not line:
                continue
            try:
                # PR-1: 结构化行取 question；failure_trajectory 暂不消费（PR-2 接入）。
                # 纯文本行 question 即整行，dedup/compile/入库口径逐字不变。
                question, _failure_trajectory = _parse_manifest_line(line)
                emb = deps.embedder.embed(question)
                hit = deps.problem_store.nearest(emb)
                if hit is not None and hit[1] >= tau_q:
                    skipped += 1
                    emit({"stage": "ingest_manifest", "status": "running",
                          "msg": f"跳过重复: {question[:30]}"})
                    continue
                spec_obj = await deps.compiler.compile(question)
                spec = _spec_to_dict(spec_obj, id_prefix=f"{problem_id_of(question)}.")
                deps.problem_store.add(question, emb, spec)
                added += 1
                emit({"stage": "ingest_manifest", "status": "running",
                      "msg": f"入库: {question[:30]}"})
            except Exception as exc:  # noqa: BLE001 — D4 逐行隔离，单行失败不中断
                failed += 1
                emit({"stage": "ingest_manifest", "status": "running",
                      "msg": f"第{i + 1}行失败: {exc}"})

    emit({"stage": "done", "status": "ok",
          "msg": f"入库 {added} 问题, 跳过 {skipped}, 失败 {failed}"})
    view = {"mode": "ingest",
            "summary": {"added": added, "skipped_dup": skipped, "failed": failed}}
    return view, {}
