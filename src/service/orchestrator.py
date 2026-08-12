"""Pipeline orchestration extracted from e2e_smoke (spec §7).

Dependency-injected so tests can run with fakes (no real LLM/embedding).
Emits progress events keyed by pipeline stage.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import asyncio
import inspect
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


# ── 失败轨迹压缩（service 层；决策 2 架构护栏）────────────────────────────
# module0 绝不 import module1；轨迹 load+slice+summarize 三件套在 service 层做，
# 压成一段紧凑文本后经 compile(failure_evidence=...) 依赖注入喂给 compiler。
# service 本就同时依赖 module0/1，import 方向合规（护栏是 module0 不依赖 module1）。
#
# 长度上限（避免撞 compiler max_tokens=4000 / 别让单条长轨迹吃满预算）：
#   _MAX_EVIDENCE_SLICES：最多取前 N 个 slice（一次失败通常集中在前若干语义段）。
#   _MAX_EVIDENCE_CHARS：紧凑文本硬字符上限，复用 summarizer 的"截断+省略号"思路兜底。
# 选值理由见 PR 报告；summarizer 已逐步截断 assistant/args/result，此处是二次总量封顶。
_MAX_EVIDENCE_SLICES = 3
_MAX_EVIDENCE_CHARS = 4000


def _compress_failure_trajectory(
    failure_trajectory: Any,
    load_trajectories_fn: Callable,
) -> str | None:
    """把失败轨迹压成紧凑文本供 Call 2 逐子问题认领证据步；无有效轨迹 → None。

    铁律（决策 铁律1，物理隔离）：本函数**只**读失败轨迹并返回文本，供 compile 压缩，
    **绝不**把读到的轨迹写进任何正例存储（problem_store / trajectory_store / module1
    index）。失败轨迹是"错误现场"，一旦当正例召回选进 SFT 即灾难。调用方须保证本函数
    的返回值只流向 deps.compiler.compile(failure_evidence=...)，不流向 ingest 写路径。

    `failure_trajectory` 取自 manifest 结构化行（轨迹传路径引用，不内联）。仅当它是
    非空字符串路径时才消费；null / 非字符串一律视为无轨迹返回 None（向后兼容）。
    """
    # 契约：轨迹传路径引用。非字符串（null/dict 等）→ 无轨迹。
    if not isinstance(failure_trajectory, str) or not failure_trajectory:
        return None

    # 决策 2：slice/summarize 三件套在 module1；service 层 import 合规（lazy，
    # 与本文件 module3 的 lazy import 同风格，避免顶层硬依赖）。
    from module1.slicer import slice_trajectory
    from module1.summarizer import summarize_slice

    trajectories = load_trajectories_fn(failure_trajectory)
    if not trajectories:
        return None

    blocks: list[str] = []
    slices_used = 0
    for traj in trajectories:
        if slices_used >= _MAX_EVIDENCE_SLICES:
            break
        for sl in slice_trajectory(traj):
            if slices_used >= _MAX_EVIDENCE_SLICES:
                break
            body = summarize_slice(sl)
            if not body:
                continue
            blocks.append(f"[trajectory_id: {traj.id}]\n{body}")
            slices_used += 1

    if not blocks:
        return None

    text = "\n\n".join(blocks)
    if len(text) > _MAX_EVIDENCE_CHARS:
        text = text[:_MAX_EVIDENCE_CHARS] + "\n...(截断)"
    return text


async def _offload(fn, *args, **kwargs):
    """把同步阻塞段挪出事件循环，且取消时不留孤儿线程。

    为什么不裸 `await asyncio.to_thread(...)`：run_ingest 由 app 层的 _run_lock
    串行化，用的是同一个 deps.pipeline 实例，写的是同一套 store。裸 to_thread 在
    取消时会让协程立刻解锁返回，而工作线程仍在改 pipeline._store / 写库 —— 用户
    紧接着发起的下一次入库就会与这个孤儿线程并发写，索引被写坏。app.py 的 finally
    还会 unlink 上传的临时文件，孤儿线程可能正在读它。

    线程无法从外部中断，所以取消时只能 shield 住等它自己跑完再把 CancelledError
    传上去：调用方失去的只是"立刻返回"，换来的是锁释放时后台确实已经收工。
    与 module1/pipeline.py:150 的 _build_index 卸载同款处理。
    """
    task = asyncio.ensure_future(asyncio.to_thread(fn, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # 重复取消也不能放弃这个 task —— 循环 shield 直到线程真正结束。
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        raise


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
        # PR-2: 结构化行取 question 编译；failure_trajectory 非 None 时压缩成
        # failure_evidence 文本喂 Call 2（据现场认领证据步 / 定 label / 覆盖 summary）。
        # 纯文本行 failure_trajectory 恒 None → failure_evidence=None，行为逐字不变。
        question, failure_trajectory = _parse_manifest_line(line)
        emit({"stage": "module0", "status": "running",
              "msg": f"编译第 {done_n}/{total} 条: {question[:30]}",
              "index": done_n, "total": total})
        # 物理隔离（决策 铁律1）：失败轨迹只用于压缩喂 compile，绝不进 module1 index /
        # trajectory_store —— 本函数只返回文本，不落任何正例存储。
        failure_evidence = _compress_failure_trajectory(
            failure_trajectory, deps.load_trajectories_fn)
        spec_obj = await deps.compiler.compile(question, failure_evidence=failure_evidence)
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
        _n_unique = len({(c.trajectory_id, c.slice_index) for c in scored})
        selection_config = selection_config or SelectionConfig(
            n=min(10, max(1, _n_unique)), min_per_problem=1
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
        _n_unique = len({(c.trajectory_id, c.slice_index) for c in scored})
        selection_config = selection_config or SelectionConfig(
            n=min(10, max(1, _n_unique)), min_per_problem=1)
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
    # None = 本次没传轨迹文件；0 = 传了但一条都没解析出来（两者必须可区分）
    traj_ingested: int | None = None

    if trajectory_path is not None:
        traj_ingested = 0
        emit({"stage": "ingest_traj", "status": "running", "msg": "准备入库..."})

        def on_traj(traj, src):
            # 在工作线程里被回调：_offload 只起一个线程跑整段，on_traj 不存在并发
            # 调用，nonlocal 累加无竞态。store 侧跨线程安全见 stores.py:95-99
            # （check_same_thread=False + WAL + autocommit）。
            nonlocal traj_ingested
            traj_ingested += 1
            if deps.trajectory_store is None:
                return
            steps = [_step_to_dict(s) for s in traj.steps]
            # 原封不动的 OpenAI messages 一并落库，供前端 Raw JSON 视图 ——
            # steps 扁平化丢掉了 tool_call_id / tool_calls[].id，只能从这里找回。
            # getattr 兜底鸭子类型的轨迹对象（测试 fake、下游自定义实现）。
            raw_messages = getattr(traj, "raw_messages", None)
            raw_json = (json.dumps(raw_messages, ensure_ascii=False)
                        if raw_messages else None)
            try:
                deps.trajectory_store.upsert(
                    traj.id, steps, source_path=src, raw_json=raw_json)
            except TypeError:
                # 老 TrajectoryStore 的 upsert 不接受 raw_json —— 降级为不存 raw，
                # 而不是让整个入库失败。Raw JSON 是增强，不该成为必需契约。
                deps.trajectory_store.upsert(traj.id, steps, source_path=src)

        _PHASE_MSG = {"slicing": "切片", "embedding": "向量化", "writing": "写入索引"}

        def on_prog(phase, done, total):
            # 在工作线程里被回调：append_event 本身线程安全（list.append + dict
            # 查都受 GIL 提供保护），只是 notify 机制走不到（runstore.py:82 的
            # RuntimeError 分支），靠 subscribe() 的 0.5s 轮询兜底。实测送达延迟
            # ~250ms，相对入库的数十秒~数分钟量级完全可接受。
            label = _PHASE_MSG.get(phase, phase)
            # 向量化的 0/N 是"这批刚开始算"，写成 "0/15" 会被读成没动静 ——
            # 这段是整条链路最慢的，且要等整个 chunk 算完才有下一条事件。
            if phase == "embedding" and done == 0:
                msg = f"{label} 0/{total}（正在计算首批…）"
            else:
                msg = f"{label} {done}/{total}"
            emit({"stage": "ingest_traj", "status": "running", "msg": msg,
                  "index": done, "total": total, "phase": phase})

        # 切片+签名+embed 是整条链路最重的同步段，直接跑在事件循环上会让 SSE 心跳、
        # 计时器、/stats 和「停止」按钮全部失联（实测入库期间心跳 0 次）。
        #
        # on_progress 是后加的可选能力：注入式 pipeline（测试 fake、下游自定义
        # 实现）可能仍是老签名，无脑传会 TypeError 炸掉整个入库。探测形参后再决定
        # 传不传 —— 进度显示是增强，不该成为新的必需契约。
        kwargs = {"on_trajectory": on_traj}
        try:
            params = inspect.signature(deps.pipeline.ingest_trajectories).parameters
        except (TypeError, ValueError):     # 内建/C 实现取不到签名
            params = {}
        if "on_progress" in params:
            kwargs["on_progress"] = on_prog
        await _offload(
            deps.pipeline.ingest_trajectories, [trajectory_path], **kwargs)

    if manifest_lines:
        for i, raw in enumerate(manifest_lines):
            line = raw.strip()
            if not line:
                continue
            try:
                # PR-2: 结构化行取 question；failure_trajectory 非 None 时压缩喂 Call 2。
                # 纯文本行 failure_trajectory 恒 None，dedup/compile/入库口径逐字不变。
                question, failure_trajectory = _parse_manifest_line(line)
                emb = await _offload(deps.embedder.embed, question)
                hit = deps.problem_store.nearest(emb)
                if hit is not None and hit[1] >= tau_q:
                    skipped += 1
                    emit({"stage": "ingest_manifest", "status": "running",
                          "msg": f"跳过重复: {question[:30]}"})
                    continue
                # 物理隔离（决策 铁律1）：失败轨迹**只**读来压缩喂 compile，绝不进
                # trajectory_store / problem_store。下方 problem_store.add 写的是
                # question + 正例 spec，与失败轨迹在两条互不共用的路径上。
                failure_evidence = _compress_failure_trajectory(
                    failure_trajectory, deps.load_trajectories_fn)
                spec_obj = await deps.compiler.compile(
                    question, failure_evidence=failure_evidence)
                spec = _spec_to_dict(spec_obj, id_prefix=f"{problem_id_of(question)}.")
                deps.problem_store.add(question, emb, spec)
                added += 1
                emit({"stage": "ingest_manifest", "status": "running",
                      "msg": f"入库: {question[:30]}"})
            except Exception as exc:  # noqa: BLE001 — D4 逐行隔离，单行失败不中断
                failed += 1
                emit({"stage": "ingest_manifest", "status": "running",
                      "msg": f"第{i + 1}行失败: {exc}"})

    traj_msg = "" if traj_ingested is None else f"处理轨迹 {traj_ingested} 条, "
    emit({"stage": "done", "status": "ok",
          "msg": f"{traj_msg}入库 {added} 问题, 跳过 {skipped}, 失败 {failed}"})
    view = {"mode": "ingest",
            "summary": {"added": added, "skipped_dup": skipped, "failed": failed,
                        "traj_ingested": traj_ingested}}
    return view, {}
