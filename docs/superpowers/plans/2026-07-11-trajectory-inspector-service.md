# 轨迹 Inspector 服务与前端 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把现有 module0/1/2/3 流水线包成一个"上传原始抱怨清单+候选回流jsonl → 异步运行 → 三栏可视化检分命中轨迹"的产品，前后端以 InspectorView + SSE 进度两个契约解耦。

**Architecture:** 新增 `src/service/`（编排层 orchestrator + 聚合层 viewmodel + RunStore seam + FastAPI 服务）+ 原生单页前端 `src/service/web/`。现有 4 个模块零改动；orchestrator 把 `e2e_smoke.py` 的 main() 逻辑抽成可 emit 进度的函数；viewmodel 把末端 ScoredCandidate ⋈ Trajectory join 成前端视图。

**Tech Stack:** Python 3.11, asyncio, FastAPI + uvicorn + python-multipart（新 optional 依赖组 `service`），原生 HTML/JS/CSS（无构建），pytest + pytest-asyncio。

**上游 spec:** `docs/superpowers/specs/2026-07-11-trajectory-inspector-service-design.md`

---

## 前置说明（实现者必读）

- **测试运行**：`.venv/bin/pytest tests/service/ -v`（`asyncio_mode = auto` 已在 pyproject 配好，async 测试无需 `@pytest.mark.asyncio`）。
- **纯离线测试**：本计划所有单元/集成测试**不依赖真实 LLM/embedding**。orchestrator 通过依赖注入接收 compiler/pipeline 工厂，测试用 Fake 替身。绝不在测试里连 LiteLLM 或加载 Qwen。
- **现有关键结构**（已核对源码，实现者可直接依赖）：
  - `module2.models.ScoredCandidate`：`trajectory_id, slice_index, trajectory_path, sub_problem_id, capability(list[str]), relevance_score, judge_confidence, loss_mask_spans(list[dict]), judge_match, embedding, bm25_tokens`
  - `module1.models.Trajectory`：`id, steps(list[Step]), raw_messages`；`Step`：`index, role, content, tool_call_name, tool_call_args, tool_result`
  - `module1.loader.load_trajectories(path) -> list[Trajectory]`
  - `module3.pipeline.select_final_dataset(candidates, *, sub_problem_ids, selection, general) -> {"targeted","general","manifest"}`；`manifest` = `{targeted_count, general_count, general_ratio}`
  - `module3.selection.SelectionConfig(n, min_per_problem, ...)`、`module3.compose.GeneralDataConfig(ratio, source_path)`
- **包注册**：新增模块后需把 `src/service` 加入 `pyproject.toml` 的 `[tool.hatch.build.targets.wheel].packages`。
- **提交粒度**：每个 Task 末尾 commit 一次。分支：实现前确认在专用分支（非 main）。

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `src/service/__init__.py` | 导出公共 API（`build_inspector_view`, `MemoryRunStore`, `run_pipeline`, `create_app`）|
| `src/service/colors.py` | taxonomy label → 稳定色的纯函数分配 |
| `src/service/viewmodel.py` | 聚合层：ScoredCandidate ⋈ spec ⋈ select_result → InspectorView dict |
| `src/service/runstore.py` | `RunStore` 协议 + `MemoryRunStore`（内存态 + asyncio 事件订阅）|
| `src/service/orchestrator.py` | `run_pipeline()`：抽自 e2e_smoke，emit 进度，依赖注入 compiler/pipeline 工厂 |
| `src/service/app.py` | FastAPI：POST /runs、SSE /runs/{id}/events、GET /runs/{id}/view、GET /runs/{id}/trajectory/{tid}、静态前端 |
| `src/service/web/index.html` | 三栏布局单页 |
| `src/service/web/style.css` | 继承 mockup 配色 |
| `src/service/web/app.js` | 上传→运行→SSE进度→三栏渲染→聚焦高亮 |
| `tests/service/test_colors.py` | 配色稳定性测试 |
| `tests/service/test_viewmodel.py` | 聚合层测试 |
| `tests/service/test_runstore.py` | RunStore 测试 |
| `tests/service/test_orchestrator.py` | 编排层测试（Fake 替身）|
| `tests/service/test_app.py` | 端点集成测试（TestClient + Fake orchestrator）|

---

## Task 1: 依赖组与包注册

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: 加 service 可选依赖组**

在 `[project.optional-dependencies]` 段（现有 `local-embed`/`onnx-embed`/`dev` 附近）加一行：

```toml
service = ["fastapi>=0.110", "uvicorn>=0.29", "python-multipart>=0.0.9"]
```

- [ ] **Step 2: 注册 wheel 包**

把 `[tool.hatch.build.targets.wheel]` 的 `packages` 列表末尾加 `"src/service"`：

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/llm_gateway", "src/module0", "src/module1", "src/module2", "src/module3", "src/uxflow_embed", "src/module0_5", "src/service"]
```

- [ ] **Step 3: 安装依赖**

Run: `.venv/bin/pip install -e ".[service,dev]"`
Expected: 成功安装 fastapi/uvicorn/python-multipart，无冲突。

- [ ] **Step 4: 建包骨架**

创建 `src/service/__init__.py`，内容：

```python
"""Trajectory Inspector service: orchestration + aggregation + HTTP/SSE."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang
```

创建空目录标记 `tests/service/__init__.py`（空文件）。

- [ ] **Step 5: 验证 import**

Run: `.venv/bin/python -c "import service; print('ok')"`
Expected: 打印 `ok`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/service/__init__.py tests/service/__init__.py
git commit -m "build(service): add service optional-deps group and package skeleton"
```

---

## Task 2: 稳定配色（colors.py）

**Files:**
- Create: `src/service/colors.py`
- Test: `tests/service/test_colors.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/service/test_colors.py`：

```python
# SPDX-License-Identifier: Apache-2.0
from service.colors import assign_colors


def test_same_labels_same_colors_deterministic():
    labels = ["self_verification", "valid_syntax_in_toolcall"]
    a = assign_colors(labels)
    b = assign_colors(labels)
    assert a == b  # 稳定：同输入同输出
    assert set(a.keys()) == set(labels)
    for v in a.values():
        assert v.startswith("#") and len(v) == 7


def test_distinct_labels_get_distinct_colors_until_palette_exhausted():
    labels = [f"cap_{i}" for i in range(3)]
    colors = assign_colors(labels)
    assert len(set(colors.values())) == 3  # 前几个各不相同


def test_palette_wraps_when_more_labels_than_colors():
    labels = [f"cap_{i}" for i in range(50)]
    colors = assign_colors(labels)
    assert len(colors) == 50  # 每个 label 都有色，超出调色板则循环复用
    assert all(c.startswith("#") for c in colors.values())


def test_order_independent_assignment():
    # 同一组 label，顺序不同，各 label 拿到的颜色应一致（按 label 排序分配）
    forward = assign_colors(["b_cap", "a_cap"])
    backward = assign_colors(["a_cap", "b_cap"])
    assert forward == backward
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest tests/service/test_colors.py -v`
Expected: FAIL with "No module named 'service.colors'"

- [ ] **Step 3: 实现 colors.py**

创建 `src/service/colors.py`：

```python
"""Assign stable colors to taxonomy labels (spec §5.3)."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

__all__ = ["PALETTE", "assign_colors"]

# 继承 brainstorm mockup 的配色，扩展到 10 色调色板
PALETTE = [
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
]


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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest tests/service/test_colors.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/service/colors.py tests/service/test_colors.py
git commit -m "feat(service): stable taxonomy label color assignment"
```

---

## Task 3: 聚合层（viewmodel.py）

**Files:**
- Create: `src/service/viewmodel.py`
- Test: `tests/service/test_viewmodel.py`

**契约**（spec §5.2）：`build_inspector_view` 把 spec 的 sub_problems、模块2 的 scored candidates、模块3 的 select_result join 成 InspectorView dict。

- [ ] **Step 1: 写失败测试**

创建 `tests/service/test_viewmodel.py`：

```python
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from service.viewmodel import build_inspector_view


@dataclass
class FakeScored:
    trajectory_id: str
    slice_index: int
    sub_problem_id: str
    capability: list
    relevance_score: float
    judge_confidence: float
    loss_mask_spans: list
    judge_match: bool = True


def _spec():
    # 最小 ProblemSpec dict（contract §1 形态）
    return {
        "raw_input": "x",
        "domain": "agentic_swe",
        "sub_problems": [
            {
                "id": "p1",
                "failure_summary": "写入 py 文件语法错误",
                "target_capability": ["valid_syntax_in_toolcall", "self_verification"],
                "confidence": 0.9,
            }
        ],
    }


def test_view_maps_problems_and_capabilities():
    scored = [
        FakeScored("t1", 0, "p1", ["valid_syntax_in_toolcall"], 0.8, 0.9,
                   [{"start_step": 1, "end_step": 2}]),
    ]
    select_result = {
        "targeted": [scored[0]],
        "manifest": {"targeted_count": 1, "general_count": 0, "general_ratio": 0.3},
    }
    view = build_inspector_view(
        run_id="r1", spec=_spec(), scored=scored, select_result=select_result
    )
    assert view["run_id"] == "r1"
    assert len(view["problems"]) == 1
    p = view["problems"][0]
    assert p["id"] == "p1"
    assert p["failure_summary"].startswith("写入")
    caps = {c["label"]: c for c in p["capabilities"]}
    assert set(caps) == {"valid_syntax_in_toolcall", "self_verification"}
    # 每个能力有稳定色
    assert caps["valid_syntax_in_toolcall"]["color"].startswith("#")
    # 命中轨迹挂到对应能力下
    vs = caps["valid_syntax_in_toolcall"]
    assert vs["hit_count"] == 1
    hit = vs["hit_trajectories"][0]
    assert hit["trajectory_id"] == "t1"
    assert hit["slice_index"] == 0
    assert hit["loss_mask_spans"] == [{"start_step": 1, "end_step": 2}]
    # self_verification 无命中
    assert caps["self_verification"]["hit_count"] == 0


def test_selected_flag_reflects_module3_targeted():
    in_set = FakeScored("t1", 0, "p1", ["valid_syntax_in_toolcall"], 0.8, 0.9,
                        [{"start_step": 0, "end_step": 1}])
    out_set = FakeScored("t2", 0, "p1", ["valid_syntax_in_toolcall"], 0.5, 0.6,
                         [{"start_step": 0, "end_step": 1}])
    select_result = {
        "targeted": [in_set],  # 只有 t1 进选集
        "manifest": {"targeted_count": 1, "general_count": 0, "general_ratio": 0.3},
    }
    view = build_inspector_view(
        run_id="r1", spec=_spec(), scored=[in_set, out_set],
        select_result=select_result,
    )
    hits = view["problems"][0]["capabilities"][0]["hit_trajectories"]
    by_traj = {h["trajectory_id"]: h for h in hits}
    assert by_traj["t1"]["selected"] is True
    assert by_traj["t2"]["selected"] is False


def test_manifest_passthrough():
    select_result = {
        "targeted": [],
        "manifest": {"targeted_count": 0, "general_count": 5, "general_ratio": 0.25},
    }
    view = build_inspector_view(
        run_id="r1", spec=_spec(), scored=[], select_result=select_result
    )
    assert view["manifest"]["general_count"] == 5
    assert view["manifest"]["general_ratio"] == 0.25


def test_candidate_capability_not_in_subproblem_is_ignored():
    # candidate 报了一个不在该 sub_problem target_capability 里的 label → 不挂
    scored = [FakeScored("t1", 0, "p1", ["some_other_label"], 0.8, 0.9,
                         [{"start_step": 0, "end_step": 1}])]
    select_result = {"targeted": [], "manifest": {
        "targeted_count": 0, "general_count": 0, "general_ratio": 0.3}}
    view = build_inspector_view(
        run_id="r1", spec=_spec(), scored=scored, select_result=select_result
    )
    total_hits = sum(c["hit_count"] for c in view["problems"][0]["capabilities"])
    assert total_hits == 0  # 无匹配能力，忽略该命中


def test_build_trajectory_index():
    from service.viewmodel import build_trajectory_index

    @dataclass
    class FakeStep:
        index: int
        role: str
        content: str
        tool_call_name: object = None
        tool_call_args: object = None
        tool_result: object = None

    @dataclass
    class FakeTraj:
        id: str
        steps: list = field(default_factory=list)

    trajs = [FakeTraj("t1", [FakeStep(0, "user", "hi")])]
    idx = build_trajectory_index(trajs)
    assert "t1" in idx
    assert idx["t1"]["trajectory_id"] == "t1"
    assert idx["t1"]["steps"][0]["role"] == "user"
    assert idx["t1"]["steps"][0]["content"] == "hi"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest tests/service/test_viewmodel.py -v`
Expected: FAIL with "No module named 'service.viewmodel'"

- [ ] **Step 3: 实现 viewmodel.py**

创建 `src/service/viewmodel.py`：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest tests/service/test_viewmodel.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/service/viewmodel.py tests/service/test_viewmodel.py
git commit -m "feat(service): InspectorView aggregation layer"
```

---

## Task 4: RunStore seam（runstore.py）

**Files:**
- Create: `src/service/runstore.py`
- Test: `tests/service/test_runstore.py`

**契约**（spec §8）：`RunStore` 协议 + V1 `MemoryRunStore`。事件支持"追加中订阅"（SSE 消费者在 run 进行中订阅，能收到后续事件）。

- [ ] **Step 1: 写失败测试**

创建 `tests/service/test_runstore.py`：

```python
# SPDX-License-Identifier: Apache-2.0
import asyncio

from service.runstore import MemoryRunStore


def test_create_returns_unique_ids():
    store = MemoryRunStore()
    a = store.create()
    b = store.create()
    assert a != b
    assert store.status(a) == "running"


def test_append_and_read_events():
    store = MemoryRunStore()
    rid = store.create()
    store.append_event(rid, {"stage": "module0", "status": "running", "msg": "x"})
    store.append_event(rid, {"stage": "done", "status": "ok"})
    events = list(store.events_snapshot(rid))
    assert len(events) == 2
    assert events[0]["stage"] == "module0"
    assert events[-1]["stage"] == "done"


def test_set_and_get_view():
    store = MemoryRunStore()
    rid = store.create()
    store.set_view(rid, {"run_id": rid, "problems": []}, {"t1": {"steps": []}})
    assert store.get_view(rid)["run_id"] == rid
    assert store.get_trajectory(rid, "t1") == {"steps": []}
    assert store.get_trajectory(rid, "missing") is None


def test_status_transitions():
    store = MemoryRunStore()
    rid = store.create()
    assert store.status(rid) == "running"
    store.mark_done(rid)
    assert store.status(rid) == "done"

    rid2 = store.create()
    store.mark_error(rid2)
    assert store.status(rid2) == "error"


def test_unknown_run_returns_none():
    store = MemoryRunStore()
    assert store.get_view("nope") is None
    assert store.status("nope") is None


async def test_subscribe_receives_events_appended_after_subscribe():
    store = MemoryRunStore()
    rid = store.create()

    received = []

    async def consumer():
        async for ev in store.subscribe(rid):
            received.append(ev)
            if ev.get("stage") == "done":
                break

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)  # let consumer start
    store.append_event(rid, {"stage": "module0", "status": "running"})
    store.append_event(rid, {"stage": "done", "status": "ok"})
    await asyncio.wait_for(task, timeout=1.0)

    assert any(e.get("stage") == "module0" for e in received)
    assert received[-1]["stage"] == "done"


async def test_subscribe_replays_existing_events():
    store = MemoryRunStore()
    rid = store.create()
    store.append_event(rid, {"stage": "module0", "status": "running"})
    store.append_event(rid, {"stage": "done", "status": "ok"})

    received = []
    async for ev in store.subscribe(rid):
        received.append(ev)
        if ev.get("stage") == "done":
            break
    assert len(received) == 2  # 订阅前已有的事件也要重放
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest tests/service/test_runstore.py -v`
Expected: FAIL with "No module named 'service.runstore'"

- [ ] **Step 3: 实现 runstore.py**

创建 `src/service/runstore.py`：

```python
"""Run state storage seam (spec §8).

V1 is in-memory. The RunStore Protocol is the seam: swap in Redis/DB later
without touching app.py. Mirrors module1's SliceStore seam pattern.
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import asyncio
import uuid
from typing import AsyncIterator, Iterable, Protocol

__all__ = ["RunStore", "MemoryRunStore"]


class RunStore(Protocol):
    def create(self) -> str: ...
    def append_event(self, run_id: str, event: dict) -> None: ...
    def events_snapshot(self, run_id: str) -> Iterable[dict]: ...
    def subscribe(self, run_id: str) -> AsyncIterator[dict]: ...
    def set_view(self, run_id: str, view: dict, trajectories: dict) -> None: ...
    def get_view(self, run_id: str) -> dict | None: ...
    def get_trajectory(self, run_id: str, traj_id: str) -> dict | None: ...
    def status(self, run_id: str) -> str | None: ...
    def mark_done(self, run_id: str) -> None: ...
    def mark_error(self, run_id: str) -> None: ...


class _Run:
    def __init__(self) -> None:
        self.status: str = "running"
        self.events: list[dict] = []
        self.view: dict | None = None
        self.trajectories: dict = {}
        self.condition = asyncio.Condition()


class MemoryRunStore:
    """In-memory RunStore. Not persistent; lost on restart (spec §10)."""

    def __init__(self) -> None:
        self._runs: dict[str, _Run] = {}

    def create(self) -> str:
        run_id = uuid.uuid4().hex[:16]
        self._runs[run_id] = _Run()
        return run_id

    def _get(self, run_id: str) -> _Run | None:
        return self._runs.get(run_id)

    def append_event(self, run_id: str, event: dict) -> None:
        run = self._get(run_id)
        if run is None:
            return
        run.events.append(event)
        # Wake any SSE subscribers waiting on new events.
        async def _notify():
            async with run.condition:
                run.condition.notify_all()
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_notify())
        except RuntimeError:
            # No running loop (sync test context): subscribers not active.
            pass

    def events_snapshot(self, run_id: str) -> list[dict]:
        run = self._get(run_id)
        return list(run.events) if run else []

    async def subscribe(self, run_id: str) -> AsyncIterator[dict]:
        run = self._get(run_id)
        if run is None:
            return
        idx = 0
        while True:
            # Drain any events already buffered.
            while idx < len(run.events):
                yield run.events[idx]
                idx += 1
            # If run finished and we've drained everything, stop.
            if run.status in ("done", "error"):
                return
            async with run.condition:
                try:
                    await asyncio.wait_for(run.condition.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass  # re-check loop (also catches races)

    def set_view(self, run_id: str, view: dict, trajectories: dict) -> None:
        run = self._get(run_id)
        if run is None:
            return
        run.view = view
        run.trajectories = trajectories

    def get_view(self, run_id: str) -> dict | None:
        run = self._get(run_id)
        return run.view if run else None

    def get_trajectory(self, run_id: str, traj_id: str) -> dict | None:
        run = self._get(run_id)
        if run is None:
            return None
        return run.trajectories.get(traj_id)

    def status(self, run_id: str) -> str | None:
        run = self._get(run_id)
        return run.status if run else None

    def _mark(self, run_id: str, status: str) -> None:
        run = self._get(run_id)
        if run is None:
            return
        run.status = status
        async def _notify():
            async with run.condition:
                run.condition.notify_all()
        try:
            asyncio.get_running_loop().create_task(_notify())
        except RuntimeError:
            pass

    def mark_done(self, run_id: str) -> None:
        self._mark(run_id, "done")

    def mark_error(self, run_id: str) -> None:
        self._mark(run_id, "error")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest tests/service/test_runstore.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/service/runstore.py tests/service/test_runstore.py
git commit -m "feat(service): RunStore seam with in-memory implementation"
```

---

## Task 5: 编排层（orchestrator.py）

**Files:**
- Create: `src/service/orchestrator.py`
- Test: `tests/service/test_orchestrator.py`

**契约**（spec §7）：`run_pipeline` 抽自 e2e_smoke 的 main()，emit 进度事件，**依赖注入** compiler/pipeline/select 三个可调用，测试用 Fake 替身（不连真实 LLM）。多行清单跨行问题 id 加行前缀避免碰撞（spec §12 开放点3，本计划采纳"加前缀"）。

- [ ] **Step 1: 写失败测试**

创建 `tests/service/test_orchestrator.py`：

```python
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from service.orchestrator import run_pipeline, PipelineDeps


@dataclass
class FakeSubProblem:
    id: str
    failure_summary: str
    target_capability: list
    confidence: float = 0.9
    origin: str = "original"
    parent_id: object = None
    raw_text: str = ""
    trajectory_signal: str = ""
    hyde_positive: list = field(default_factory=list)
    keywords: list = field(default_factory=list)
    route: str = "pass"


@dataclass
class FakeFilters:
    languages: object = None
    tools_used: object = None
    has_verification_step: object = None


@dataclass
class FakeSpec:
    raw_input: str
    domain: str
    sub_problems: list


@dataclass
class FakeScored:
    trajectory_id: str
    slice_index: int
    trajectory_path: str
    sub_problem_id: str
    capability: list
    relevance_score: float
    judge_confidence: float
    loss_mask_spans: list
    judge_match: bool = True


class FakeCompiler:
    def __init__(self):
        self._n = 0

    async def compile(self, raw_input):
        self._n += 1
        sp = FakeSubProblem(
            id="p1",
            failure_summary=f"summary::{raw_input}",
            target_capability=["valid_syntax_in_toolcall"],
        )
        # add structured_filters attribute expected by serializer
        sp.structured_filters = FakeFilters()
        return FakeSpec(raw_input=raw_input, domain="agentic_swe", sub_problems=[sp])


class FakePipeline:
    async def run_scored(self, *, trajectory_paths, problem_specs):
        # one hit per sub_problem
        out = []
        for spec in problem_specs:
            for sp in spec["sub_problems"]:
                out.append(FakeScored(
                    "t1", 0, "dummy.jsonl", sp["id"],
                    ["valid_syntax_in_toolcall"], 0.8, 0.9,
                    [{"start_step": 0, "end_step": 1}],
                ))
        return out


def fake_select(candidates, *, sub_problem_ids, selection, general):
    return {
        "targeted": list(candidates),
        "general": [],
        "manifest": {"targeted_count": len(candidates),
                     "general_count": 0, "general_ratio": 0.3},
    }


def _deps(tmp_traj):
    return PipelineDeps(
        compiler=FakeCompiler(),
        pipeline=FakePipeline(),
        select_fn=fake_select,
        load_trajectories_fn=lambda p: [],
    )


async def test_run_pipeline_emits_stages_and_builds_view(tmp_path):
    traj = tmp_path / "t.jsonl"
    traj.write_text('{"id":"t1","messages":[]}\n')
    events = []
    view, trajectories = await run_pipeline(
        manifest_lines=["代码总有语法错误"],
        trajectory_path=traj,
        deps=_deps(traj),
        emit=lambda ev: events.append(ev),
    )
    stages = [e["stage"] for e in events]
    assert "module0" in stages
    assert "module1" in stages
    assert "module3" in stages
    assert view["problems"][0]["id"].endswith("p1")
    assert view["manifest"]["targeted_count"] == 1


async def test_multiline_manifest_prefixes_problem_ids(tmp_path):
    traj = tmp_path / "t.jsonl"
    traj.write_text('{"id":"t1","messages":[]}\n')
    view, _ = await run_pipeline(
        manifest_lines=["抱怨A", "抱怨B"],
        trajectory_path=traj,
        deps=_deps(traj),
        emit=lambda ev: None,
    )
    ids = [p["id"] for p in view["problems"]]
    assert len(ids) == 2
    assert len(set(ids)) == 2  # 跨行不碰撞
    assert all("p1" in i for i in ids)


async def test_blank_manifest_lines_skipped(tmp_path):
    traj = tmp_path / "t.jsonl"
    traj.write_text('{"id":"t1","messages":[]}\n')
    view, _ = await run_pipeline(
        manifest_lines=["抱怨A", "", "  "],
        trajectory_path=traj,
        deps=_deps(traj),
        emit=lambda ev: None,
    )
    assert len(view["problems"]) == 1  # 空行不编译
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest tests/service/test_orchestrator.py -v`
Expected: FAIL with "No module named 'service.orchestrator'"

- [ ] **Step 3: 实现 orchestrator.py**

创建 `src/service/orchestrator.py`：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest tests/service/test_orchestrator.py -v`
Expected: 3 passed

> 注：`SelectionConfig`/`GeneralDataConfig` 的 import 在 module3 分支内惰性加载，测试传入 fake select_fn 且不进该分支（fake select 忽略 config），不会触发真实 module3 import 问题。若测试触及该分支，module3 是纯 Python 无重依赖，import 安全。

- [ ] **Step 5: Commit**

```bash
git add src/service/orchestrator.py tests/service/test_orchestrator.py
git commit -m "feat(service): pipeline orchestrator with progress emit and DI"
```

---

## Task 6: FastAPI 服务（app.py）

**Files:**
- Create: `src/service/app.py`
- Test: `tests/service/test_app.py`

**契约**（spec §5.1）：4 个端点 + 静态前端。`create_app(store, run_fn)` 工厂便于测试注入。POST /runs 后台 asyncio task 跑 run_fn。

- [ ] **Step 1: 写失败测试**

创建 `tests/service/test_app.py`：

```python
# SPDX-License-Identifier: Apache-2.0
import asyncio

from fastapi.testclient import TestClient

from service.app import create_app
from service.runstore import MemoryRunStore


async def _fake_run(*, manifest_lines, trajectory_path, deps, emit, run_id,
                    selection_config=None, general_config=None):
    emit({"stage": "module0", "status": "running", "msg": "x"})
    emit({"stage": "module3", "status": "running", "msg": "y"})
    view = {
        "run_id": run_id,
        "problems": [{"id": "L1.p1", "failure_summary": "s",
                      "confidence": 0.9, "capabilities": []}],
        "manifest": {"targeted_count": 0, "general_count": 0, "general_ratio": 0.3},
    }
    trajectories = {"t1": {"trajectory_id": "t1", "steps": [
        {"index": 0, "role": "user", "content": "hi",
         "tool_call_name": None, "tool_call_args": None, "tool_result": None}]}}
    return view, trajectories


def _client():
    store = MemoryRunStore()
    app = create_app(store=store, run_fn=_fake_run, deps=object())
    return TestClient(app), store


def test_post_runs_returns_run_id():
    client, _ = _client()
    resp = client.post(
        "/runs",
        files={
            "manifest": ("m.txt", "代码总有语法错误\n改完不验证", "text/plain"),
            "trajectories": ("t.jsonl", '{"id":"t1","messages":[]}\n',
                             "application/x-ndjson"),
        },
    )
    assert resp.status_code == 200
    assert "run_id" in resp.json()


def test_view_available_after_run_completes():
    client, store = _client()
    rid = client.post("/runs", files={
        "manifest": ("m.txt", "x", "text/plain"),
        "trajectories": ("t.jsonl", '{"id":"t1","messages":[]}\n', "application/x-ndjson"),
    }).json()["run_id"]

    # background task runs in the TestClient's event loop; poll view
    for _ in range(50):
        r = client.get(f"/runs/{rid}/view")
        if r.status_code == 200:
            break
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == rid
    assert body["problems"][0]["id"] == "L1.p1"


def test_view_409_before_done():
    client, store = _client()
    rid = store.create()  # created but never run → still "running"
    r = client.get(f"/runs/{rid}/view")
    assert r.status_code == 409


def test_view_404_unknown_run():
    client, _ = _client()
    r = client.get("/runs/nope/view")
    assert r.status_code == 404


def test_trajectory_endpoint_returns_steps():
    client, _ = _client()
    rid = client.post("/runs", files={
        "manifest": ("m.txt", "x", "text/plain"),
        "trajectories": ("t.jsonl", '{"id":"t1","messages":[]}\n', "application/x-ndjson"),
    }).json()["run_id"]
    for _ in range(50):
        if client.get(f"/runs/{rid}/view").status_code == 200:
            break
    r = client.get(f"/runs/{rid}/trajectory/t1")
    assert r.status_code == 200
    assert r.json()["steps"][0]["content"] == "hi"


def test_trajectory_404_unknown():
    client, _ = _client()
    rid = client.post("/runs", files={
        "manifest": ("m.txt", "x", "text/plain"),
        "trajectories": ("t.jsonl", '{"id":"t1","messages":[]}\n', "application/x-ndjson"),
    }).json()["run_id"]
    for _ in range(50):
        if client.get(f"/runs/{rid}/view").status_code == 200:
            break
    r = client.get(f"/runs/{rid}/trajectory/missing")
    assert r.status_code == 404
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/pytest tests/service/test_app.py -v`
Expected: FAIL with "No module named 'service.app'"

- [ ] **Step 3: 实现 app.py**

创建 `src/service/app.py`：

```python
"""FastAPI service for the Trajectory Inspector (spec §5.1)."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from __future__ import annotations

import asyncio
import json
import pathlib
import tempfile
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from service.runstore import MemoryRunStore

__all__ = ["create_app"]

_WEB_DIR = pathlib.Path(__file__).parent / "web"


def create_app(
    *,
    store: Any = None,
    run_fn: Callable | None = None,
    deps: Any = None,
) -> FastAPI:
    """Build the app. Inject store/run_fn/deps for testing; defaults for prod."""
    store = store or MemoryRunStore()
    if run_fn is None:
        from service.orchestrator import run_pipeline
        run_fn = run_pipeline

    app = FastAPI(title="Trajectory Inspector")

    async def _background_run(run_id: str, manifest_text: str, traj_path: pathlib.Path):
        def emit(ev: dict) -> None:
            store.append_event(run_id, ev)
        try:
            lines = manifest_text.splitlines()
            view, trajectories = await run_fn(
                manifest_lines=lines,
                trajectory_path=traj_path,
                deps=deps,
                emit=emit,
                run_id=run_id,
            )
            store.set_view(run_id, view, trajectories)
            store.append_event(run_id, {"stage": "done", "status": "ok"})
            store.mark_done(run_id)
        except Exception as exc:  # noqa: BLE001 — surface as error event, don't crash server
            store.append_event(run_id, {"stage": "done", "status": "error",
                                        "msg": str(exc)})
            store.mark_error(run_id)

    @app.post("/runs")
    async def create_run(manifest: UploadFile, trajectories: UploadFile):
        manifest_text = (await manifest.read()).decode("utf-8")
        traj_bytes = await trajectories.read()
        # persist uploaded jsonl to a temp file for the loader/pipeline
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".jsonl", mode="wb")
        tmp.write(traj_bytes)
        tmp.close()
        run_id = store.create()
        asyncio.create_task(
            _background_run(run_id, manifest_text, pathlib.Path(tmp.name))
        )
        return {"run_id": run_id}

    @app.get("/runs/{run_id}/events")
    async def stream_events(run_id: str):
        if store.status(run_id) is None:
            raise HTTPException(status_code=404, detail="unknown run")

        async def gen():
            async for ev in store.subscribe(run_id):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/runs/{run_id}/view")
    async def get_view(run_id: str):
        status = store.status(run_id)
        if status is None:
            raise HTTPException(status_code=404, detail="unknown run")
        view = store.get_view(run_id)
        if view is None:
            # run exists but not finished (or errored without view)
            raise HTTPException(status_code=409, detail=f"run not ready: {status}")
        return JSONResponse(view)

    @app.get("/runs/{run_id}/trajectory/{trajectory_id}")
    async def get_trajectory(run_id: str, trajectory_id: str):
        if store.status(run_id) is None:
            raise HTTPException(status_code=404, detail="unknown run")
        traj = store.get_trajectory(run_id, trajectory_id)
        if traj is None:
            raise HTTPException(status_code=404, detail="unknown trajectory")
        return JSONResponse(traj)

    # Static frontend (mounted last so API routes take precedence)
    if _WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")

    return app
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/pytest tests/service/test_app.py -v`
Expected: 6 passed

> 注：若 StaticFiles mount 在 `_WEB_DIR` 不存在时报错，Task 6 尚未建 web/ 目录——`if _WEB_DIR.exists()` 守卫确保测试期不 mount。Task 7 建 web/ 后自动生效。

- [ ] **Step 5: Commit**

```bash
git add src/service/app.py tests/service/test_app.py
git commit -m "feat(service): FastAPI endpoints for runs/events/view/trajectory"
```

---

## Task 7: 前端单页（web/）

**Files:**
- Create: `src/service/web/index.html`
- Create: `src/service/web/style.css`
- Create: `src/service/web/app.js`

**布局**（spec 决策10-12）：方案A 三栏（问题+能力 / 命中轨迹列表 / 轨迹详情）；能力点选聚焦、后端稳定色；非聚焦能力片段淡显+小标签；"显示全部能力"总开关。

> 前端无自动化测试（原生页面），验证靠手动跑服务看页面。TDD 不适用本 Task。

- [ ] **Step 1: 写 index.html**

创建 `src/service/web/index.html`：

```html
<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>轨迹 Inspector</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body>
  <header>
    <h1>轨迹 Inspector</h1>
    <div class="uploads">
      <label>用户清单(.txt)：<input type="file" id="manifest" accept=".txt"></label>
      <label>候选回流(.jsonl)：<input type="file" id="trajectories" accept=".jsonl"></label>
      <button id="run">运行</button>
      <label class="toggle"><input type="checkbox" id="show-all"> 显示全部能力</label>
    </div>
    <div id="manifest-summary" class="manifest"></div>
  </header>

  <div id="progress" class="progress hidden">
    <p class="subtitle" id="progress-msg">先上传两个文件，点运行…</p>
  </div>

  <main id="workspace" class="split3 hidden">
    <section id="col-problems" class="col">
      <div class="label">问题 → 能力</div>
      <div id="problems"></div>
    </section>
    <section id="col-hits" class="col">
      <div class="label">命中轨迹</div>
      <div id="hits"></div>
    </section>
    <section id="col-detail" class="col">
      <div class="label">轨迹详情</div>
      <div id="detail"></div>
    </section>
  </main>

  <script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: 写 style.css**

创建 `src/service/web/style.css`（继承 mockup 配色）：

```css
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 0; color: #222; font-size: 13px; }
header { padding: 12px 16px; border-bottom: 1px solid #ddd; }
h1 { font-size: 16px; margin: 0 0 8px; }
.uploads { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
.uploads label { font-size: 12px; }
button { padding: 6px 16px; cursor: pointer; }
.toggle { margin-left: auto; }
.manifest { margin-top: 6px; color: #888; font-size: 11px; }
.subtitle { color: #888; }
.hidden { display: none; }
.progress { padding: 24px 16px; }

.split3 { display: flex; gap: 8px; padding: 8px; height: calc(100vh - 90px); }
.col { border: 1px solid #ccc; padding: 8px; overflow-y: auto; }
#col-problems { flex: 0 0 26%; }
#col-hits { flex: 0 0 28%; }
#col-detail { flex: 1; }
.label { font-weight: 600; color: #555; margin-bottom: 6px; font-size: 11px; }

.problem { background: #ffe0e0; padding: 5px 6px; margin: 4px 0; border-radius: 3px; cursor: pointer; }
.problem.active { outline: 2px solid #e64980; }
.capability { padding: 4px 6px; margin: 3px 0 3px 10px; border-radius: 3px; cursor: pointer; font-size: 12px; }
.capability.dimmed { opacity: .5; }
.capability.focused { outline: 2px solid currentColor; font-weight: 600; }

.hit { background: #eef; padding: 5px 6px; margin: 4px 0; border-radius: 3px; cursor: pointer; }
.hit.active { outline: 2px solid #4263eb; }
.hit .selected-badge { font-size: 9px; background: #37b24d; color: #fff; padding: 1px 5px; border-radius: 8px; margin-left: 4px; }

.step { padding: 4px 6px; margin: 3px 0; border-radius: 3px; background: #f6f6f6; white-space: pre-wrap; }
.step.user { background: #f0f0f0; }
.step.tool { background: #e8f5e8; }
.step.hit-span { border-left: 4px solid; }
.step.hit-dim { opacity: .55; }
.step .cap-tag { font-size: 9px; padding: 1px 5px; border-radius: 2px; color: #fff; margin-left: 4px; }
```

- [ ] **Step 3: 写 app.js**

创建 `src/service/web/app.js`：

```javascript
// Trajectory Inspector frontend (spec §9, decisions 10-12).
const $ = (id) => document.getElementById(id);

let state = {
  runId: null,
  view: null,
  activeProblem: null,   // problem id
  activeCap: null,       // focused capability label
  showAll: false,        // "显示全部能力" toggle
  activeTraj: null,      // {trajectory_id, capLabel}
  trajCache: {},
};

$("run").addEventListener("click", startRun);
$("show-all").addEventListener("change", (e) => {
  state.showAll = e.target.checked;
  renderDetail();
  renderProblems();
});

async function startRun() {
  const mf = $("manifest").files[0];
  const tj = $("trajectories").files[0];
  if (!mf || !tj) { alert("请先上传清单和轨迹两个文件"); return; }
  const fd = new FormData();
  fd.append("manifest", mf);
  fd.append("trajectories", tj);

  $("progress").classList.remove("hidden");
  $("workspace").classList.add("hidden");
  $("progress-msg").textContent = "上传中…";

  const resp = await fetch("/runs", { method: "POST", body: fd });
  const { run_id } = await resp.json();
  state.runId = run_id;
  subscribeEvents(run_id);
}

function subscribeEvents(runId) {
  const es = new EventSource(`/runs/${runId}/events`);
  es.onmessage = async (e) => {
    const ev = JSON.parse(e.data);
    $("progress-msg").textContent = `[${ev.stage}] ${ev.msg || ev.status}`;
    if (ev.stage === "done") {
      es.close();
      if (ev.status === "error") {
        $("progress-msg").textContent = "运行出错: " + (ev.msg || "");
        return;
      }
      await loadView(runId);
    }
  };
  es.onerror = () => { es.close(); };
}

async function loadView(runId) {
  const resp = await fetch(`/runs/${runId}/view`);
  state.view = await resp.json();
  $("progress").classList.add("hidden");
  $("workspace").classList.remove("hidden");
  renderManifest();
  renderProblems();
  $("hits").innerHTML = "";
  $("detail").innerHTML = "";
}

function renderManifest() {
  const m = state.view.manifest || {};
  $("manifest-summary").textContent =
    `最终选集: targeted=${m.targeted_count} · general=${m.general_count} · ratio=${m.general_ratio}`;
}

function renderProblems() {
  const el = $("problems");
  el.innerHTML = "";
  for (const p of state.view.problems) {
    const pd = document.createElement("div");
    pd.className = "problem" + (p.id === state.activeProblem ? " active" : "");
    pd.textContent = `❗ ${p.failure_summary} (${p.confidence.toFixed(2)})`;
    pd.onclick = () => selectProblem(p.id);
    el.appendChild(pd);

    if (p.id === state.activeProblem) {
      for (const cap of p.capabilities) {
        const cd = document.createElement("div");
        const focused = cap.label === state.activeCap;
        const dimmed = state.activeCap && !focused && !state.showAll;
        cd.className = "capability" + (focused ? " focused" : "") + (dimmed ? " dimmed" : "");
        cd.style.color = cap.color;
        cd.style.background = cap.color + "22";
        cd.textContent = `${cap.label} (${cap.hit_count})`;
        cd.onclick = () => focusCapability(p.id, cap.label);
        el.appendChild(cd);
      }
    }
  }
}

function selectProblem(pid) {
  state.activeProblem = pid;
  state.activeCap = null;
  state.activeTraj = null;
  renderProblems();
  renderHits();
  $("detail").innerHTML = "";
}

function focusCapability(pid, label) {
  state.activeCap = (state.activeCap === label) ? null : label;
  renderProblems();
  renderHits();
  renderDetail();
}

function currentProblem() {
  return state.view.problems.find((p) => p.id === state.activeProblem);
}

// Union of hit trajectories across capabilities (or focused capability only)
function currentHits() {
  const p = currentProblem();
  if (!p) return [];
  const caps = state.activeCap && !state.showAll
    ? p.capabilities.filter((c) => c.label === state.activeCap)
    : p.capabilities;
  const byTraj = {};
  for (const cap of caps) {
    for (const h of cap.hit_trajectories) {
      const key = `${h.trajectory_id}#${h.slice_index}`;
      if (!byTraj[key]) byTraj[key] = { ...h, caps: [] };
      byTraj[key].caps.push({ label: cap.label, color: cap.color, spans: h.loss_mask_spans });
    }
  }
  return Object.values(byTraj);
}

function renderHits() {
  const el = $("hits");
  el.innerHTML = "";
  const hits = currentHits();
  for (const h of hits) {
    const hd = document.createElement("div");
    const isActive = state.activeTraj && state.activeTraj.trajectory_id === h.trajectory_id
      && state.activeTraj.slice_index === h.slice_index;
    hd.className = "hit" + (isActive ? " active" : "");
    const badge = h.selected ? '<span class="selected-badge">已入选</span>' : "";
    hd.innerHTML = `${h.trajectory_id} · slice${h.slice_index} · ${h.caps.length}片段 ${badge}`;
    hd.onclick = () => selectTrajectory(h);
    el.appendChild(hd);
  }
}

async function selectTrajectory(hit) {
  state.activeTraj = { trajectory_id: hit.trajectory_id, slice_index: hit.slice_index };
  await ensureTrajectory(hit.trajectory_id);
  renderHits();
  renderDetail();
}

async function ensureTrajectory(tid) {
  if (state.trajCache[tid]) return;
  const resp = await fetch(`/runs/${state.runId}/trajectory/${tid}`);
  state.trajCache[tid] = await resp.json();
}

// Which capabilities' spans should highlight this step (focus vs show-all)
function spansForStep(stepIndex, hit) {
  const out = [];
  for (const cap of hit.caps) {
    const inSpan = (cap.spans || []).some(
      (s) => stepIndex >= s.start_step && stepIndex <= s.end_step);
    if (!inSpan) continue;
    const focused = !state.activeCap || state.showAll || cap.label === state.activeCap;
    out.push({ ...cap, focused });
  }
  return out;
}

function renderDetail() {
  const el = $("detail");
  el.innerHTML = "";
  if (!state.activeTraj) return;
  const traj = state.trajCache[state.activeTraj.trajectory_id];
  if (!traj) return;
  const hit = currentHits().find(
    (h) => h.trajectory_id === state.activeTraj.trajectory_id
      && h.slice_index === state.activeTraj.slice_index);

  for (const step of traj.steps) {
    const sd = document.createElement("div");
    let cls = "step " + step.role;
    const caps = hit ? spansForStep(step.index, hit) : [];
    const focusedCaps = caps.filter((c) => c.focused);
    const dimCaps = caps.filter((c) => !c.focused);

    if (focusedCaps.length) {
      cls += " hit-span";
      sd.style.borderLeftColor = focusedCaps[0].color;
      sd.style.background = focusedCaps[0].color + "22";
    } else if (dimCaps.length) {
      cls += " hit-dim";  // 非聚焦能力命中：淡显（decision 11）
    }
    sd.className = cls;

    const roleIcon = { user: "👤", assistant: "🤖", tool: "⚙️", system: "⚙" }[step.role] || "";
    let text = `${roleIcon} ${step.role}`;
    if (step.tool_call_name) text += ` 🔧 ${step.tool_call_name}`;
    const body = (step.content || step.tool_result || "").slice(0, 300);
    sd.textContent = `${text}: ${body}`;

    // 小色标签：命中该 step 的所有能力（含非聚焦，decision 11）
    for (const c of caps) {
      const tag = document.createElement("span");
      tag.className = "cap-tag";
      tag.style.background = c.color;
      tag.textContent = c.label;
      if (!c.focused) tag.style.opacity = ".6";
      sd.appendChild(tag);
    }
    el.appendChild(sd);
  }
}
```

- [ ] **Step 4: 手动验证服务启动 + 页面加载**

Run:
```bash
.venv/bin/python -c "from service.app import create_app; app = create_app(); print([r.path for r in app.routes])"
```
Expected: 打印路由列表，含 `/runs`、`/runs/{run_id}/events`、`/runs/{run_id}/view`、`/runs/{run_id}/trajectory/{trajectory_id}`，且无异常（web/ 目录已存在，StaticFiles 正常 mount）。

- [ ] **Step 5: 跑全量 service 测试确认无回归**

Run: `.venv/bin/pytest tests/service/ -v`
Expected: 全部 passed（web/ 建好后 StaticFiles mount，app 测试仍 green）。

- [ ] **Step 6: Commit**

```bash
git add src/service/web/
git commit -m "feat(service): native single-page Inspector frontend (3-column, focus highlight)"
```

---

## Task 8: 生产装配入口 + 冒烟脚本

**Files:**
- Create: `scripts/inspector_serve.py`
- Modify: `src/service/__init__.py`

**目的**：把真实 compiler/pipeline/embedding 装配进 `PipelineDeps`，提供 `uvicorn` 启动入口。这是唯一连真实 LLM/embedding 的地方，不进单元测试。

- [ ] **Step 1: 导出公共 API**

修改 `src/service/__init__.py`，追加：

```python
"""Trajectory Inspector service: orchestration + aggregation + HTTP/SSE."""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

from service.app import create_app
from service.orchestrator import PipelineDeps, run_pipeline
from service.runstore import MemoryRunStore, RunStore
from service.viewmodel import build_inspector_view, build_trajectory_index

__all__ = [
    "create_app",
    "PipelineDeps",
    "run_pipeline",
    "MemoryRunStore",
    "RunStore",
    "build_inspector_view",
    "build_trajectory_index",
]
```

- [ ] **Step 2: 写生产装配脚本**

创建 `scripts/inspector_serve.py`：

```python
"""Launch the Trajectory Inspector service with real LLM + embedding.

Usage:
    .venv/bin/python scripts/inspector_serve.py
    # then open http://localhost:8000
"""
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dengxuanliang

import functools
import os
import pathlib

from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent.parent / ".env")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import uvicorn  # noqa: E402

from llm_gateway import LLMGateway, GatewayConfig  # noqa: E402
from module0 import QueryCompiler, Taxonomy  # noqa: E402
from module0.embedding import EmbeddingModel  # noqa: E402
from module1.pipeline import TrajectoryPipeline, PipelineConfig  # noqa: E402
from module1.loader import load_trajectories  # noqa: E402
from module3.pipeline import select_final_dataset  # noqa: E402
from service import create_app, MemoryRunStore, PipelineDeps, run_pipeline  # noqa: E402


def build_app():
    root = pathlib.Path(__file__).parent.parent
    model = os.environ.get("MODULE0_TEST_MODEL", "gpt-4o-mini")
    emb = EmbeddingModel()
    taxonomy = Taxonomy.load(root / "fixtures" / "taxonomy_v0.json")

    gw_config = GatewayConfig(
        litellm_base=os.environ.get("LITELLM_BASE", "http://localhost:4000/v1"),
        litellm_key=os.environ.get("LITELLM_KEY", ""),
        transport_stuck_seconds=0,
    )
    gateway = LLMGateway(gw_config)  # long-lived; entered on startup

    compiler = QueryCompiler(
        gateway=gateway, taxonomy=taxonomy, model=model, embedding_model=emb)
    cfg = PipelineConfig(
        judge_model=model, recall_top_n=20, min_confidence=0.7, embedding_model=emb)
    pipeline = TrajectoryPipeline(config=cfg, gateway=gateway)

    deps = PipelineDeps(
        compiler=compiler,
        pipeline=pipeline,
        select_fn=select_final_dataset,
        load_trajectories_fn=load_trajectories,
    )

    store = MemoryRunStore()
    app = create_app(store=store, run_fn=run_pipeline, deps=deps)

    @app.on_event("startup")
    async def _open_gateway():
        await gateway.__aenter__()

    @app.on_event("shutdown")
    async def _close_gateway():
        await gateway.__aexit__(None, None, None)

    return app


if __name__ == "__main__":
    uvicorn.run(build_app(), host="127.0.0.1", port=8000)
```

- [ ] **Step 3: 验证装配 import 无误（不启动服务）**

Run: `.venv/bin/python -c "import service; print(service.__all__)"`
Expected: 打印导出列表，无 import 错误。

> 注：`scripts/inspector_serve.py` 的 `build_app()` 连真实 embedding/gateway，不在此步执行；仅验证 `service` 包导出正确。真实启动需 `.env` 就位 + Qwen 模型缓存，属手动验收。

- [ ] **Step 4: 跑全量测试套件确认无回归**

Run: `.venv/bin/pytest -q`
Expected: 现有 273 tests + 新增 service tests 全部 passed（新测试不触真实 LLM）。

- [ ] **Step 5: Commit**

```bash
git add src/service/__init__.py scripts/inspector_serve.py
git commit -m "feat(service): production wiring + uvicorn launch script"
```

---

## Self-Review（写计划后自查）

**1. Spec coverage:**
- spec §5.1 端点 → Task 6 ✅
- spec §5.2 InspectorView schema → Task 3 ✅
- spec §5.3 稳定配色 → Task 2 ✅
- spec §6 聚合逻辑 → Task 3 ✅
- spec §7 编排层 → Task 5 ✅
- spec §8 RunStore seam → Task 4 ✅
- spec §9 前端三栏 → Task 7 ✅
- spec §11 依赖变更 → Task 1 ✅
- spec 决策10-12（三栏/淡显+标签/总开关）→ Task 7 ✅
- spec §12 开放点3（跨行 id 前缀）→ Task 5 采纳"L{n}." 前缀 ✅

**2. Placeholder scan:** 无 TBD/TODO；每个代码步含完整代码；测试步含完整断言。✅

**3. Type consistency:**
- `build_inspector_view(*, run_id, spec, scored, select_result)` — Task 3 定义，Task 5 调用一致 ✅
- `build_trajectory_index(trajectories)` — Task 3 定义，Task 5 调用一致 ✅
- `PipelineDeps(compiler, pipeline, select_fn, load_trajectories_fn)` — Task 5 定义，Task 8 装配一致 ✅
- `run_pipeline(*, manifest_lines, trajectory_path, deps, emit, run_id, ...)` — Task 5 定义，Task 6 `_fake_run` 与 Task 8 调用签名一致 ✅
- `MemoryRunStore` 方法（create/append_event/subscribe/set_view/get_view/get_trajectory/status/mark_done/mark_error）— Task 4 定义，Task 6 使用一致 ✅
- `create_app(*, store, run_fn, deps)` — Task 6 定义，Task 8 调用一致 ✅
- InspectorView 字段（run_id/problems/manifest；capability.color/hit_count/hit_trajectories；hit.selected/loss_mask_spans）— Task 3 产出与 Task 7 前端消费一致 ✅

**开放点残留**（不阻塞实现，实现时可定）：
- manifest 展示位置 → Task 7 放 header 的 `#manifest-summary`（已定）。
- taxonomy parent 富化 → viewmodel 里 `parent: None` 占位，spec §10 标为后续增强（已留 seam）。
