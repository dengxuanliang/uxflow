# Inspector 服务 Code Review 修复方案

> **状态**：方案定稿，待评审。
> **范围**：修复 code review 发现的 13 个问题（4 Important 会真实触发 + 1 Important 数据语义 + 8 Minor）。
> **原则**：最小侵入。前端修状态/XSS/竞态，后端修 cancel 生命周期 + 存储淘汰 + 几处防御。**不改 module0/1/2/3 的核心算法**——问题5的"虚假归属"在 viewmodel 聚合层解决，不动 judge/rerank。
> **上游**：本轮 code review（4 个 reviewer 交叉验证）。

---

## 改动文件总览

| 文件 | 涉及问题 | 类型 |
|---|---|---|
| `src/service/web/app.js` | 1, 2, 4, 8, 13 | 前端逻辑/XSS/竞态 |
| `src/service/web/index.html` | 4 | 运行按钮禁用 |
| `src/service/app.py` | 3 | cancel 生命周期 |
| `src/service/runstore.py` | 9 | 存储淘汰 + notify 引用保持 |
| `src/service/viewmodel.py` | 5, 7, 10, 11 | 聚合层折叠/过滤/防御 |
| `src/module1/pipeline.py` | 6 | on_progress 异常隔离 |
| `scripts/gen_demo_trajectories.py` + 重新生成 jsonl | 12 | tool_call id 唯一化 |
| `tests/service/*` | 1,2,3,5,6,7,9,11 | 回归锁定 |

---

## 前端修复（app.js + index.html）

### 问题1 [Important] 跨 run 状态污染 → 显示错数据

**根因**：`resetProgress()` 只清进度相关状态，`view/activeProblem/activeCap/activeTraj/trajCache` 从不重置；`trajCache` 只按 `trajectory_id` 键控，两次上传若有同名 ID 会命中旧缓存直接返回旧内容（`ensureTrajectory` 的 `if (state.trajCache[tid]) return`）。

**修法**：在 `startRun()` 里、`resetProgress()` 之前，重置全部视图/选择/缓存状态。新增一个 `resetRunState()`：

```javascript
function resetRunState() {
  state.view = null;
  state.activeProblem = null;
  state.activeCap = null;
  state.activeTraj = null;
  state.trajCache = {};
}
```
`startRun()` 里 `resetProgress()` 后调用 `resetRunState()`。

**额外加固**：`trajCache` 键改为 `runId + "#" + tid`，彻底杜绝跨 run 碰撞——`ensureTrajectory(tid)` 和 `renderDetail`/`selectTrajectory` 读缓存处统一用 `cacheKey(tid) = state.runId + "#" + tid`。

**测试**：前端无自动化测试，靠手动验证（跑 A→开轨迹→换文件跑 B→开同名轨迹应拉新内容）。

---

### 问题2 [Important] XSS：三处 innerHTML 未转义

**根因**：`cap.label`（renderProblems 里能力项）、`trajectory_id`（renderHits）、`step.role`（renderDetail 的 step-head）三处直接进 innerHTML 没走 escapeHtml。同一 `cap.label` 在 modal 里是转义的，问题栏漏了。

**修法**：三处包 `escapeHtml()`：
- renderProblems 能力项：`● ${escapeHtml(cap.label)}`
- renderHits：`<span class="hit-id">${escapeHtml(h.trajectory_id)}</span>`
- renderDetail step-head：`${roleIcon} ${escapeHtml(step.role)}`

**验证**：手动——上传一条 `id` 含 `<img src=x onerror=alert(1)>` 的轨迹，确认不弹窗、原样文本显示。

---

### 问题4 [Important] 重复点运行 / 上传阶段点停止失效

**根因**：
- (a) 运行按钮运行期不禁用，中途再点开第二个 EventSource 不关旧的。
- (b) `state.runId` 在 `await fetch("/runs")` 之后才赋值，上传窗口内点停止直接 `return`，计时器不停也不取消。
- (c) `/runs` POST 无 try/catch，失败抛未处理异常、计时器永跑。

**修法**：
1. **运行按钮禁用**：`startRun()` 开头 `$("run").disabled = true`；在 run 结束的所有出口（done 分支、onerror、stopRun）恢复 `$("run").disabled = false`。index.html 无需改（disabled 用 JS 控制），但 CSS 加一条 `button#run:disabled { opacity:.5; cursor:not-allowed; }`。
2. **subscribeEvents 开头关旧流**：`if (state.es) { state.es.close(); state.es = null; }`（双保险，即使按钮禁用被绕过）。
3. **停止支持上传阶段**：引入 `state.stopping` 标志。`startRun` 里设 `state.stopping = false`；`stopRun()` 去掉 `if (!state.runId) return` 的早退，改为：设 `state.stopping = true`、停计时器、隐藏停止、恢复运行按钮；若 `state.runId` 已存在才发 `/cancel`。`startRun` 在 `await fetch` 拿到 run_id 后检查 `if (state.stopping) { 发 cancel 或直接不 subscribe; return; }`。
4. **POST 包 try/catch**：`await fetch("/runs")` 包 try/catch，失败时 `setMsg("上传失败: ...")` + 停计时器 + 恢复按钮。

**测试**：手动——(a) 运行中按钮应禁用点不动；(b) 上传瞬间点停止应真停；(c) 关掉服务再点运行应显示上传失败而非卡死。

---

### 问题8 [Minor] ETA 计算与注释不符

**根因**：注释说"从第一条完成后外推"，实际 `perItem = (now - startedAt) / done`（含上传+module0 warmup），早期 ETA 系统性偏高。

**修法**：改成从第一条完成时刻测量：
```javascript
// done>=1 且已记录首完成时刻后：用「首完成之后的耗时 / 已完成的后续条数」外推
if (done >= 1 && state.compileFirstDoneAt === null) {
  state.compileFirstDoneAt = Date.now();
} else if (done >= 2 && state.compileFirstDoneAt !== null && done < total) {
  const perItem = (Date.now() - state.compileFirstDoneAt) / (done - 1);
  const remain = perItem * (total - done);
  $("prog-eta").textContent = `编译约剩 ~${fmtDur(remain / 1000)}`;
}
```
注意：`done - 1` 做分母，需 `done >= 2` 才有意义（否则除零/无样本）；`done === 1` 时只记录时刻不显示 ETA。

**测试**：手动观察多条清单编译时 ETA 更贴近实际。

---

### 问题13 [Minor] 第二次 run elapsed 残留旧值

**根因**：`resetProgress()` 不重置 `#prog-elapsed`，新 run 前 500ms 显示旧时间。

**修法**：`resetProgress()` 加 `$("prog-elapsed").textContent = "⏱ 0:00";`

---

### 问题6-frontend 无（6 在后端）

---

## 后端修复

### 问题3 [Important] cancel 在任务启动前到达 → 永久卡死（app.py）

**根因**：清理逻辑全在 `_background_run` 的 `try/finally` 内，只有协程体开始执行才会跑。若 `task.cancel()` 在协程首次被调度前到达，body 不执行 → 临时文件泄漏 + run 停在 `running` + SSE 永久挂。窗口极窄但后果永久。

**修法**：用 `task.add_done_callback` 做兜底清理，不依赖协程体是否启动。改 `create_run`：

```python
run_id = store.create()
task = asyncio.create_task(
    _background_run(run_id, manifest_text, pathlib.Path(tmp.name))
)
_tasks[run_id] = task

def _finalize(t: asyncio.Task, rid=run_id, path=pathlib.Path(tmp.name)):
    # Safety net: runs no matter what — covers cancel-before-body-start,
    # where _background_run's try/finally never executed.
    path.unlink(missing_ok=True)          # idempotent (missing_ok)
    _tasks.pop(rid, None)
    if store.status(rid) == "running":    # body never finalized → force terminal
        store.append_event(rid, {"stage": "done", "status": "cancelled", "msg": "已停止"})
        store.mark_error(rid)

task.add_done_callback(_finalize)
return {"run_id": run_id}
```

**关键点**：
- `_background_run` 的现有 `finally`（unlink + pop）**保留**——正常路径它先执行，`_finalize` 里 unlink 幂等（missing_ok）、pop 幂等、status 已是 terminal 不重复标记。两者不冲突。
- `_finalize` 是 done_callback，任务以任何方式结束（正常/异常/取消/取消前未启动）都会触发。
- 顺带覆盖问题#2变体（`_background_run` 里未被 `except Exception` 捕获的其他 BaseException）：那种情况 body 的 finally 仍跑 unlink，但没 mark_error → `_finalize` 的 `status=="running"` 检查会补上终态。

**测试**（test_app.py 新增）：
- `test_cancel_before_body_finalizes`：构造一个 run_fn 里第一步就 await 一个可控 event，POST 后立刻 cancel，poll 直到 status 变 error，断言 view 409 + 有 cancelled 终态事件 + （难测临时文件，可跳过）。
- 实操上"取消前 body 未启动"极难稳定复现，改为测 `_finalize` 的兜底语义：直接测"一个卡住的 run 被 cancel 后一定进 terminal"（已有 `test_cancel_stops_run_and_marks_error` 覆盖主路径，补一个断言 done_callback 后 `_tasks` 被清空）。

---

### 问题9 [Minor] MemoryRunStore 无淘汰 → 内存无界增长（runstore.py）

**根因**：`_runs` 永不淘汰，每个 run 的 events/view/trajectories 保留到进程结束。

**修法**：加一个**简单的容量上限 + FIFO 淘汰**（LRU 过度设计，V1 够用）。`MemoryRunStore.__init__` 加 `max_runs=50`；`create()` 里若 `len(self._runs) >= max_runs`，淘汰最早的**已终态**run（不淘汰 running 中的）：

```python
def __init__(self, max_runs: int = 50) -> None:
    self._runs: dict[str, _Run] = {}
    self._max_runs = max_runs

def create(self) -> str:
    self._evict_if_needed()
    run_id = uuid.uuid4().hex[:16]
    self._runs[run_id] = _Run()
    return run_id

def _evict_if_needed(self) -> None:
    if len(self._runs) < self._max_runs:
        return
    # dict 保序：从最早插入的开始，淘汰第一个已终态的 run
    for rid, run in list(self._runs.items()):
        if run.status in ("done", "error"):
            del self._runs[rid]
            if len(self._runs) < self._max_runs:
                return
```

**问题5/5(fire-and-forget notify 引用)**：reviewer 标为 bounded/低危。顺手加固——`_notify` task 存入一个 `self._notify_tasks: set`，done 时 discard，防 GC。改 `append_event`/`_mark` 里的 `loop.create_task(_notify())`：
```python
t = loop.create_task(_notify())
self._notify_tasks.add(t)
t.add_done_callback(self._notify_tasks.discard)
```
（`__init__` 加 `self._notify_tasks = set()`）

**测试**（test_runstore.py 新增）：
- `test_evicts_oldest_terminal_run_at_capacity`：max_runs=3，create 3 个并全 mark_done，第 4 个 create 应淘汰最早的；running 中的不被淘汰。

---

### 问题5 [Important] + 问题7 [Minor]：每能力虚假归属 + hit_count 含 miss（viewmodel.py）

**根因**：后端 judge 按 slice 判定、把整个 `target_capability` 列表盖到每个候选（`rerank.py:43`）。所以一个 sub_problem 的多个能力命中集完全相同、spans 相同 → UI "N片段"虚高、同段贴多个色标签。且 `hit_count` 把 judge miss（衰减 0.3、可能空 spans）也算进去。

**修法（在 viewmodel 聚合层，不动 judge/rerank 算法）**：两步——
1. **过滤 miss**（问题7）：hit_trajectories 只保留 `judge_match=True` 且 `loss_mask_spans` 非空的候选。`ScoredCandidate` 有 `judge_match` 字段，viewmodel 能读到。
2. **折叠重复归属**（问题5）：因为同一 sub_problem 下每个能力的命中集相同，把"每能力独立挂命中"改成"**该 sub_problem 的命中集只算一次**，能力标签作为该命中的属性列出"。

具体重构 `build_inspector_view` 的 capabilities 构造：

```python
# 该 sub_problem 的去重命中集（按 slice 唯一），只保留真正 judge 命中且有 spans 的
seen = {}
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
            "selected": (c.trajectory_id, c.slice_index, sp_id) in selected_keys,
            "loss_mask_spans": spans,
        }

# capabilities：每个 label 仍列出（用于左栏彩色标签 + 聚焦），
# 但 hit_trajectories 指向同一份去重命中集；hit_count = 真实命中数
sub_hits = list(seen.values())
capabilities = []
for label in target_caps:
    capabilities.append({
        "label": label,
        "parent": None,
        "color": colors.get(label, "#888888"),
        "hit_count": len(sub_hits),
        "hit_trajectories": sub_hits,   # 同一份，不再每能力重复
    })
```

**这样的效果**：
- 左栏仍显示每个能力（彩色标签、可聚焦）——保留能力区分。
- 中栏"命中轨迹"不再因多能力重复计数；"N片段"= 真实命中 slice 数。
- 右栏 detail：`currentHits()` 前端仍会把多个能力的 caps 合到同一 traj，但现在它们指向同一份 hit（同 spans），前端渲染时同一 step 仍可能贴多个能力色标签——**这是符合预期的**：一个 slice 确实被判定为演示了这几个能力（judge 是按 sub_problem 整体判的）。关键改善是**不再虚高计数、不再把 miss 当 hit**。

> **诚实标注**：judge 的判定粒度本就是"这个 slice 是否演示了该 sub_problem 的目标能力（整组）"，不是逐能力。所以"同一 slice 贴多个能力标签"在数据层面是**真实的**（它确实同时关联这些能力），不是 bug。问题5 的真正修复是**消除重复计数**，而非强行拆分 judge 粒度（那需要改后端算法，超出最小侵入）。若未来要真正逐能力判定，另开任务改 judge/rerank。

**测试**（test_viewmodel.py 新增/改）：
- `test_hit_count_excludes_judge_miss`：构造一个 judge_match=False 的候选，断言不进 hit_trajectories、hit_count 不含它。
- `test_empty_spans_excluded`：judge_match=True 但 spans 空 → 不算命中。
- `test_multi_capability_no_double_count`：一个 sub_problem 两个能力、一个命中 slice，断言两个能力的 hit_count 都是 1（而非各自重复），且指向同一命中。
- 现有 `test_view_maps_problems_and_capabilities` 需同步更新（原来 FakeScored 没设 judge_match，默认 True；确认它有 spans）。

---

### 问题10 [Minor] modal 路由永远 pass（viewmodel.py）

**根因**：只有 route=pass 的 sub_problem 才进 view，所以 modal "路由"行恒显示 pass。

**修法**：viewmodel 的 compile 字段**去掉 route**（冗余），或前端 modal 不显示 route 行。选**后端去掉 route**——`compile` dict 不再放 route，前端 openCompileModal 的"置信度/路由"行改为只显示置信度 + origin。改动更集中。
- viewmodel `compile` 去掉 `"route"`。
- app.js openCompileModal：`row("置信度 / 来源", <code>conf</code> · origin)`，去掉 route。

**测试**：更新 test_viewmodel 里 compile 字段断言（去掉 route 断言）。

---

### 问题11 [Minor] viewmodel.py 用 sp["id"] 硬索引

**根因**：其余字段都 `.get`，`id` 单独用 `sp["id"]`，缺 id 会 KeyError。

**修法**：`sp_id = sp.get("id")`；若为 None 则 `continue` 跳过该 sub_problem（公共契约入口防御）。

**测试**：`test_subproblem_without_id_skipped`：spec 里塞一个无 id 的 sub_problem，断言不崩、被跳过。

---

### 问题6 [Minor] on_progress 无异常隔离（pipeline.py）

**根因**：`on_progress(done, total)` 若抛异常会中断整个 judge 阶段、丢弃已累积的 all_scored。

**修法**：包 try/except：
```python
if on_progress is not None:
    try:
        on_progress(done, total)
    except Exception:  # noqa: BLE001 — progress reporting must never abort scoring
        pass
```

**测试**（test_orchestrator.py 或 test_pipeline）：`test_run_scored_survives_failing_on_progress`：传一个必抛的 on_progress，断言 run_scored 仍正常返回全部候选。（注意：这个测试在 module1 层更合适，但 module1 测试用真实依赖；可在 service 层用 FakePipeline 不适用——改为直接测 `TrajectoryPipeline.run_scored` 需要 mock judge。评估：放 tests/module1，用已有的 fake gateway/judge 夹具。）

---

## 演示数据修复

### 问题12 [Minor] tool_call id 重复（gen_demo_trajectories.py）

**根因**：`f"c{name[:2]}"` 全局只产出 4 种 id，跨步骤大量重复。reviewer 确认**当前无害**（loader 不读 tool_call id），但为将来 id 关联做防御 + 更真实。

**修法**：id 加序号。`a()` 函数签名加一个步骤计数，或用全局递增：
```python
_tc_counter = [0]
def a(content, name=None, args=None):
    m = {"role": "assistant", "content": content}
    if name:
        _tc_counter[0] += 1
        m["tool_calls"] = [{"id": f"call_{_tc_counter[0]:03d}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]
    return m
```
改后重新跑 `gen_demo_trajectories.py` 生成新 jsonl，再跑一次 slicer 校验（确认切片数不变、14 条仍合法）。

**测试**：非测试资产，跑生成脚本 + jsonl 合法性校验 + slicer 冒烟。

---

## 执行顺序与验证

**建议实施批次**（每批可独立验证/提交）：
1. **后端 Important**：问题3（app.py cancel 兜底）+ 问题9（runstore 淘汰）→ 跑 test_app + test_runstore。
2. **后端聚合**：问题5+7+10+11（viewmodel）+ 问题6（pipeline）→ 跑 test_viewmodel + test_orchestrator + tests/module1。
3. **前端**：问题1+2+4+8+13（app.js/index.html/css）→ 手动验证 + 冒烟。
4. **数据**：问题12（gen 脚本 + 重新生成）→ jsonl 校验 + slicer 冒烟。

**全量回归**：`.venv/bin/pytest -m "not requires_model" -q` 应全绿；`ruff check` clean。

**最终手动验证清单**（重启服务后）：
- [ ] 跑 A → 开轨迹 → 换文件跑 B → 开同名轨迹，内容是 B 的（问题1）
- [ ] 上传含 `<img onerror>` id 的轨迹，不弹窗（问题2）
- [ ] 运行中按钮禁用；上传瞬间点停止能停；关服务点运行显示上传失败（问题4）
- [ ] 命中轨迹"N片段"= 真实 slice 数，不再虚高（问题5/7）
- [ ] modal 无 route 行（问题10）
- [ ] 顶栏渐变、编译详情 modal 等既有功能不回归

---

## 不做的事（明确边界）

- **不改 judge/rerank 的判定粒度**（问题5 的根本"逐能力判定"）——那是后端算法变更，超出本轮"修 review 问题"范围。本方案在聚合层消除重复计数，达到 UI 正确，判定粒度维持"按 sub_problem 整组"。
- **不引入 LRU/TTL 复杂淘汰**（问题9）——V1 用简单容量上限 + FIFO 淘汰终态 run 即可。
- **不做前端自动化测试框架**——前端仍靠手动验证（无构建、原生页）。
