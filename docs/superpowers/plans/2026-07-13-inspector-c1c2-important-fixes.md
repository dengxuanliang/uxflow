# Inspector 服务 C1/C2 + Important 修复方案（第二轮 review）

> **状态**：方案定稿，待复核。
> **范围**：仅修 C1、C2、I3、I4、I5 五个问题。**严格最小侵入，不污染其他部分。**
> **来源**：第二轮全实现终审（3 个 reviewer）。
> **硬约束**：只改必要文件的必要行；不动 module0/2/3 算法；不改 create_app 的 deps 契约（避免波及所有测试）；不动 module1 的公共 loader API 语义。

---

## 决策依据（为什么这样修，而非别的）

在写具体改动前，先交代每个问题的"最小侵入边界"是怎么定的——这决定了改哪个文件、不碰哪个文件。

### C1 共享实例污染 → 选 **Semaphore(1) 串行化**，不选 per-run 工厂

两个候选：
- **(A) per-run deps 工厂**：`create_app` 的 `deps` 参数改成 `deps_factory`，每个 run 新建 compiler/pipeline。**代价**：改 create_app 契约 → 所有注入 fake deps 的测试（test_app/test_orchestrator）全要改；embedding 模型不能每次重建（太贵）需拆分共享/私有。侵入大、波及测试。
- **(B) `asyncio.Semaphore(1)` 串行化**：在 `_background_run` 执行体外包一层 `async with _run_lock`，同时只有一个 run 真正跑 pipeline。**代价**：app.py 加 ~3 行，deps 契约不变，测试不变。

**选 (B)**。理由：(1) 侵入最小、零测试波及；(2) 真实场景是"一个人检分轨迹"，串行完全够，且 pipeline 本就是 CPU/LLM 密集、并行也抢不到资源；(3) 排队语义正确——第二个 run 等第一个跑完，不会污染。**注意**：锁只包"真正跑 pipeline 的部分"，不包整个 `_background_run`（否则 cancel 一个排队中的 run 会被锁阻塞）。见下方精确实现。

### C2 阻塞事件循环 → **orchestrator 层把 run_scored 的同步部分卸载到线程**

`run_scored` 是 async（内部 await LLM judge），不能整体 `to_thread`。真正阻塞循环的是 `_build_index`（同步 embedding + 切片，无 await）。但 `_build_index` 是 `run_scored` 内部第一步，从外部无法单独卸载它而不改 pipeline。

两个候选：
- **(A) 改 pipeline.py**：把 `_build_index` 用 `await asyncio.to_thread(self._build_index, ...)` 包起来。**代价**：改 module1 的 run_scored 内部——但这是 service 引入的并发需求，module1 单元测试用 fake embedder 秒回、不受影响。侵入 module1 但很小（1 处 await 包裹）。
- **(B) 只加 Semaphore（C1 的 B）不解决 C2**：串行化不能让单个 run 的 embedding 不阻塞循环——SSE 进度在建索引期间仍冻结、cancel 仍不响应。

**C2 必须动到能让 CPU 段脱离事件循环的地方。** 最小侵入是 **(A)**：在 `pipeline.run_scored` 里把 `self._build_index(trajectory_paths)` 改成 `await asyncio.to_thread(self._build_index, trajectory_paths)`。这一行让整个建索引（同步 embedding）在线程池跑，事件循环空出来 flush SSE + 响应 cancel。judge 阶段本就是 await（不阻塞），无需动。

> **诚实标注 C2 的边界**：`_build_index` 卸到线程后，事件循环在建索引期间能跑 SSE；但 cancel 在建索引进行中仍不能"中断线程里的 encode"（Python 线程无法强制中断）——它会在 `to_thread` 返回后、下一个 await 点生效。这已是不改底层 embedding 接口的前提下能做到的最好，且解决了"进度冻结"这个主症状。judge 阶段的 cancel 响应性不受影响（那里有真 await）。

### I4 坏行炸批 → **在 loader.py 加 skip-and-count，保持 API 兼容**

坏行在 `load_trajectories` 里 `json.loads` 抛出，且 pipeline._build_index 直接调它——所以修必须在 loader。担心"污染 module1"：但这个改动是**纯加固、向后兼容**——合法输入行为完全不变，只是坏行从"抛异常炸全部"变成"跳过并计数"。这不改变 API 签名、不改变合法数据的返回。module1 现有测试（用干净 fixture）不受影响。这是 loader 应有的健壮性，不算污染。

**边界**：只改 `load_trajectories` 的循环（try/except 单行），不动 `parse_trajectory`、不加新参数、不改返回类型。

### I3 非 UTF-8 manifest 500 → **app.py create_run 包 decode**

纯 service 层，改 `create_run` 的 manifest decode 一处，非法→400。零波及。

### I5 loadView 未 generation 保护 → **app.js subscribeEvents 传 myGen 给 loadView**

纯前端，改 `subscribeEvents`/`loadView` 加 generation 守卫。零后端影响。

---

## 改动清单

| # | 文件 | 改动 | 侵入范围 |
|---|---|---|---|
| C1 | `src/service/app.py` | 加 `_run_lock = asyncio.Semaphore(1)`，`_background_run` 里只把"跑 run_fn"部分包进 `async with _run_lock` | +~4 行，契约不变 |
| C2 | `src/module1/pipeline.py` | `run_scored` 里 `self._build_index(...)` → `await asyncio.to_thread(self._build_index, ...)`；顶部 `import asyncio` | +1 import, 改 1 行 |
| I3 | `src/service/app.py` | `create_run` 的 manifest decode 包 try→400 | +~4 行 |
| I4 | `src/module1/loader.py` | `load_trajectories` 循环里 `json.loads` 包 try/except，坏行跳过 | 改 ~4 行 |
| I5 | `src/service/web/app.js` | `subscribeEvents` 捕获 myGen 传给 `loadView`；`loadView` 加 generation 守卫 | 改 ~4 行 |
| — | 测试 | test_app 加 I3 + C1 串行；test_loader 加 I4；（C2/I5 无自动化测试，手动/冒烟） | 加法 |

---

## 精确实现

### C1 — app.py：Semaphore 串行化 pipeline 执行

在 `create_app` 里、`_tasks` 附近加锁：
```python
    # Track in-flight background tasks so a run can be cancelled (see /cancel).
    _tasks: dict[str, asyncio.Task] = {}
    # Serialize actual pipeline execution: the injected compiler/pipeline hold
    # per-run mutable state (self._store is reset each run_scored), so concurrent
    # runs would corrupt each other. A run waits its turn rather than interleave.
    _run_lock = asyncio.Semaphore(1)
```

`_background_run` 里，**只把跑 run_fn 的部分**包进锁（保持 emit/set_view/mark 在锁内，temp 清理在 finally 锁外）：
```python
    async def _background_run(run_id: str, manifest_text: str, traj_path: pathlib.Path):
        def emit(ev: dict) -> None:
            store.append_event(run_id, ev)
        try:
            async with _run_lock:      # serialize pipeline execution (C1)
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
        except asyncio.CancelledError:
            store.append_event(run_id, {"stage": "done", "status": "cancelled",
                                        "msg": "已停止"})
            store.mark_error(run_id)
        except Exception as exc:  # noqa: BLE001
            store.append_event(run_id, {"stage": "done", "status": "error",
                                        "msg": str(exc)})
            store.mark_error(run_id)
        finally:
            traj_path.unlink(missing_ok=True)
            _tasks.pop(run_id, None)
```
**关键点**：
- 锁只包 run_fn 执行段。`except`/`finally` 在锁外——所以一个排队中（等锁）的 run 被 cancel 时，`CancelledError` 在 `async with _run_lock`（await 获取锁）处抛出，正常进 except 分支标记 cancelled、finally 清理。**排队中的 cancel 立即生效，不被阻塞。**
- 一个正在跑的 run 被 cancel → CancelledError 在 run_fn 内的 await 点抛出 → 传到这里的 except → 锁通过 `async with` 自动释放 → 下一个排队 run 拿到锁。**锁不会因 cancel 泄漏。**

> **排队 UX（方案自审发现）**：串行化后，排队的第二个 run 在拿到锁前不发任何事件，前端会静默冻在"上传中/module0"。**决策：在 `async with _run_lock` 之前 emit 一个排队提示**，让用户知道在排队而非卡死：
> ```python
>         try:
>             if _run_lock.locked():
>                 emit({"stage": "module0", "status": "running",
>                       "msg": "前一个任务运行中，排队等待…"})
>             async with _run_lock:
>                 ...
> ```
> 这一行零额外契约（复用现有 module0 事件形状，前端 handleEvent 原样显示 msg），成本极低，消除"冻死"观感。`_run_lock.locked()` 判断当前是否被占用——被占用才提示排队。

**测试**（test_app.py 加）：`test_two_concurrent_runs_serialized`——用一个记录"进入/离开"时刻的 run_fn，同时发两个 POST，断言第二个的"进入"在第一个"离开"之后（不重叠）。

### C2 — pipeline.py：卸载 _build_index 到线程

顶部加 `import asyncio`（确认现有 imports；pipeline.py 目前无 asyncio import）。
`run_scored` 里：
```python
        self._store = MemoryIndex()
        self._traj_paths.clear()
        # Offload the synchronous, CPU-bound index build (slicing + embedding)
        # to a thread so it doesn't block the event loop — keeps SSE progress
        # flushing and makes cancellation responsive at the thread boundary (C2).
        await asyncio.to_thread(self._build_index, trajectory_paths)
        if self._store.size == 0:
            return []
```
只改这一行（`self._build_index(trajectory_paths)` → `await asyncio.to_thread(...)`）+ 加 import。judge 阶段不动。

> **线程安全核对**：`_build_index` 内部只写 `self._store`（新建的 MemoryIndex）和 `self._traj_paths`（刚 clear 过），不与其他协程共享——因为 C1 的 Semaphore 保证同时只有一个 run 在跑，所以 `to_thread` 里的 `self.` 写入无并发竞争。**C1 和 C2 组合是安全的**：C1 保证单 run 独占实例，C2 让那个 run 的 CPU 段离开事件循环。

**测试**：C2 无法用 fake 测（fake embedder 秒回、不阻塞）。手动验证：真实跑时观察 SSE 进度在建索引期间不再冻结。方案不加自动化测试（诚实标注：这是真实模型才可验的行为）。

### I3 — app.py：manifest decode 容错

`create_run` 开头：
```python
    @app.post("/runs")
    async def create_run(manifest: UploadFile, trajectories: UploadFile):
        raw = await manifest.read()
        try:
            manifest_text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="用户清单必须是 UTF-8 编码的文本文件") from None
        traj_bytes = await trajectories.read()
        ...
```
（`HTTPException` 已 import。decode 在 temp 文件创建之前，非法时直接 400，无 temp 泄漏。）

**测试**（test_app.py 加）：`test_non_utf8_manifest_returns_400`——POST 一个 `b"\xff\xfe..."` 的 manifest，断言 400（而非 500）。

### I4 — loader.py：坏行跳过

`load_trajectories` 循环：
```python
    trajectories = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed JSON rather than abort the whole file
            if not isinstance(data, dict):
                continue  # valid JSON but not a trajectory object (null/42/[..]/"s")
            trajectories.append(parse_trajectory(data))
    return trajectories
```
**保持**：合法输入行为完全不变；无新参数；返回类型不变。只是坏行不再炸全部。

> **必修点（方案自审发现）**：只 catch `json.JSONDecodeError` 不够——合法 JSON 但非 dict 的行（`null`、`42`、`[1,2,3]`、`"foo"`）能过 `json.loads`，随后在 `parse_trajectory` 的 `data.get(...)` 抛 `AttributeError`，照样炸全批。因此必须加 `isinstance(data, dict)` 守卫。测试要覆盖**非 dict 行**（不只是坏 JSON 行）。

> **可选增强（本方案不做，避免扩范围）**：把跳过的行数记录/返回。当前只静默跳过——因为返回额外计数会改 API 签名（污染）。若你要"跳了几行"的可观测性，那属于 I4 的可选项，可另议。

**测试**（tests/module1/test_loader.py 加，或新建）：`test_malformed_line_skipped`——一个 4 行 jsonl：行1 合法轨迹、行2 坏 JSON（`}{`）、行3 合法 JSON 但非 dict（`[1,2,3]`）、行4 合法轨迹，断言返回 2 条合法轨迹（不抛异常）。**必须含非 dict 行**以锁住 isinstance 守卫。

### I5 — app.js：loadView generation 守卫

`subscribeEvents` 捕获当前 gen，done 时传给 loadView：
```javascript
function subscribeEvents(runId) {
  if (state.es) { state.es.close(); state.es = null; }
  const myGen = state.runGen;          // capture this run's generation (I5)
  const es = new EventSource(`/runs/${runId}/events`);
  state.es = es;
  es.onmessage = async (e) => {
    const ev = JSON.parse(e.data);
    if (myGen !== state.runGen) { es.close(); return; }  // superseded → ignore
    handleEvent(ev);
    if (ev.stage === "done") {
      es.close();
      stopTimer();
      $("stop").style.display = "none";
      $("run").disabled = false;
      if (ev.status === "error" || ev.status === "cancelled") {
        setMsg((ev.status === "cancelled" ? "已停止" : "运行出错: ") + (ev.msg || ""));
        return;
      }
      await loadView(runId, myGen);
    }
  };
  es.onerror = () => { ... };  // unchanged
}
```
`loadView` 加守卫：
```javascript
async function loadView(runId, myGen) {
  const resp = await fetch(`/runs/${runId}/view`);
  if (myGen !== state.runGen) return;   // a newer run started during the fetch — drop stale view (I5)
  state.view = await resp.json();
  ...
}
```
**效果**：用户在 /view 拉取窗口内点了新运行 → `state.runGen` 已被新 startRun bump → 旧 loadView 拿到响应后检测 `myGen !== state.runGen` → 直接 return，不把旧数据渲染盖到新 run。

**测试**：前端无自动化测试，`node -c` 语法校验 + 手动验证。

---

## 实施批次

1. **后端并发核心**：C1（app.py Semaphore）+ C2（pipeline to_thread）→ 跑 test_app + test_orchestrator + tests/module1，确认串行测试通过、其余无回归。
2. **后端健壮性**：I3（app.py decode）+ I4（loader skip）→ 跑 test_app + tests/module1。
3. **前端**：I5（app.js）→ node -c + 手动。

每批一个 subagent implementer + review。

## 验证

- `.venv/bin/pytest -m "not requires_model" -q` 全绿（现有 311 + 新增）。
- `.venv/bin/ruff check` 改动文件 clean。
- `node -c src/service/web/app.js` 通过。
- 手动（重启服务）：多 run 串行不串数据（C1）、真实跑时进度不冻结（C2）、上传 GBK txt 得 400 提示（I3）、含坏行的 jsonl 仍能跑出结果（I4）、/view 窗口内重运行不显示旧数据（I5）。

## 明确不做（边界）

- **不改 create_app 的 deps 契约**（不用 per-run 工厂）——用 Semaphore 达到 C1 目的，零测试波及。
- **不动 module0/2/3 算法**。
- **不改 loader API 签名**（I4 只加 try/except，不返回跳过计数）。
- **不修其余 Minor/优化点**（M6-M13、日志、max_runs 背压、on_event→lifespan 等）——本方案严格限定在 C1/C2/I3/I4/I5。
- **C2 不追求"线程内 encode 可中断"**——Python 线程无法强制中断；本方案解决"进度冻结"主症状，cancel 在 to_thread 边界生效。
