# Inspector 搜索/入库 双功能与持久化 实现方案

> **状态**：方案定稿，待复核。
> **范围**：实现 `2026-07-14-inspector-search-ingest-persistence-design.md` spec。把 Inspector 从"单次上传→全量重跑→内存态"改造成 **读写分离双功能**：`/search`（只读检索）+ `/ingest`（持久化入库），共享 `uxflow.db`。
> **硬约束**：module0/1/2/3 **内部算法零改动**；只在 pipeline 外围加入口、在 service 层加端点、加 3 个 SQLite Store。所有 Store 复用现有 `SqliteSliceStore` 的连接写法（`isolation_level=None` + WAL + autocommit）。现有 `/runs` 端点与 `MemoryRunStore` **保留不动**（向后兼容，老前端/测试不破）。
> **来源**：spec §2 决策 + §12 D1–D4 定稿。

---

## 决策依据（已在 spec 定稿，这里只记实现边界）

- **D1 problem_id** = normalized sha1[:16]（strip+折叠空白+NFC，不 lowercase）。hash 挡逐字重复、τ_q 挡语义重复，互补。
- **D2 新问题 LRU**：`get_or_compile` 收口，顺序 `持久 nearest → 进程 LRU → compile`。纯性能层，不碰去重语义。
- **D3 judge miss 拆分**：pipeline 层过滤，只把 miss 传给现有 `judge_batch`，cached 项经 `dict→JudgeResult` 适配器**保序**合并。不碰 module1 内部。
- **D4 ingest 失败**：跨文件不回滚 + 清单逐行 try/except 隔离，结尾报汇总。

**核对过的硬约束**（源码已验）：
- `rerank(hits, judge_results, ...)` 要求两列表**等长同序**（`zip` + 长度校验）→ D3 合并必须保序，且 `hits` 与合并后的 `judge_results` 一一对应。
- `judge_batch(slices=...) → list[JudgeResult]` 顺序与输入一致。
- `JudgeResult(match: bool, confidence: float, spans: list[dict], reasoning: str)` → 适配器目标形状。
- `TrajectoryPipeline(store_factory=...)` seam 已存在，注入 `SqliteSliceStore` 即持久化，pipeline 内部零改。
- `Embedder.embed(text) -> list[float]`（`uxflow_embed` 协议）+ `EmbeddingModel` 实例已在 `inspector_serve.py` 构建。

---

## 改动清单

| # | 文件 | 改动 | 侵入 |
|---|------|------|------|
| S1 | `src/module1/sqlite_store.py` 或新 `src/service/stores.py` | 3 个新 Store：`SqliteProblemStore` / `SqliteTrajectoryStore` / `SqliteJudgeCache` + 各自 Protocol | 新增文件 |
| S2 | `src/service/dedup.py`（新） | `normalize_question` + `problem_id_of` + `cosine` + `get_or_compile`（LRU） | 新增 |
| S3 | `src/module1/pipeline.py` | 加 `ingest_trajectories(paths)` + `search(specs, *, judge_cache)`；`_build_index` 增写 TrajectoryStore（可选回调） | 加方法，不改现有 |
| S4 | `src/service/orchestrator.py` | 加 `run_search` + `run_ingest`；`PipelineDeps` 加 3 个 store 字段 | 加函数 |
| S5 | `src/service/app.py` | 加 `POST /search`、`POST /ingest`、`GET /stats`；`/runs` 保留 | 加端点 |
| S6 | `src/service/web/{index.html,app.js,style.css}` | header 双模式（搜索框+入库区+状态条）；结果三栏渲染复用 | 前端 |
| S7 | `scripts/inspector_serve.py` | 接线：`resolve_db_path` + 注入 4 store + `store_factory` | 改 build_app |
| S8 | `scripts/calibrate_question_dedup.py`（新） | τ_q 校准脚本（仿 calibrate_mount_threshold） | 新增，非阻塞 |
| — | 测试 | 每个 Store 单测 + pipeline ingest/search 单测 + orchestrator + app 端点 | 加法 |

**关键：`sub_problem_id` 前缀改造**——现状 orchestrator 用 `L{行号}.{sp.id}`，跨 run 会重复导致 JudgeCache 撞键。改为 `{problem_id}.{sp.id}`（见 S4）。这是持久化正确性的**必修点**。

---

## 精确实现

### S1 — 三个 SQLite Store（新 `src/service/stores.py`）

复用 `sqlite_store.py` 的 `_emb_to_blob`/`_blob_to_emb`、连接 pragma。每个 Store 一个 Protocol + 一个 Sqlite 实现。

**`SqliteProblemStore`**（spec §5.1）：
```python
_SCHEMA_PROBLEMS = """
CREATE TABLE IF NOT EXISTS problems (
    problem_id TEXT PRIMARY KEY,
    raw_question TEXT NOT NULL,
    question_embedding BLOB,
    spec_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);"""

class SqliteProblemStore:
    def __init__(self, db_path):
        # 同 SqliteSliceStore: connect(isolation_level=None) + WAL + busy_timeout
        # 启动载入 (problem_id, embedding) 到内存镜像供 nearest 线性扫描
        self._emb_mirror: list[tuple[str, np.ndarray]] = []
        self._load()

    def add(self, raw_question, embedding, spec, *, created_at) -> str:
        pid = problem_id_of(raw_question)       # D1: normalized sha1[:16]
        self._conn.execute("INSERT OR REPLACE INTO problems VALUES (?,?,?,?,?)",
                           (pid, raw_question, _emb_to_blob(embedding),
                            json.dumps(spec, ensure_ascii=False), created_at))
        self._mirror_upsert(pid, embedding)
        return pid

    def nearest(self, embedding):
        # 线性扫内存镜像算 cosine，返回 (pid, cosine, spec) 或 None
        # 空库返回 None；调用方比 τ_q 判命中
        ...

    def get(self, problem_id): ...   # 读单条 spec_json
    def count(self): ...
```
> **nearest 实现**：V1 问题量级小（人工检分场景），**线性扫描内存镜像**足够，不引向量索引。镜像存 `np.ndarray` 预归一化，cosine = 点积。空库/零向量返回 None。

**`SqliteTrajectoryStore`**（spec §5.2）：`upsert(traj_id, steps, source_path, created_at)` / `get(traj_id) -> {"trajectory_id", "steps"}` / `count()`。`steps` 用 `viewmodel._step_to_dict` 同构 json。`INSERT OR REPLACE` 幂等。

**`SqliteJudgeCache`**（spec §5.3）：
```python
def get(self, sub_problem_id, trajectory_id, slice_index) -> dict | None:
    # 返回 {"match": bool, "confidence": float, "spans": [...]} 或 None
def put(self, sub_problem_id, trajectory_id, slice_index, verdict, *, created_at):
    # INSERT OR REPLACE; verdict = {"match","confidence","spans"}
```
PK `(sub_problem_id, trajectory_id, slice_index)`。**无失效逻辑**（spec §5.3 成对稳定性）。

**测试**（`tests/service/test_stores.py` 新）：
- ProblemStore：add→get 往返；nearest 空库返 None；nearest 命中返最高 cosine；重复 add 同问题幂等（同 pid 覆盖）；重开连接后数据还在（持久性）。
- TrajectoryStore：upsert→get steps 往返；重复 upsert 幂等；重开持久。
- JudgeCache：put→get 往返；miss 返 None；重开持久；PK 三元组区分不同 slice。

### S2 — dedup 工具（新 `src/service/dedup.py`）

```python
import hashlib, unicodedata, re
import numpy as np

def normalize_question(q: str) -> str:
    q = unicodedata.normalize("NFC", q).strip()
    return re.sub(r"\s+", " ", q)            # 折叠内部空白；不 lowercase (D1)

def problem_id_of(q: str) -> str:
    return hashlib.sha1(normalize_question(q).encode("utf-8")).hexdigest()[:16]

def cosine(a, b) -> float:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0

class ProblemCompiler:
    """get_or_compile 收口: 持久 nearest → 进程 LRU → compile (D2)."""
    def __init__(self, compiler, problem_store, embedder, *, tau_q, lru_cap=128):
        self._lru = OrderedDict()   # question_norm → spec
        ...
    async def get_or_compile(self, question):
        # 1. embed + problem_store.nearest ≥ τ_q → (spec, dedup_info)  [持久命中]
        # 2. LRU 命中 → (spec, None)                                   [进程命中]
        # 3. compile → 存 LRU → (spec, None)                           [新算]
        # 注意: 只读, 绝不写 problem_store (决策1)
```
> **LRU 与去重的关系**：`get_or_compile` 是**只读**路径的编译收口。持久 nearest 永远排第一位——LRU 不能替代它。`/ingest` 清单子通路**不走** `get_or_compile`（它要写库），单独实现 embed→nearest→add 逻辑，但可共享 LRU 写入（compile 后填 LRU）。

**测试**（`tests/service/test_dedup.py`）：normalize 折叠空白/NFC/保留大小写；problem_id 稳定性（同问题同 id、改大小写变 id）；cosine 边界（零向量→0）；ProblemCompiler 三级命中顺序（持久优先于 LRU 优先于 compile）用 fake。

### S3 — pipeline 加两入口（`src/module1/pipeline.py`）

**不改** `run` / `run_scored` / `_process_sub_problem`。新增：

```python
def ingest_trajectories(self, trajectory_paths, *, on_trajectory=None):
    """写路径: build_index 到持久 store, 不 reset。可选回调把全文交给 TrajectoryStore。"""
    for path in trajectory_paths:
        path = pathlib.Path(path)
        trajectories = load_trajectories(path)
        for traj in trajectories:
            self._traj_paths[traj.id] = str(path)
            if on_trajectory is not None:
                on_trajectory(traj, str(path))     # orchestrator 写 TrajectoryStore
            for sl in slice_trajectory(traj):
                sig = extract_signature(sl, embedding_model=self._config.embedding_model)
                self._store.add(sig)               # SqliteSliceStore.add → upsert 持久
                self._store.set_slice_source(sl.trajectory_id, sl.slice_index, sl)
    # 无 _reset_store()! 增量入库

async def search(self, *, problem_specs, judge_cache, on_progress=None):
    """读路径: 对已持久 self._store 召回+精判(带缓存)+score。不 reset/不 build。"""
    all_scored = []
    total = sum(len(s.get("sub_problems", [])) for s in problem_specs)
    done = 0
    for spec in problem_specs:
        for sp in spec.get("sub_problems", []):
            all_scored.extend(await self._score_sub_problem_cached(sp, judge_cache))
            done += 1
            if on_progress: on_progress(done, total)
    all_scored.sort(key=lambda c: c.relevance_score, reverse=True)
    return all_scored
```

**`_score_sub_problem_cached`**（D3 核心，新方法，不改原 `_score_sub_problem`）：
```python
async def _score_sub_problem_cached(self, sub_problem, judge_cache):
    sp_id = sub_problem.get("id", "unknown")
    # recall (同 _score_sub_problem)
    hits = self._store.recall(...)
    kept_hits, slices = [], []
    for hit in hits:
        sl = self._store.get_slice(hit.signature.trajectory_id, hit.signature.slice_index)
        if sl: kept_hits.append(hit); slices.append(sl)
    if not slices: return []

    # D3: 拆 cached / miss，保序
    verdicts = [None] * len(kept_hits)
    miss_idx, miss_slices = [], []
    for i, hit in enumerate(kept_hits):
        cached = judge_cache.get(sp_id, hit.signature.trajectory_id, hit.signature.slice_index)
        if cached is not None:
            verdicts[i] = _dict_to_judge_result(cached)   # 适配器
        else:
            miss_idx.append(i); miss_slices.append(slices[i])

    # 只对 miss 调 judge_batch
    if miss_slices:
        miss_results = await self._judge.judge_batch(
            slices=miss_slices,
            target_capability=sub_problem.get("target_capability", []),
            trajectory_signal=sub_problem.get("trajectory_signal", ""))
        for j, i in enumerate(miss_idx):
            jr = miss_results[j]
            verdicts[i] = jr
            judge_cache.put(sp_id, kept_hits[i].signature.trajectory_id,
                            kept_hits[i].signature.slice_index,
                            {"match": jr.match, "confidence": jr.confidence, "spans": jr.spans},
                            created_at=self._now())   # now 注入, 见下

    # update_labels 只对 miss 的 match 项 (D3: cached 命中跳过, 首次已写)
    for j, i in enumerate(miss_idx):
        if verdicts[i].match:
            self._store.update_labels(kept_hits[i].signature.trajectory_id,
                                      kept_hits[i].signature.slice_index,
                                      sub_problem.get("target_capability", []))

    # verdicts 现与 kept_hits 严格同序、无 None → 喂 rerank
    scored = rerank(kept_hits, verdicts, sub_problem, trajectory_path="")
    for sc in scored:
        sc.trajectory_path = self._traj_paths.get(sc.trajectory_id, "")
    return scored
```
适配器（模块内私有）：
```python
def _dict_to_judge_result(d):
    from module1.judge import JudgeResult
    return JudgeResult(match=d["match"], confidence=d["confidence"],
                       spans=d["spans"], reasoning="cached")
```
> **保序不变量**（rerank 硬约束）：`verdicts` 按 `kept_hits` 的索引原地填充，cached 与 miss 都写回**同一位置**，最终 `len(verdicts)==len(kept_hits)` 且一一对应。miss_slices 传给 judge_batch 后返回顺序与 miss_slices 一致，按 `miss_idx` 映射回原位。**这是 D3 正确性的核心，测试必须锁。**
> **时间注入**：`judge_cache.put` 需 `created_at`。pipeline 不该调 `datetime.now()`（对齐仓库"纯层不碰时钟"约定）。方案：`search()` 接一个 `now_fn` 参数（默认由 orchestrator 注入 ISO 串），`_score_sub_problem_cached` 用它。**或**让 JudgeCache.put 内部用 `CURRENT_TIMESTAMP` SQL 默认——更简单，倾向后者，pipeline 完全不碰时间。

**测试**（`tests/module1/test_pipeline_search_ingest.py`）：
- `ingest_trajectories` 写 SqliteSliceStore 后 `size>0`，重开 store 数据在，重复 ingest 不翻倍（upsert）。
- `search` 全 miss：judge_cache 从空到满，调用 judge_batch 一次。
- `search` 全 cached：judge_batch **零调用**（用 spy fake judge 断言 call_count==0），结果与全 miss 一致。
- **保序**：构造 5 hit、3 cached 2 miss，断言 rerank 收到的 verdicts 顺序与 hits 对齐（cached 项 confidence 用哨兵值验证落在正确位置）。

### S4 — orchestrator 两入口（`src/service/orchestrator.py`）

`PipelineDeps` 加字段：
```python
@dataclass
class PipelineDeps:
    compiler: Any
    pipeline: Any
    select_fn: Callable
    load_trajectories_fn: Callable
    problem_store: Any = None      # 新
    trajectory_store: Any = None   # 新
    judge_cache: Any = None        # 新
    embedder: Any = None           # 新 (embed 去重用)
```
> 全部默认 None → **现有 `run_pipeline` 与所有旧测试不受影响**（它们不传新字段）。

**`run_search`**：
```python
async def run_search(question, *, deps, emit, run_id, tau_q, selection_config=None, general_config=None):
    emit({"stage": "search0", "status": "running", "msg": "分析问题...", "index": 0, "total": 1})  # 3阶段(定案3)
    emb = deps.embedder.embed(question)
    hit = deps.problem_store.nearest(emb)
    dedup = None
    if hit and hit[1] >= tau_q:
        pid, sim, spec = hit
        spec_lines = [spec]                          # spec_json 已是编译产物
        dedup = {"matched_problem_id": pid,
                 "matched_question": deps.problem_store.get(pid)["raw_question"],
                 "similarity": sim}
        emit({"stage": "search0", "status": "running", "msg": f"命中已有问题 (相似度 {sim:.2f})"})
    else:
        spec_obj = await deps.compiler.compile(question)     # 不写 problem_store!
        spec = _spec_to_dict(spec_obj, id_prefix=f"{problem_id_of(question)}.")  # D1 前缀
        spec_lines = [spec]
    # search1 检索: 缓存召回精判
    all_sub = [sp for s in spec_lines for sp in s["sub_problems"]]
    scored = await deps.pipeline.search(
        problem_specs=spec_lines, judge_cache=deps.judge_cache,
        on_progress=lambda d,t: emit({"stage":"search1","status":"running","msg":f"检索 {d}/{t}","index":d,"total":t}))
    select_result = deps.select_fn(scored, sub_problem_ids=[sp["id"] for sp in all_sub], ...)
    view = build_inspector_view(run_id=run_id, spec=_merged(spec_lines), scored=scored, select_result=select_result)
    view["mode"] = "search"; view["dedup"] = dedup
    trajectories = {}   # 详情走 TrajectoryStore, 不内联 (见 app /trajectory)
    return view, trajectories
```
> **sub_problem_id 前缀 = `{problem_id}.`**（命中已有问题时，spec_json 里已含带前缀的 id，直接用；新问题用 `problem_id_of(question).`）。这保证 JudgeCache 键跨 run 稳定。**必修点。**

**`run_ingest`**：
```python
async def run_ingest(*, manifest_lines=None, trajectory_path=None, deps, emit, run_id, tau_q):
    added = skipped = failed = 0
    if trajectory_path:
        emit({"stage":"ingest_traj","status":"running","msg":"切片+签名+写库..."})
        def on_traj(traj, path):
            deps.trajectory_store.upsert(traj.id, [_step_to_dict(s) for s in traj.steps],
                                         source_path=path, created_at=None)
        deps.pipeline.ingest_trajectories([trajectory_path], on_trajectory=on_traj)
    if manifest_lines:
        for i, line in enumerate(l.strip() for l in manifest_lines):
            if not line: continue
            try:
                emb = deps.embedder.embed(line)
                hit = deps.problem_store.nearest(emb)
                if hit and hit[1] >= tau_q:
                    skipped += 1
                    emit({"stage":"ingest_manifest","status":"running","msg":f"跳过重复: {line[:30]}"})
                    continue
                spec_obj = await deps.compiler.compile(line)     # D4: 逐行 try/except
                spec = _spec_to_dict(spec_obj, id_prefix=f"{problem_id_of(line)}.")
                deps.problem_store.add(line, emb, spec, created_at=None)
                added += 1
                emit({"stage":"ingest_manifest","status":"running","msg":f"入库: {line[:30]}"})
            except Exception as exc:   # noqa: BLE001 — D4 逐行隔离
                failed += 1
                emit({"stage":"ingest_manifest","status":"running","msg":f"第{i+1}行失败: {exc}"})
    emit({"stage":"done","status":"ok","msg":f"入库 {added} 问题, 跳过 {skipped}, 失败 {failed}"})
    view = {"mode":"ingest","summary":{"added":added,"skipped_dup":skipped,"failed":failed}}
    return view, {}
```
> `created_at=None` → Store 内部用 SQL `CURRENT_TIMESTAMP`（承接 S3 时间注入决策，service 层也不碰时钟）。

**测试**（`tests/service/test_orchestrator_search_ingest.py`，全 fake）：
- run_search 命中已有：fake problem_store.nearest 返高 cosine → view.dedup 非空、**compiler.compile 零调用**。
- run_search 新问题：nearest 返 None → compile 被调 1 次、**problem_store.add 零调用**（只读！）。
- run_ingest 清单逐行隔离：3 行，第 2 行 compiler 抛错 → added==2, failed==1, 不中断。
- run_ingest 去重：第 2 行 nearest 命中 → skipped==1, compile 未对该行调用。
- run_ingest 仅轨迹 / 仅清单 / 两者 三种组合都跑通。

### S5 — app.py 三端点（保留 `/runs`）

```python
@app.post("/search")
async def search(payload: dict):           # {"question": str}
    q = (payload.get("question") or "").strip()
    if not q: raise HTTPException(400, "问题不能为空")
    run_id = store.create()
    task = asyncio.create_task(_bg_search(run_id, q))   # 复用 _run_lock 串行
    ...
    return {"run_id": run_id}

@app.post("/ingest")
async def ingest(manifest: UploadFile = None, trajectories: UploadFile = None):
    if manifest is None and trajectories is None:
        raise HTTPException(400, "至少上传一个文件")    # D4 前提
    # manifest decode 容错 (复用 I3 模式); trajectories 存 temp
    run_id = store.create()
    task = asyncio.create_task(_bg_ingest(run_id, manifest_text, traj_path))
    return {"run_id": run_id}

@app.get("/stats")
def stats():
    return {"problems": deps.problem_store.count(),
            "trajectories": deps.trajectory_store.count(),
            "signatures": getattr(deps.pipeline._store, "size", 0)}
```
`_bg_search` / `_bg_ingest` 复刻现有 `_background_run` 的结构（`_run_lock` 串行、try/except/finally、temp 清理、`_finalize` 回调），只是调 `run_search`/`run_ingest`。`GET /runs/{id}/view`、`/events`、`/trajectory/{id}` **原样复用**——search/ingest 都写进同一 `store`。

> **`/trajectory/{id}` 改造**：现状从 `run.trajectories` 内存取。持久化后，search 返回的 `trajectories={}`（不内联）。改 `get_trajectory` **回退查 `deps.trajectory_store`**：先查 run 内存（老 `/runs` 路径兼容），miss 再查持久 store。这样 search 结果的详情栏能跨重启读。

**测试**（`tests/service/test_app_search_ingest.py`）：
- POST /search 空问题 → 400；正常 → run_id，SSE 到 done，/view 返 mode=search。
- POST /ingest 两文件都缺 → 400；仅清单 / 仅轨迹 → 各自跑通。
- GET /stats 返三计数。
- /trajectory/{id} 在 search 后能从持久 store 取到全文。

### S6 — 前端双模式

`index.html` header 替换上传区为两行 + 状态条（spec §9）：
```html
<div class="mode-search">
  <input id="question" placeholder="输入新用户问题…">
  <button id="btn-search">🔍 搜索</button>
</div>
<div class="mode-ingest">
  <label class="file-slot">用户清单.txt <input type="file" id="manifest" accept=".txt"></label>
  <label class="file-slot">回流轨迹.jsonl <input type="file" id="trajectories" accept=".jsonl"></label>
  <button id="btn-ingest">📥 入库</button>
</div>
<div id="stats-bar" class="stats-bar"></div>
```
`app.js`：
- `startSearch()`：POST /search {question} → 复用现有 `subscribeEvents`/`loadView`（含 I5 generation 守卫）。
- `startIngest()`：POST /ingest FormData(可只含一个文件) → subscribeEvents；done 后刷新 stats、飘汇总。
- **stepper 按 mode 切换结构（定案3）**：`handleEvent` 识别事件 stage 前缀——`search0/search1` 走 3 节点 stepper（分析→检索→完成），`module0..3` 走原 5 节点，`ingest_traj/ingest_manifest` 走 2 节点（写库→完成）。抽一个 `setStepper(mode)` 在 run 启动时重建节点。
- `loadView` 处理 `view.dedup`：非空则顶部飘「≈ 与已有问题重复：{matched_question}（相似度 x.xx）」。
- `loadStats()`：GET /stats → 渲染状态条；页面加载 + 每次 ingest done 后调。
- 结果三栏渲染（`renderProblems`/`renderHits`/`renderDetail`）**完全不动**。
- ingest done 的 view 无 problems → 前端识别 `view.mode==="ingest"` 时只更新状态条+汇总，不清空/不渲染三栏。

`style.css`：加 `.mode-search`/`.mode-ingest`/`.stats-bar`/`.dedup-banner` 样式，复用现有配色变量。

**测试**：`node -c src/service/web/app.js` + 手动。

### S7 — 生产线接线（`scripts/inspector_serve.py`）

```python
from uxflow_paths import resolve_db_path, ensure_parent
from service.stores import SqliteProblemStore, SqliteTrajectoryStore, SqliteJudgeCache
from module1.sqlite_store import SqliteSliceStore

db = resolve_db_path(); ensure_parent(db)
slice_store = SqliteSliceStore(db)          # 单例, 长生命周期
pipeline = TrajectoryPipeline(config=cfg, gateway=gateway,
                              store_factory=lambda: slice_store)   # 注入持久索引
deps = PipelineDeps(
    compiler=compiler, pipeline=pipeline,
    select_fn=select_final_dataset, load_trajectories_fn=load_trajectories,
    problem_store=SqliteProblemStore(db),
    trajectory_store=SqliteTrajectoryStore(db),
    judge_cache=SqliteJudgeCache(db),
    embedder=emb)
```
> **关键：`store_factory` 返回同一 slice_store 单例**（不是每次 new）。因为 search 要对**已持久、已入库**的索引召回；若每次 new，`_reset_store`（我们没调）和内存镜像都会丢。search/ingest 都操作这**同一个** SqliteSliceStore 实例。
> **τ_q**：从 `os.environ.get("UXFLOW_QUESTION_DEDUP_THRESHOLD", "0.90")` 读，传给 orchestrator。

### S8 — τ_q 校准脚本（`scripts/calibrate_question_dedup.py`，非阻塞）

仿 `calibrate_mount_threshold.py`：真实 Qwen 下取「等价问题对」与「相似但不等价对」，测 cosine 分布，打印建议 τ_q。**不阻塞主线实现**——先用默认 0.90 跑通，校准后调 env。

---

## 实施批次

> **进度**：
> - 批 1 ✅ 完成（2026-07-14）——`src/service/stores.py`（3 Store + 3 Protocol）+ `src/service/dedup.py`（normalize/problem_id_of/cosine）+ 28 新测试全绿，全量无回归，ruff clean。`ProblemCompiler`/`get_or_compile` LRU 延到批 3。
> - 批 2 ✅ 完成（2026-07-14）——`pipeline.py` 只加不改（`_dict_to_judge_result` + `ingest_trajectories` + `_score_sub_problem_cached` + `search`，现有 5 方法零改动，仅 import 扩 `JudgeResult`）+ 6 新测试全绿（含严格保序测试：探针发现召回集→隔位分 cached/miss→双哨兵逐 key 核对）。全量 391 passed 无回归，ruff clean。
> - 批 3 ✅ 完成（2026-07-14）——`orchestrator.py` 加 `run_search`/`run_ingest` + `PipelineDeps` 追加 4 默认 None 字段（前四字段顺序不变、`run_pipeline`/`_spec_to_dict` 零改动，仅 `__all__` 扩展）；`dedup.py` 加 `ProblemCompiler`（D2 只读 LRU，serialize_fn 避循环 import）+ 10 新测试。**只读铁律已锁**（新问题 search：`compiler._n==1` 且 `add_calls==[]`）；D4 逐行隔离已锁。全量 401 passed 无回归，ruff clean。
> - 批 4 ✅ 完成（2026-07-14）——`app.py` 纯新增 144 行 0 删（`create_app` 扩 `search_fn`/`ingest_fn`/`tau_q`；`_bg_search`/`_bg_ingest`/`_finalize_simple`/`_finalize_ingest` 复刻 `_background_run` 并发结构、共用 `_run_lock`；`/search`+`/ingest`+`/stats` 端点；`/trajectory` 加持久 store 回退）。核对 done-emit：run_search 无 done→`_bg_search` 补，run_ingest 有 done→`_bg_ingest` 不补，SSE 恰好一条 done。`_background_run`/`_finalize` 零改动。批 4 测试 12 + app 回归 15 全绿，全量 413 passed 无回归，ruff clean。
> - 批 5 ✅ 完成（2026-07-14）——前端双模式（index.html 加搜索框/入库区/状态条/dedup-banner，#run 隐藏保留；app.js 加 STAGE_SETS/setStepper/startSearch/startIngest/loadStats + loadView 按 mode 分支；style.css 新样式）+ `inspector_serve.py` 接线（resolve_db_path 单例 slice_store + 注入 4 store + tau_q env）。**接线暴露并修复一个真实跨线程 bug**：service SQLite 连接需 `check_same_thread=False`（FastAPI threadpool + to_thread 会跨线程访问 build_app 时创建的连接；`/stats` 也改 async）——单测/静态检查发现不了，靠端到端构造真 store + TestClient 才验出。见 memory `sqlite-store-check-same-thread-asgi`。node -c + 静态 2 passed + 端到端接线冒烟全通，全量 413 passed 无回归，ruff clean。

1. **持久层**：S1（3 Store）+ S2（dedup 工具）→ `tests/service/test_stores.py` + `test_dedup.py` 全绿。**先落地、可独立验证。**
2. **pipeline 入口**：S3（ingest_trajectories + search + _score_sub_problem_cached）→ `test_pipeline_search_ingest.py`，重点锁**保序**与**缓存零重复精判**。
3. **编排**：S4（run_search + run_ingest + PipelineDeps 扩字段）→ `test_orchestrator_search_ingest.py`，锁**只读不写库**与**逐行隔离**。
4. **服务**：S5（3 端点 + /trajectory 回退）→ `test_app_search_ingest.py`。
5. **前端 + 接线**：S6（双模式 UI）+ S7（生产线注入）→ node -c + 手动重启验证。
6. **校准**（可后置）：S8。

每批一个 implementer subagent + review。批次间有依赖（2 依赖 1，3 依赖 2…），**串行推进**。

## 验证

- `.venv/bin/pytest -m "not requires_model" -q` 全绿（现有全部 + 新增）。**现有 `/runs`、MemoryRunStore、orchestrator.run_pipeline 测试零回归**（新字段全默认 None）。
- `.venv/bin/ruff check` 改动文件 clean。
- `node -c src/service/web/app.js`。
- **手动重启冒烟**（真实 LLM+embedding）：
  1. `/ingest` 传轨迹 jsonl → `/stats` 轨迹数增 → **重启服务** → `/stats` 仍在（持久性）。
  2. `/ingest` 传清单 → 再传同一清单 → 第二次全 skipped（去重）。
  3. `/search` 一个新问题 → 三栏出结果 → 搜同一问题 → dedup banner 命中已有 + **秒出**（LRU/缓存）。
  4. `/search` 结果点开轨迹 → 详情栏全文（走 TrajectoryStore）→ 重启 → 同轨迹详情仍可读。
  5. `/search` 老问题第二次 → judge 零调用（看日志/耗时）。

## 明确不做（边界）

- **不删/不改 `/runs` 端点与 MemoryRunStore**——保留单次检分老路径，向后兼容。
- **不动 module0/1/2/3 算法**，不改 `run`/`run_scored`/`_process_sub_problem`/`_score_sub_problem`（新增 `_score_sub_problem_cached`，不改旧的）。
- **不引向量索引库**（problem nearest 线性扫内存镜像，V1 量级够）。
- **不做写时预补精判**（惰性 D3；离线 backfill 留后续 spec §11）。
- **不做问题软删除/编辑**、多写者并发、鉴权限流（spec §11 非目标）。
- **τ_q 不硬校准阻塞主线**——默认 0.90 跑通，S8 校准后调 env。

## 复核定案（2026-07-14）

1. **时间注入 = SQL `CURRENT_TIMESTAMP`**：三个 Store 的 `created_at` 列用 `DEFAULT CURRENT_TIMESTAMP`，`put`/`add`/`upsert` 不传时间。pipeline 与 orchestrator **完全不碰时钟**（比 module0.5 的 now_fn 注入更少穿参；此处无需可复现时间戳，故不必对齐 module0.5）。S3 里 `judge_cache.put` 去掉 `created_at` 入参。
2. **持久 slice_store = 单例 lambda 注入**：`store_factory=lambda: slice_store` 返回同一实例（S7）。零改 pipeline `__init__`。语义上 factory 从"每次 new"降级为"恒返单例"——**在 plan/代码注释里显式标注这层意图**，避免后人以为能靠 factory 拿到隔离 store。
3. **/search 进度 = 精简 3 阶段**（`search0 分析` → `search1 检索` → `done`），**不复用** 5 阶段 stepper。前端需按 `view.mode`/事件 stage 前缀切换 stepper 结构（见 S6）。理由：search 无"软加分/优选"语义，5 阶段文案误导。
