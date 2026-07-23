# 入选去重 · 证据折叠 · SQLite 库管理 设计

- 状态：已确认（2026-07-23 拍板），待实现
- 分支：`feat/rubric-capability-card`（沿用当前分支或另开）
- 影响面：`module3/`（去重+优选）、`service/`（viewmodel、app、新增 database 管理）、`service/web/`（前端）、`scripts/inspector_serve.py`（接线）

## 1. 背景与动机

三处独立诉求，合并为一个 spec：

1. **最终入选 SFT 样本未跨子问题去重。** 同一 `(trajectory_id, slice_index)` 被多个子问题各自选中时，`compose_dataset` 的 `targeted_count = len(targeted)` 会把它算作多条。用户预期：两个子问题各选中"同一条"轨迹 slice → 最终入选应为 **1 条**，而非 2 条。
2. **命中卡上"决定性证据"直接内联展开。** `renderHits()` 把 `⭐ 决定性证据: step X · 命中判据: …` 直接渲染在每张命中卡上，信息密度高、遮挡列表浏览。改为默认收起、点"证据"按钮就地展开。
3. **无法在运行时管理 SQLite 库。** DB 路径在 `build_app()` 启动时解析一次，4 个 store 绑死到单一连接。用户需要：列出/切换当前入库的 `.db`、清空当前库。

## 2. 已确认决策

| # | 决策点 | 结论 |
|---|--------|------|
| D1 | 诉求1 去重语义 | **数据层去重 + 合并 loss_mask 掩码**（并集），不是纯显示修复 |
| D2 | 诉求1 合并时机 | **选择前合并**：`N` 等于唯一样本数（先塌缩成唯一 slice 集，再 `select_set`） |
| D3 | 诉求2 展开方式 | **行内 accordion**：命中卡下方就地展开/收起，不弹 modal |
| D4 | 诉求3 范围 | **清除 + 运行时热切换**（不重启服务） |
| D5 | 诉求3 库发现 | **扫描 db 所在目录列出 `*.db`**；新建库=切换到不存在的路径（连接即建空库） |
| D6 | 诉求3 清除语义 | **删文件重建**（连同 `-wal`/`-shm`），前端二次确认，运行中拒绝 |

## 3. 铁律 / 护栏

- **模块依赖方向不变**：诉求1 的改动全部收敛在 `module3/` + `service/viewmodel.py`，不引入新的跨模块依赖。
- **热切换绝不 mid-run**：切换/清除库前抢 `_run_lock`；有任务在跑则拒绝（返回 409），不破坏正在执行的 pipeline 状态。
- **删文件前先释放句柄**：删除 `.db` 前 `close()` 所有相关连接，避免 Windows/NFS 句柄占用与 WAL 残留。
- **删除高风险**：前端"清空当前库"必须二次确认；删文件不可逆。
- **测试注入路径不受影响**：诉求3 通过**原地改写 `deps` 字段**实现热切，不改 `PipelineDeps` dataclass 定义，`create_app` 的直接依赖注入（测试用 fakes）行为不变。

## 4. 关键事实（实现前已核验，附行号）

**诉求1（去重）相关：**

- 一个候选是 `module2/models.py::ScoredCandidate`，逻辑键为 `(trajectory_id, slice_index, sub_problem_id)`。
- 同一 slice 跨子问题时：`embedding` / `bm25_tokens` **相同**（都源自 `hit.signature`，见 `rerank.py:48-49`）；`relevance_score` / `loss_mask_spans` / `evidence_step` / `criteria_hit` **各子问题不同**（来自 per-sub_problem 的 judge 结果）。→ 合并必须"并集掩码"，且**排序分要能重建**（见 5.1）。
- `deduplicate`（`dedup.py:37-74`）现仅在**同一 sub_problem_id 内**去近重（`_cosine>0.95` 或 `minhash.jaccard≥0.9`，第 56-67 行的 `sub_problem_id ==` 守卫）。
- `select_set`（`selection.py:34-113`）按子问题选：`counts`/`cover` 以单个 `candidate.sub_problem_id` 记账（第 68-72、107-110 行），`min_per_problem` 保底阶段按 `sub_id` 建池（第 51-73 行）。
- **`build_inspector_view`（`viewmodel.py:39-83`）的 hit 列表来自 `scored`（per-sub_problem `ScoredCandidate`），不是来自 `targeted`。** 每个 sub_hit 的 `evidence_step`/`criteria_hit` 直接取自对应的 `ScoredCandidate`（第 80-81 行），**已天然按子问题正确对应**。`targeted` 仅贡献 `selected` 布尔标志（第 39-42、76 行的 `selected_keys`）。→ **`MergedCandidate` 无需携带按子问题的证据字典**；viewmodel 唯一要改的是 `selected_keys` 的键。
- **`targeted` 的下游消费者共 4 处**，改 `MergedCandidate` 后都受影响：
  - `viewmodel.py:40` — 读 `c.sub_problem_id` 建 `selected_keys`（本 spec 要改）。
  - `compose.py:46` — `targeted_count = len(targeted)`（口径不变，值随之正确）。
  - `scripts/e2e_smoke.py:250-255` — 遍历 `targeted` 打印 `c.sub_problem_id`（**会 `AttributeError`**，须改为 `c.sub_problem_ids`）。
  - `scripts/complex_smoke_report.py:101` + `_render_report` — 只读 `c.trajectory_id`，**不受影响**。

**诉求3（库管理）相关：**

- `app.py` 端点与 `orchestrator` 都在**请求时**读 `deps.problem_store` 等字段（非在 `build_app` 时捕获局部变量），因此**原地改写 `deps` 字段**即可让热切生效，无需改 `create_app` 签名。
- `pipeline` 的 slice_store 经 `store_factory=lambda: slice_store`（`inspector_serve.py:73`）注入。`store_factory` 只在 `run_scored`/`run` 的 `_reset_store()` 时调用（`pipeline.py:70-71`）；`search`/`ingest` 路径**不 reset**，直接用 `self._store`。→ 热切后必须让 `pipeline._store` 也指向新库的 slice_store（不能只换 factory），见 7.1。

## 5. 诉求 1：跨子问题去重 + 合并掩码

### 5.1 新增 `MergedCandidate`（`module3/merge.py`，新文件）

"多归属候选"：把共享 `(trajectory_id, slice_index)` 的多个 `ScoredCandidate` 塌缩成一个。`@dataclass`，字段与构造规则如下（`selection.py`/`dedup.py`/`compose.py` 通过 duck-typing 读，故字段名须与 `ScoredCandidate` 尽量兼容）：

| 字段 | 类型 | 构造规则 |
|------|------|----------|
| `trajectory_id` | `str` | 合并键，相同 |
| `slice_index` | `int` | 合并键，相同 |
| `trajectory_path` | `str` | 取任一（相同） |
| `sub_problem_ids` | `list[str]` | 各成员 `sub_problem_id` 去重并集，**保持首次出现顺序** |
| `relevance_by_problem` | `dict[str, float]` | `{sub_problem_id: relevance_score}`；同一 slice 同一子问题若出现多次取 `max` |
| `relevance_score` | `float` | `max(relevance_by_problem.values())`——供全局排序与 `_diversity_gain` |
| `loss_mask_spans` | `list[dict]` | 各成员 spans 的**并集**，按 `(start_step, end_step)` 去重后升序排列（见 5.2 规范化） |
| `embedding` | `list[float]` | 取任一（相同） |
| `bm25_tokens` | `list[str]` | 取任一（相同） |
| `capability` | `list[str]` | 各成员 capability 去重并集 |
| `judge_match` | `bool` | 恒 `True`（进入合并的都是 trainable，见 5.2 过滤在前） |

**不携带** `evidence_step` / `criteria_hit` / `evidence_by_problem`：证据由 viewmodel 从 `scored` 侧读取（见 §4 已核验事实、5.3），`MergedCandidate` 只服务于"选择 + 计数 + 显示已入选标志"。

### 5.2 module3 流水线改造（`module3/pipeline.py`）

`select_final_dataset` 顺序改为（合并置于 dedup 之前，dedup/select 全程以 `MergedCandidate` 为单位）：

```
trainable(过滤 judge_match & 非空 spans)
  → merge_by_slice        # 新：按 (traj, slice) 塌缩，N 的语义从此=唯一 slice 数
  → deduplicate           # 近重塌缩，改为多归属语义
  → select_set            # 覆盖度/配额多归属记账
  → compose_dataset       # targeted_count = len(targeted) 自然正确
```

**(a) `merge_by_slice(candidates) -> list[MergedCandidate]`**（`module3/merge.py` 新函数）
按 `(trajectory_id, slice_index)` 分组，按 5.1 规则塌缩。span 并集的规范化：

```python
seen = {}                      # (start_step, end_step) -> span dict
for c in group:
    for s in c.loss_mask_spans:
        seen[(s["start_step"], s["end_step"])] = s
merged_spans = [seen[k] for k in sorted(seen)]
```

**(b) `selection.py` 覆盖度/配额多归属化**——核心改动。把所有对单个 `candidate.sub_problem_id` 的读写，改为遍历 `candidate.sub_problem_ids`：

- **min_per_problem 保底池**（现 51-73 行）：一个 slice 若同时属于 `p1`/`p2`，选它后 `p1` 与 `p2` 的 `counts` **同时 +1**。因此保底阶段遍历 `sub_problem_ids` 时要跳过"该候选已被选入 `chosen`"（用一个 `chosen_keys: set[(traj,slice)]` 去重，避免同一 slice 因属于多个子问题被重复 append）。
- **`cap_per_problem` 检查**（现 62-64、79-82 行）：候选被 cap 挡下，当且仅当它的**任一** `sub_problem_id` 已达 cap（保守：命中即挡）。
- **`counts`/`cover` 记账**（现 68-72、107-110 行）：选中一个候选后，对其 `sub_problem_ids` 中**每个** `sub_id` 都更新 `counts[sub_id]+=1`、`cover[sub_id]+=relevance_by_problem[sub_id]`。
- **`coverage_gain`**（现 85-96 行）：改为对候选的每个 `sub_id` 分别算增益后**求和**：
  ```python
  gain = 0.0
  for sid in cand.sub_problem_ids:
      cur = cover.get(sid, 0.0)
      r = cand.relevance_by_problem[sid]
      if config.coverage_cap_per_problem is None:
          gain += r
      else:
          gain += max(0.0, min(cur + r, config.coverage_cap_per_problem) - cur)
  gain += config.lam * _diversity_gain(vec, chosen_vecs)
  ```
- `_diversity_gain` / `_vector` 不变（`embedding` 是单一向量）。

**(c) `dedup.py` 近重塌缩吸收归属**（现 52-72 行）：
- 相似判定的守卫从 `candidate.sub_problem_id == kept.sub_problem_id` 改为 `set(candidate.sub_problem_ids) & set(kept.sub_problem_ids)`（归属集合有交集才算候选去重范围）。
- 被判为近重而丢弃时，**不是简单 `continue`**：要把被丢弃者的 `sub_problem_ids` / `relevance_by_problem` / `loss_mask_spans` **并入幸存者**（否则丢覆盖）。实现上给幸存者做一次"吸收"合并（复用 5.1 的并集规则）。`relevance_score` 取并后 max。

**(d) 预算口径**（`orchestrator.py:245`、`316`）：`SelectionConfig(n=min(10, max(1, len(scored))))` 的 `scored` 是合并**前**数量。改为先算合并后唯一 slice 数 `n_unique`，用 `n=min(10, max(1, n_unique))`。可在 orchestrator 调 `merge_by_slice` 前置计算，或让 `select_final_dataset` 接受"合并后再定 n"的约定——**定为后者**：`select_final_dataset` 内部合并后，若传入的 `selection.n` 超过合并后候选数，`select_set` 现有 `budget=min(n, len(candidates))` 已兜底，故 orchestrator 只需把 `len(scored)` 换成"去重 slice 数"避免 n 虚高即可（非硬性，但对齐语义）。

### 5.3 viewmodel 改造（`viewmodel.py`，仅一处）

- `selected_keys`（第 39-42 行）从 `(c.trajectory_id, c.slice_index, c.sub_problem_id)` 改为 `(c.trajectory_id, c.slice_index)`——因为 `MergedCandidate` 无单一 `sub_problem_id`。相应地第 76 行的 `selected` 判定改为 `(c.trajectory_id, c.slice_index) in selected_keys`。
- **其余不动**：hit 列表仍来自 `scored`，`evidence_step`/`criteria_hit` 仍从 `ScoredCandidate` 读（§4 已核验，天然按子问题正确）。
- 效果：同一 slice 在它命中的所有子问题卡片下都标"已入选"，且"最终入选 N 条"（`manifest.targeted_count`）= 唯一 slice 数。

### 5.4 受影响测试（逐文件）

| 文件 | 现状 | 处理 |
|------|------|------|
| `tests/module3/conftest.py` | `mk_candidate` 造 `ScoredCandidate`-like，只有单 `sub` | 新增 `mk_merged` fixture 或让 `merge_by_slice` 兜住单归属；保留 `mk_candidate` 供 rerank/合并前用 |
| `tests/module3/test_dedup.py:58` `test_same_slice_kept_when_it_covers_different_subproblems` | **断言当前行为**：同一 slice 两子问题**保留为 2 条** | **本 spec 故意推翻**：合并后应为 1 条多归属候选。改断言为"合并成 1 条且 `sub_problem_ids=={p1,p2}`"（去重前会被 `merge_by_slice` 先合并，此测试语义要重写到合并层） |
| `tests/module3/test_selection.py` | 断言 `c.sub_problem_id`（如第 28-29、39 行） | 改为读 `c.sub_problem_ids`；覆盖度/配额用例按多归属记账重算期望 |
| `tests/module3/test_pipeline.py` | `c.trajectory_id` 计数（第 21-22 行） | 计数口径不变（本就按 traj），但要新增多归属计数=唯一数的用例 |
| `tests/module3/test_compose.py` | `targeted_count == len` | 不变 |
| `tests/service/test_viewmodel.py` | 依赖 `select_result.targeted` 的 `sub_problem_id` | 改为 `MergedCandidate`-like 输入，验证 `selected` 跨卡片一致 |
| `scripts/e2e_smoke.py:253` | 打印 `c.sub_problem_id` | 改为 `c.sub_problem_ids`（避免 `AttributeError`） |

**新增用例**：① 两子问题各选中同一 `(traj,slice)` → `targeted_count==1`；② 合并后 `loss_mask_spans` 为并集且有序去重；③ viewmodel 下该 slice 在两个子问题卡片下 `selected` 均为 `True`；④ `min_per_problem=1` 且 `p1`/`p2` 只有一个共享 slice 时，选 1 条即同时满足两个保底。

## 6. 诉求 2：证据折叠（行内 accordion）

纯前端，改 `service/web/app.js` + `style.css`，`index.html` 不动。

- `renderHits()`（app.js:531-542）：不再内联渲染证据行。命中卡上放一个 `证据` 小按钮，**仅当** `evidence_step != null` 或 `criteria_hit` 非空时显示。
- 点击按钮 → `e.stopPropagation()`（防止误触 `selectTrajectory`）→ 在该卡片下方就地展开/收起证据面板。
- 面板内容：决定性证据步 + 命中判据；多归属（同一 slice 命中多个子问题）时**按子问题分组**列出，数据来自 5.3 保留的按子问题证据。
- 详情栏（`renderDetail`）里 `⭐ 决定性证据` 步标星逻辑保持不变。

## 7. 诉求 3：SQLite 库管理（热切 + 清除）

### 7.1 新增 `service/database.py` 的 `DatabaseManager`

单一职责：持有"当前库路径 + 4 个 store 实例 + pipeline 引用"，提供列举/切换/清除。所有 4 个 Sqlite store（`SqliteSliceStore`、`SqliteProblemStore`、`SqliteTrajectoryStore`、`SqliteJudgeCache`）都已有 `close()`（各自 `self._conn.close()`）。

**构造与状态：**

```python
class DatabaseManager:
    def __init__(self, db_path, deps, pipeline, *, run_lock):
        self._path = pathlib.Path(db_path)
        self._deps = deps            # 原地改写其 problem_store/trajectory_store/judge_cache
        self._pipeline = pipeline    # 需同时改写 pipeline._store（search/ingest 不 reset）
        self._run_lock = run_lock    # 与 app._run_lock 同一把
        self.slice_store = None      # store_factory 读它
        self._open(self._path)       # 建 4 个 store，绑定 deps + slice_store + pipeline._store
```

- **接线（`inspector_serve.py`）**：`store_factory = lambda: mgr.slice_store`（不再捕获固定 `slice_store` 局部量）。`DatabaseManager` 在 `deps` 与 `pipeline` 构造后创建，`_open` 负责把 4 个 store 装配进 `deps` 与 `pipeline._store`。因此 `build_app` 需能拿到 `_run_lock` —— 见 7.5。

| 方法 | 行为 |
|------|------|
| `current() -> Path` | 返回 `self._path` |
| `list_databases() -> list[dict]` | 扫描 `self._path.parent` 下所有 `*.db`（跳过 `-wal`/`-shm`），每项 `{name, path, size_bytes, problems, trajectories, is_current}`。计数：对当前库直接读已开的 store；对其它库开一个**临时只读连接** `SELECT COUNT(*)` 后即关，异常则计数填 `null`（损坏库不该让整个列举失败） |
| `switch(path)` | `_close_all()` → `_open(path)`（原地改 `deps` 三字段 + `slice_store` + `pipeline._store`）→ 更新 `self._path`。路径不存在时 `_connect` 自动建表=新建空库 |
| `clear_current()` | 记住 `p=self._path` → `_close_all()` → 删 `p`、`p+"-wal"`、`p+"-shm"`（`missing_ok=True`）→ `_open(p)` 重建空库 |
| `_open(path)` (私有) | 建 4 store；`self._deps.problem_store/trajectory_store/judge_cache = ...`；`self.slice_store = SqliteSliceStore(path)`；`self._pipeline._store = self.slice_store` |
| `_close_all()` (私有) | 对 4 个 store 逐个 `close()`（吞 `close()` 异常，保证删文件前句柄尽量释放） |

- **锁的持有者是端点，不是 manager**：`switch`/`clear_current` 假定调用时锁已被占用检查通过（见 7.2），自身不抢锁——避免与端点重复加锁死锁。manager 方法是同步的（纯 SQLite 连接操作，无 await）。
- **WAL 删除注意**：删 `.db` 前已 `_close_all()`，`close()` 会触发 SQLite 关闭时的 WAL checkpoint；随后显式删 `-wal`/`-shm` 兜底残留。

### 7.2 新增 HTTP 端点（app.py）

```
GET  /databases          → { current: "<path>", databases: [ {name,path,size_bytes,problems,trajectories,is_current}, ... ] }
POST /databases/switch   { "path": "<path>" }  → 200 {current,...} / 409 有任务运行中 / 400 路径非法
POST /databases/clear    (无 body)             → 200 {current,...} / 409 有任务运行中
```

- 端点从 `deps`（或闭包）拿到 `DatabaseManager` 实例。若 `deps` 无 manager（测试用 fakes、老接线）→ 三个端点返回 501/404，不崩。
- **运行中拒绝**：`switch`/`clear` 先判 `_run_lock.locked()`；被占用 → `raise HTTPException(409, "有任务运行中，无法切换/清除库")`。**不要 `async with _run_lock`**——那会排队等待而非拒绝，与"绝不 mid-run 切换"的护栏相悖。
- **切换即清运行态**：切库后旧 run 的 view/trajectory 属于旧库语境，端点成功后前端须重置工作区（见 7.3）；后端 `MemoryRunStore` 无需清（run_id 仍可查，只是数据来自旧库，可接受）。
- `switch` 的 `path` 校验：拒绝空串；相对路径按 `current().parent` 解析或按 CWD——**定为**：非绝对路径时相对 `current().parent`（同目录新建库最常见）。
- 均为 `async def`（与既有 store-touching 端点一致，避免 SQLite 跨线程问题）；manager 方法本身同步，在事件循环线程内直接调用。

### 7.3 前端（`index.html` header + `app.js` + `style.css`）

- **DOM**（`index.html`）：在 `#stats-bar` 同一块（`.stats-bar` 旁）加一个 `.db-bar`，含：`<select id="db-select">`（列出库）、`<input id="db-new-path">`+`<button id="db-switch">`（新建/切换到自定义路径）、`<button id="db-clear">🗑 清空当前库</button>`。
- **`app.js` 新增**：`loadDatabases()`（`GET /databases` → 填充 select，当前库设为 selected 且加 `· 当前` 后缀）；`switchDatabase(path)`（`POST /databases/switch`）；`clearDatabase()`（先 `confirm("将永久删除当前库文件，不可恢复，确认？")` → `POST /databases/clear`）。
- `#db-select` 的 `change` → `switchDatabase(value)`；`#db-switch` → 用 `#db-new-path` 值 switch；`#db-clear` → `clearDatabase()`。
- **成功后**：`loadStats()` + `loadDatabases()` + 重置工作区（`$("workspace").classList.add("hidden")`、`hideDedupBanner()`、清 `state.view`/`state.trajCache`）。
- **409 处理**：`resp.status === 409` → `alert("有任务运行中，无法切换/清除库")`，不改状态。
- 页面加载时 `loadDatabases()` 与现有 `loadStats()` 一起调。
- 样式复用现有 header 风格；`confirm()` 作为二次确认（无需自建 modal，删除是低频操作）。

### 7.4 安全护栏（落实到实现）

- 删文件前 `_close_all()` 释放句柄（7.1）。
- 前端删除强制 `confirm()` 二次确认（删文件不可逆，高风险）。
- 后端 `switch`/`clear` 运行中返回 409（判 `_run_lock.locked()`，不排队）。
- `list_databases` 对损坏/无法打开的库计数填 `null` 而非抛错。

### 7.5 接线改动（`inspector_serve.py` + `app.py`）

现状 `_run_lock` 是 `create_app` 内的局部量（`app.py:52`），`DatabaseManager` 与端点都要用到它。改动方案（择一，实现时定）：

- **方案A（推荐）**：`create_app` 内构造 `DatabaseManager`（需要 `deps` 已带 pipeline 引用），把 `_run_lock` 传给它；三个 `/databases*` 端点定义在 `create_app` 内，直接闭包访问 `_run_lock` 与 manager。`inspector_serve.py` 仅需把 `store_factory` 改成 `lambda: <manager>.slice_store`——但 manager 在 `create_app` 内才建，故 pipeline 的 `store_factory` 需延迟绑定：**改为** `deps` 增加一个可选 `db_manager` 字段（默认 None），`create_app` 若发现 `deps.db_manager is None` 且具备构造条件则就地建之，并回填 `pipeline` 的 store 指向。
- **方案B**：在 `inspector_serve.py` 构造 `DatabaseManager`（此处能同时拿到 deps/pipeline），把 manager 注入 `create_app(db_manager=...)`；`_run_lock` 则由 `create_app` 创建后回注给 manager（`manager.attach_lock(lock)`）。

**定为方案B**：接线集中在 `inspector_serve.py`，`create_app` 只多一个 `db_manager=None` 可选参数并在建锁后 `db_manager.attach_lock(_run_lock)`。测试不传 `db_manager` → `/databases*` 端点返回 501，其余行为不变。`PipelineDeps` dataclass **不改**。

## 8. 非目标 / 后续增强

- 库的多用户并发隔离、权限控制（V1 单用户本地服务）。
- 库之间的数据迁移/合并。
- `min_turns` 等其它选择配置的调参（沿用现值）。
- 通用数据（general data）加载仍未实现（`compose_dataset` 现状保留）。

## 9. 实现顺序建议

1. 诉求2（纯前端、零风险、独立）。
2. 诉求1（module3 + viewmodel + 测试；数据层核心改动）。
3. 诉求3（新增 database 管理 + 端点 + 前端 + 接线）。

三块彼此独立，可分别成 PR。

## 10. 验收标准（每块的"完成"定义）

**诉求1：**
- [ ] 两个子问题各选中同一 `(traj, slice)` → `manifest.targeted_count == 1`，`out["targeted"]` 长度为 1，该项 `sub_problem_ids` 含两个子问题。
- [ ] 合并项 `loss_mask_spans` 是各子问题 spans 的去重并集且升序。
- [ ] viewmodel 下该 slice 在两个子问题卡片下 `selected` 均为 `True`。
- [ ] `min_per_problem=1`、`p1`/`p2` 仅共享一个 slice 时，选中该 1 条即同时满足两个保底。
- [ ] `scripts/e2e_smoke.py` 不再 `AttributeError`（改用 `sub_problem_ids`）。
- [ ] `test_dedup.py:58` 已按新语义重写；`uv run pytest -m "not requires_model" tests/module3 tests/service/test_viewmodel.py` 全绿。

**诉求2：**
- [ ] 命中卡默认不显示证据文本；有 evidence/criteria 时出现"证据"按钮。
- [ ] 点按钮就地展开/收起，不触发选中轨迹（`stopPropagation` 生效）。
- [ ] 多归属 slice 的证据按子问题分组显示。
- [ ] `tests/service/test_inspector_frontend_static.py` 若断言 DOM 结构需同步。

**诉求3：**
- [ ] `GET /databases` 列出目录下所有 `.db` 并标记当前库。
- [ ] 切到不存在路径 → 自动建空库并生效（stats 归零）。
- [ ] 切到已有库 → search/ingest 立即对新库生效（验证 `pipeline._store` 已换）。
- [ ] 有任务运行中调 switch/clear → 409，运行不受影响。
- [ ] clear 删文件后重建空库，`-wal`/`-shm` 不残留。
- [ ] 不传 `db_manager` 的现有测试（`tests/service/test_app*.py`）全绿，行为不变。
